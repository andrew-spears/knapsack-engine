"""
Parallel data generation. Saves to .npz for later training.

Usage:
    python run_datagen.py                              # 5000 games, half of cores
    python run_datagen.py --games 50000 --workers 64   # big run
    python run_datagen.py --depth 3 --fanout 20        # deeper search
    python run_datagen.py --leaf-model value_net.pt    # NN leaf evaluation
    python run_datagen.py --out data/run1.npz          # custom output path

Output .npz contains:
    stashed:   (N, num_types) int array
    remaining: (N, num_types) int array
    values:    (N,) float array
"""

import argparse
import multiprocessing as mp
import os
import time
import numpy as np
import torch
from game import GameConfig, sample_transitions, play_games_batched
from engine import Engine, expand_to_leaves
from model import make_leaf_fn, make_array_leaf_fn, load_model

# Prevent fork deadlock with PyTorch's internal thread pools
mp.set_start_method("spawn", force=True)


# --- Batched worker (--leaf-model): array-based expansion, lockstep across games ---

def worker_batched(args):
    worker_id, num_games, depth, fanout, leaf_model_path = args
    torch.set_num_threads(1)
    config = GameConfig.small()
    model = load_model(leaf_model_path, config)
    array_leaf_fn = make_array_leaf_fn(model, config)
    engine = Engine(depth, fanout, config, array_leaf_fn=array_leaf_fn)
    B = config.num_bundles
    T = config.num_types

    # Cap concurrent games to limit memory: leaves grow as (fanout*num_bundles)^depth
    leaves_per_game = B * (fanout * B) ** depth
    bytes_per_leaf = T * 8 * 2  # stashed + remaining arrays
    max_memory = 500 * 1024 * 1024  # 500 MB per worker
    chunk_size = max(1, min(num_games, max_memory // (leaves_per_game * bytes_per_leaf)))

    all_stashed, all_remaining, all_values = [], [], []
    total_leaves = 0
    games_done = 0

    for chunk_start in range(0, num_games, chunk_size):
        n = min(chunk_size, num_games - chunk_start)
        result = play_games_batched(n, engine, config, collect_data=True)
        all_stashed.append(result["stashed"])
        all_remaining.append(result["remaining"])
        all_values.append(result["values"])
        total_leaves += result["total_leaves"]

        games_done += n
        if games_done % max(1, num_games // 10) < chunk_size or games_done == num_games:
            total_samples = sum(len(v) for v in all_values)
            print(f"  Worker {worker_id}: {games_done}/{num_games} games, "
                  f"{total_samples} samples, {total_leaves:,} leaves")

    return (np.concatenate(all_stashed), np.concatenate(all_remaining),
            np.concatenate(all_values), total_leaves)


# --- Sequential worker (no --leaf-model): fast numba search ---

def worker_sequential(args):
    worker_id, num_games, depth, fanout, _ = args
    config = GameConfig.small()
    engine = Engine(depth, fanout, config)
    T = config.num_types

    # Pre-allocate with estimated size (num_games * num_rounds * num_bundles)
    est_samples = num_games * config.num_rounds * config.num_bundles
    stashed_arr = np.empty((est_samples, T), dtype=np.int32)
    remaining_arr = np.empty((est_samples, T), dtype=np.int32)
    values_arr = np.empty(est_samples, dtype=np.float32)
    count = 0

    for game_i in range(num_games):
        if (game_i + 1) % max(1, num_games // 5) == 0:
            print(f"  Worker {worker_id}: game {game_i + 1}/{num_games}")

        stashed = config.init_stashed
        remaining = config.init_pool

        for _round in range(config.num_rounds):
            if sum(remaining) < config.draw_size:
                break

            transitions, _, _ = sample_transitions(stashed, remaining, config)

            best_value = -1e9
            best_idx = 0
            for idx, (s, r) in enumerate(transitions):
                v = engine.search_value(s, r)
                stashed_arr[count] = s
                remaining_arr[count] = r
                values_arr[count] = v
                count += 1
                if v > best_value:
                    best_value = v
                    best_idx = idx

            stashed, remaining = transitions[best_idx]

    return stashed_arr[:count], remaining_arr[:count], values_arr[:count], engine.node_count


def main():
    p = argparse.ArgumentParser(description="Generate training data with parallel search")
    p.add_argument("--games", type=int, default=5000)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--fanout", type=int, default=10)
    p.add_argument("--workers", type=int, default=None, help="Defaults to half of cpu count")
    p.add_argument("--leaf-model", type=str, default=None, help="Model .pt file for NN leaf evaluation")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    config = GameConfig.small()
    if args.out is None:
        samples_est = args.games * config.num_rounds * config.num_bundles
        args.out = f"data/d{args.depth}_f{args.fanout}_{samples_est // 1000}k.npz"
    n_workers = args.workers or mp.cpu_count() // 2
    games_per_worker = args.games // n_workers
    remainder = args.games % n_workers

    print(f"Generating {args.games} games across {n_workers} workers")
    print(f"Search: depth={args.depth}, fanout={args.fanout}")
    if args.leaf_model:
        print(f"Leaf model: {args.leaf_model}")

    # Warm up numba and array-based expansion
    print("Compiling numba functions...")
    engine = Engine(args.depth, args.fanout, config)
    engine.search_value(config.init_stashed, config.init_pool)
    dummy_s = np.zeros((1, config.num_types), dtype=np.int64)
    dummy_r = np.array([config.init_pool], dtype=np.int64)
    expand_to_leaves(dummy_s, dummy_r, args.depth, args.fanout, config)

    # Benchmark with the path that will actually run
    t_bench = time.time()
    n_bench = 20
    if args.leaf_model:
        bench_model = load_model(args.leaf_model, config)
        bench_leaf_fn = make_leaf_fn(bench_model, config)
        bench_engine = Engine(args.depth, args.fanout, config, leaf_fn=bench_leaf_fn)
        for _ in range(n_bench):
            bench_engine.search_value(config.init_stashed, config.init_pool)
    else:
        for _ in range(n_bench):
            engine.search_value(config.init_stashed, config.init_pool)
    secs_per_call = (time.time() - t_bench) / n_bench

    rounds = config.num_rounds
    calls_per_game = rounds * config.num_bundles
    total_calls = args.games * calls_per_game
    est_serial = total_calls * secs_per_call
    est_parallel = est_serial / n_workers
    print(f"Benchmark: {secs_per_call*1000:.1f}ms/search call")
    print(f"Estimate: {args.games} games x ~{rounds} rounds x {config.num_bundles} bundles "
          f"= ~{total_calls:,} search calls")
    print(f"Estimated time: ~{est_parallel:.0f}s ({est_parallel/60:.1f}min) "
          f"across {n_workers} workers")
    if args.leaf_model:
        print(f"  (batched mode will be faster than this estimate)")

    worker_fn = worker_batched if args.leaf_model else worker_sequential
    worker_args = []
    for i in range(n_workers):
        n = games_per_worker + (1 if i < remainder else 0)
        if n > 0:
            worker_args.append((i, n, args.depth, args.fanout, args.leaf_model))

    t0 = time.time()
    with mp.Pool(len(worker_args)) as pool:
        results = pool.map(worker_fn, worker_args)
    elapsed = time.time() - t0

    # Merge results (already numpy arrays now)
    all_stashed = np.concatenate([r[0] for r in results])
    all_remaining = np.concatenate([r[1] for r in results])
    all_values = np.concatenate([r[2] for r in results])
    total_nodes = sum(r[3] for r in results)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, stashed=all_stashed, remaining=all_remaining, values=all_values)

    print(f"\nSaved {len(all_values)} samples to {args.out}")
    print(f"Time: {elapsed:.1f}s ({args.games / elapsed:.0f} games/s)")
    print(f"Nodes/leaves evaluated: {total_nodes:,}")
    print(f"Value range: [{all_values.min():.1f}, {all_values.max():.1f}], "
          f"mean: {all_values.mean():.1f}")


if __name__ == "__main__":
    main()
