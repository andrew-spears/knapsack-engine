import argparse
import numpy as np
import time
from game import GameConfig, play_game, play_games_batched
from engine import Engine
from model import load_model, make_leaf_fn, make_array_leaf_fn, greedy_nn_action


config = GameConfig.small()


def run_benchmark(name, n, action_fn=None, eng=None, batched=False):
    start = time.time()
    if batched:
        result = play_games_batched(n, eng, config)
        scores = result["scores"]
        nodes = result["total_leaves"]
    else:
        scores = [play_game(config, action_fn) for _ in range(n)]
        nodes = eng.node_count if eng else 0
    elapsed = time.time() - start

    nodes_per_move = nodes / n / config.num_rounds if nodes else 0
    us_per_node = elapsed / nodes * 1e6 if nodes else 0

    p10, p50, p90 = np.percentile(scores, [10, 50, 90])
    stats = f"{np.mean(scores):6.1f}  {np.std(scores):5.1f}  {p10:6.1f}  {p50:6.1f}  {p90:6.1f}  {elapsed/n:7.3f}s"
    node_stats = f"{nodes_per_move:12.0f}  {us_per_node:7.1f}" if nodes else f"{'--':>12s}  {'--':>8s}"
    print(f"{name:>30s}  {n:5d}  {stats}  {node_stats}")


def main():
    p = argparse.ArgumentParser(description="Benchmark search strategies")
    p.add_argument("--model", type=str, default=None, help="Model .pt file for NN leaf evaluation")
    p.add_argument("--fanout", type=int, default=10, help="Fanout for search (default: 10)")
    p.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3], help="Depths to test (default: 1 2 3)")
    p.add_argument("--trials", type=int, default=50, help="Games per config (default: 50)")
    p.add_argument("--baseline-trials", type=int, default=1000, help="Games for baselines (default: 1000)")
    args = p.parse_args()

    model = None
    if args.model:
        model = load_model(args.model, config)

    print(f"Config: {config.num_types} types, {config.num_bundles} bundles, "
          f"draw={config.draw_size}, ~{config.num_rounds} rounds/game")
    print(f"Fanout: {args.fanout}, depths: {args.depths}, "
          f"trials: {args.trials}/config, {args.baseline_trials}/baseline")
    if model:
        print(f"Model: {args.model}")
    print()

    # --- Warmup ---
    print("Warming up numba...", end=" ", flush=True)
    play_game(config, Engine(1, 5, config).get_action)
    print("done.\n")

    # --- Header ---
    print(f"{'Strategy':>30s}  {'n':>5s}  {'mean':>6s}  {'std':>5s}  {'p10':>6s}  {'p50':>6s}  {'p90':>6s}  {'s/game':>8s}  {'nodes/move':>12s}  {'us/node':>8s}")
    print("-" * 108)

    # --- Baselines ---
    run_benchmark("Random", args.baseline_trials,
                  action_fn=lambda t: np.random.randint(len(t)))

    eng = Engine(0, 0, config)
    run_benchmark("Heuristic (depth=0)", args.baseline_trials,
                  action_fn=eng.get_action, eng=eng)

    if model:
        run_benchmark("NN only (no search)", args.baseline_trials,
                      action_fn=lambda t: greedy_nn_action(model, t, config))

    # --- Heuristic leaf at each depth (batched) ---
    print()
    for depth in args.depths:
        eng = Engine(depth, args.fanout, config)
        run_benchmark(f"Heuristic d={depth}, f={args.fanout}",
                      args.trials, eng=eng, batched=True)

    # --- NN leaf at each depth (batched) ---
    if model:
        print()
        for depth in args.depths:
            array_leaf_fn = make_array_leaf_fn(model, config)
            eng = Engine(depth, args.fanout, config, array_leaf_fn=array_leaf_fn)
            run_benchmark(f"NN leaf d={depth}, f={args.fanout}",
                          args.trials, eng=eng, batched=True)


if __name__ == "__main__":
    main()
