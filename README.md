# Draft Game Engine

A configurable engine for solving single-player drafting (e.g. drafting cards/athletes) games using expectimax search.
=======
A configurable engine for solving single-player collection/knapsack-style games using expectimax search, with iterative neural network training (Expert Iteration).
>>>>>>> bc5b47f (cleaned up data gen and training loop, mdoels, eval, and more stats)

## The game

A **pool** contains known quantities of goods across several types. Each round:

1. A batch of goods is drawn randomly from the pool.
2. Each drawn good is randomly assigned to some number of **bundles** (with overlap — the same good can appear in multiple bundles).
3. The player picks one bundle to keep. All drawn goods are removed from the pool regardless.
4. Repeat until the pool is exhausted.

The player's score is determined by the counts of each type they've collected, evaluated by a configurable scoring function.

## Configuration

All game parameters live in `GameConfig`:

| Parameter        | Description                                  |
| ---------------- | -------------------------------------------- |
| `num_types`      | Number of distinct good types                |
| `init_pool`      | Tuple of initial counts per type             |
| `draw_size`      | Goods drawn from pool each round             |
| `num_bundles`    | Number of bundles to choose from             |
| `overlap_degree` | How many bundles each drawn good appears in  |
| `score_fn`       | `f(type_value, count) -> score` for one type |

`GameConfig.small()` is used for fast iteration (5 types, 4 bundles, ~6 rounds/game).

## Search engine

Expectimax search alternates chance nodes (random draws, averaged over Monte Carlo samples) and decision nodes (pick best bundle). `depth` controls lookahead rounds, `fanout` controls Monte Carlo samples per chance node.

Two search paths in `Engine`:
- **`search_value(stashed, remaining)`** — single root state. Uses numba `_search` (heuristic leaf) or Python `expand_tree` + `leaf_fn` (NN leaf).
- **`search_roots_batch(root_s, root_r)`** — many roots at once. Uses numba `expand_to_leaves` + `array_leaf_fn`. Much faster for data generation and benchmarking.

## Neural network training

Iterative Expert Iteration loop: search generates training data, NN learns to predict search values, NN becomes the leaf evaluator for better search.

### Workflow

**Train:**
```bash
python run_train.py data/d2_f10_r*.npz --output models/my_model.pt --epochs 200
# auto-detects MPS/CUDA, saves loss plot as model_loss.png
```

### File naming convention

Data lives in `data/`, models in `models/`. Files encode search parameters:
- Data: `data/d{depth}_f{fanout}_r{round}.npz` (e.g. `data/d2_f10_r3.npz`)
- Models: `models/d{depth}_f{fanout}_r{round}.pt` (e.g. `models/d2_f10_r3.pt`)
- Standalone datagen default: `data/d{depth}_f{fanout}_{samples}k.npz`

## Files

| File | Description |
| --- | --- |
| `game.py` | `GameConfig`, scoring, bundle assignment, transition sampling |
| `engine.py` | Expectimax search: numba `_search`, array-based `expand_to_leaves`, `Engine` class |
| `model.py` | `ValueNet`, leaf function factories (`make_leaf_fn`, `make_array_leaf_fn`), model save/load |
| `run_datagen.py` | Parallel data generation with multiprocessing. Uses `spawn` to avoid PyTorch fork deadlocks |
| `run_train.py` | Train from .npz files, saves loss plot, auto-detects GPU |
| `run_iterate.sh` | Full iterative training loop (datagen → train → repeat) |
| `benchmark.py` | Benchmark search strategies at various depths, with optional NN model |
| `fanout_variance.py` | Measure search value variance as function of fanout |
| `test_prediction.py` | Test NN prediction accuracy vs search values |
| `sync_remote.sh` | Push code to remote server |
| `sync_local.sh` | Pull data/models from remote server |
