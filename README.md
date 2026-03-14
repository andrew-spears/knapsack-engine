# Draft Game Engine

A configurable engine for solving single-player drafting (e.g. drafting cards/athletes) games using expectimax search.

## The game

A **pool** contains known quantities of goods across several types. Each round:

1. A batch of goods is drawn randomly from the pool.
2. Each drawn good is randomly assigned to some number of **bundles** (with overlap — the same good can appear in multiple bundles).
3. The player picks one bundle to keep. All drawn goods are removed from the pool regardless.
4. Repeat until the pool is exhausted.

The player's score is determined by the counts of each type they've collected, evaluated by a configurable scoring function.

## Example

An example game playthrough from `demo.py`:

```
GAME CONFIG:
=================================
Num types: 5
Pool (num goods of each type):
  Type 1: 8
  Type 2: 7
  Type 3: 6
  Type 4: 5
  Type 5: 4
Goods drawn per round: 5
Bundles per draw: 4
Overlap (number of bundles each good appears in): 2
Rounds: 6
=================================

SCORING RULE:
=================================
def power_two_score(type_value, count):
    '''Default scoring: 2 pays face value, 4 pays double, others pay negative.'''
    if count == 0:
        return 0
    if count == 2:
        return 2 * type_value
    if count == 4:
        return 8 * type_value
    return -type_value * count
=================================

Playing game with engine search (depth=3, fanout=10)...
  Round 1: drew [4 4 1 1 3]
           bundle 0: 4 4 1 3 >>>
           bundle 1: 1
           bundle 2: 1 3
           bundle 3: 4 4 1
  Stash:     [ 1   0   1   2   0]
  Remaining: [ 6   7   5   3   4]

  Round 2: drew [2 5 1 4 2]
           bundle 0: 5 1 4 2 >>>
           bundle 1: 1
           bundle 2: 2
           bundle 3: 2 5 4 2
  Stash:     [ 2   1   1   3   1]
  Remaining: [ 5   5   5   2   3]

  Round 3: drew [1 1 4 2 1]
           bundle 0: 1 >>>
           bundle 1: 2 1
           bundle 2: 1 1 4 2 1
           bundle 3: 1 4
  Stash:     [ 3   1   1   3   1]
  Remaining: [ 2   4   5   1   3]

  Round 4: drew [3 1 2 5 2]
           bundle 0: 1 2 5 2 >>>
           bundle 1: 3 1 2 5
           bundle 2: 2
           bundle 3: 3
  Stash:     [ 4   3   1   3   2]
  Remaining: [ 1   2   4   1   2]

  Round 5: drew [5 3 2 2 5]
           bundle 0: 3 2 5
           bundle 1: 2
           bundle 2: 5 3 2
           bundle 3: 5 2 5 >>>
  Stash:     [ 4   4   1   3   4]
  Remaining: [ 1   0   3   1   0]

  Round 6: drew [3 1 3 4 3]
           bundle 0: 1
           bundle 1: 3 1 3 4 3 >>>
           bundle 2: 3
           bundle 3: 3 4 3
  Stash:     [ 5   4   4   4   4]
  Remaining: [ 0   0   0   0   0]

  Score: 107
  Nodes searched: 774564, avg 129094.0 per move
```

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

The goal of the engine is to pick the best action (bundle) given the current stash and remaining pool. This reduces to evaluating the value of a given state (stash + remaining) and picking the bundle that leads to the best expected value after the next random draw.
We do this using expectimax search, which expands a game tree alternating between chance nodes (random draws, averaged over Monte Carlo samples) and decision nodes (pick best bundle). `depth` controls lookahead rounds, `fanout` controls Monte Carlo samples per chance node. At the leaves of this tree, we use a value function to estimate the expected final score from that state. This can be a simple heuristic or a learned neural network. We then propagate these values up the tree to make the best decision at the root.

`Engine` has two search methods:

- **`search_value(stashed, remaining)`** — single root state. Uses numba `_search` to traverse the tree with a simple heuristic function, or Python `expand_tree` + `leaf_fn` to expand all leaves, evaluate the leaf function (e.g. a neural network) on all at once, then propagate values up.
- **`search_value_batch(root_s, root_r)`** — many roots at once. Uses numba `expand_to_leaves` + `array_leaf_fn`. Much faster for data generation and benchmarking.

## Neural network training

A fairly good initial approach is to use search with the actual score function as our leaf value function. To improve this, we need a better value function on the leaves. To this end, we can train a neural network to predict the search value of states. This effectively distills the search into a fast-to-evaluate function, which can then be used as the leaf function to emulate a deeper search at very little extra computation cost. Extrapolating this, we can run an expert iteration loop: search generates training data in batches, NN learns to predict search values, NN becomes the leaf evaluator for better search.

### Basic workflow

**Generate data with search:**

`run_datagen.py --games {games} --depth {depth} --fanout {fanout} --leaf-model [model.pt]`

Generates `games` games in parallel with the given search parameters, saves states and search values to .npz files as supervised training examples. If `--leaf-model` is provided, uses that NN as the leaf function instead of the default heuristic.

**Train NN on generated data:**

`run_train.py [data.npz] --epochs {epochs} --hidden {hidden_size} --layers {num_layers}`

Trains a model with the given architecture on the provided data, saves model to .pt file.

### File naming convention

Data lives in `data/`, models in `models/`. Files encode search parameters:

- Data: `data/d{depth}_f{fanout}_{num_samples}.npz` (e.g. `data/d2_f10_240k.npz`)
- Models: derived from data file, e.g. `data/d2_f10_240k.npz -> models/d2_f10_240k_h128_l4_e50.pt`

## Future ideas

- **Cross-type combo scoring**: Replace per-type score table with pattern-based scoring. A combo is a `(pattern_vector, points)` pair — score = `min(stashed // pattern)` over nonzero entries, times points. e.g. "5 points for each complete set of types 2,3,7" → pattern `[0,1,1,0,0,0,1,0,0,0]`, points=5. Can also encode current count-based scoring: "pairs of type 3" → pattern `[0,0,2,0,...]`. Precompute patterns into a 2D array, score in numba with one min-reduction per pattern.
