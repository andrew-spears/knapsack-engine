# Draft Game Engine

A configurable engine for solving single-player drafting (e.g. drafting cards/athletes) games using expectimax search.

## The game

A **pool** contains known quantities of goods across several types. Each round:

1. A batch of goods is drawn randomly from the pool.
2. Each drawn good is randomly assigned to some number of **bundles** (with overlap — the same good can appear in multiple bundles).
3. The player picks one bundle to keep. All drawn goods are removed from the pool regardless.
4. Repeat until the pool is exhausted.

The player's score is determined by the counts of each type they've collected, evaluated by a configurable scoring function.

## Why this structure

This is a general framework for a class of problems: you have a shrinking pool of resources, partial information about what's coming, and each turn you make a constrained selection from randomly-presented options. The overlapping bundles create tension — choosing one bundle means forgoing goods that are shared with other bundles. The scoring function (which maps per-type counts to points) encodes what collection patterns are valuable.

By changing `GameConfig`, you can model different knapsack-like scenarios:

- **Set collection**: score rewards completing sets of specific sizes
- **Majority games**: score rewards having the most of a type
- **Avoidance games**: some types have negative value, bundles force you to take them alongside valuable types
- **Threshold games**: goods are worthless until you hit a count, then valuable

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

## Search engine

The engine uses **expectimax search** — alternating chance nodes (random draws, averaged over Monte Carlo samples) and decision nodes (pick best bundle). The `depth` parameter controls how many rounds to look ahead, and `fanout` controls how many random draws to sample at each chance node.

Scoring is precomputed into a 2D lookup table (`score_table[type_idx, count]`) so the entire search runs in numba-jitted code with no Python overhead.

```python
from game import GameConfig
from engine import Engine

config = GameConfig()
eng = Engine(depth=4, fanout=15, config=config)
```

## Files

- `game.py` — `GameConfig`, scoring, bundle assignment, transition sampling
- `engine.py` — Expectimax search (`_search`, numba-jitted) and `Engine` class
- `benchmark.py` — Benchmark various depth/fanout configurations
- `test_run.py` — Play through a single game with full output

## Future work

Train a neural network value function via self-play: use search to generate (state, value) training pairs, then replace the leaf heuristic with the learned model. This follows the Expert Iteration / AlphaZero pattern — search quality improves as the leaf evaluator improves, producing better training signal in turn.
