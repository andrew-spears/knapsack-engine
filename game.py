"""
Game: one player collects goods from a shared pool.

A pool of goods has known quantities per type (e.g. 10 type-0, 9 type-1, ...).
Each round, goods are drawn from the pool and each good is randomly assigned
to `overlap_degree` of the `num_bundles` bundles. The player picks one bundle
to keep; unpicked goods are discarded. Repeat until the pool is exhausted.
Score based on collected counts.
"""

import numpy as np
import numba
from dataclasses import dataclass, field
from typing import Callable


# --- Scoring ---

def pair_quad_score(type_value, count):
    """Default scoring: paired evens good, odds bad.
    type_value is 1-indexed face value."""
    if count == 0:
        return 0
    if count == 2:
        return 2 * type_value
    if count == 4:
        return 8 * type_value
    return -type_value * count


# --- Config ---

@dataclass
class GameConfig:
    num_types: int = 10
    init_pool: tuple = tuple(range(20, 10, -1))  # type i has 20-i goods initially
    draw_size: int = 10                           # goods drawn per round
    num_bundles: int = 5                         # choices per round
    overlap_degree: int = 2                      # bundles each good appears in
    score_fn: Callable = field(default=pair_quad_score)

    @property
    def num_rounds(self):
        return sum(self.init_pool) // self.draw_size

    @property
    def init_stashed(self):
        return (0,) * self.num_types

    @staticmethod
    def small():
        """Small game config for fast iteration."""
        return GameConfig(
            num_types=5,
            init_pool=(8, 7, 6, 5, 4),
            draw_size=5,
            num_bundles=4,
            overlap_degree=2,
            score_fn=pair_quad_score,
        )

    def make_score_table(self, max_count=None):
        """Build score_table[type_idx, count] as a 2D numpy array.
        type_idx is 0-based; score_fn receives 1-indexed type_value."""
        if max_count is None:
            max_count = max(self.init_pool) + self.draw_size
        table = np.zeros((self.num_types, max_count + 1), dtype=np.float64)
        for i in range(self.num_types):
            for c in range(max_count + 1):
                table[i, c] = self.score_fn(i + 1, c)
        return table


# --- Numba scoring from table ---

@numba.jit(nopython=True, cache=True)
def total_score_from_table(stashed, score_table):
    total = 0.0
    for i in range(stashed.shape[0]):
        total += score_table[i, stashed[i]]
    return total


# --- Assignment generation ---

@numba.jit(nopython=True, cache=True)
def generate_assignments(draw_size, num_bundles, overlap_degree):
    """For each drawn good, pick `overlap_degree` random bundles.
    Returns assignments array of shape (draw_size, overlap_degree)."""
    assignments = np.empty((draw_size, overlap_degree), dtype=numba.int64)
    # temp array for Fisher-Yates partial shuffle
    perm = np.empty(num_bundles, dtype=numba.int64)
    for j in range(draw_size):
        for i in range(num_bundles):
            perm[i] = i
        for i in range(overlap_degree):
            swap_idx = np.random.randint(i, num_bundles)
            perm[i], perm[swap_idx] = perm[swap_idx], perm[i]
            assignments[j, i] = perm[i]
    return assignments


# --- Sampling (numpy, used for the outer game loop) ---

_rng = np.random.default_rng()


def sample_transitions(stashed, remaining, config):
    """Sample one draw and return (transitions, draw, assignments).
    draw is 0-indexed type indices."""
    num_types = config.num_types
    draw_size = config.draw_size
    num_bundles = config.num_bundles
    overlap_degree = config.overlap_degree

    rem_arr = np.array(remaining, dtype=np.int64)
    counts = _rng.multivariate_hypergeometric(rem_arr, draw_size)
    draw = np.repeat(np.arange(num_types, dtype=np.int64), counts)
    _rng.shuffle(draw)

    # new remaining after removing all drawn goods
    new_remaining = tuple(int(remaining[i] - counts[i]) for i in range(num_types))

    # random bundle assignments
    assignments = generate_assignments(draw_size, num_bundles, overlap_degree)

    # build each bundle's transition
    transitions = []
    for b in range(num_bundles):
        s = list(stashed)
        for j in range(draw_size):
            for r in range(overlap_degree):
                if assignments[j, r] == b:
                    s[draw[j]] += 1
                    break
        transitions.append((tuple(s), new_remaining))

    return transitions, draw, assignments


def _format_bundles(draw, assignments, action, config):
    """Show each bundle as a row, with the chosen one marked."""
    lines = []
    for b in range(config.num_bundles):
        goods = []
        for j in range(config.draw_size):
            for r in range(config.overlap_degree):
                if assignments[j, r] == b:
                    goods.append(str(draw[j] + 1))
                    break
        contents = " ".join(goods) if goods else "(empty)"
        marker = " >>>" if b == action else ""
        lines.append(f"           bundle {b}: {contents}{marker}")
    return "\n".join(lines)


def play_game(config, action_fn, verbose=False):
    """Play one game. action_fn(transitions) -> int index of chosen bundle.

    Returns final score as a float.
    """
    score_table = config.make_score_table()
    stashed = config.init_stashed
    remaining = config.init_pool

    for round_num in range(config.num_rounds):
        if sum(remaining) < config.draw_size:
            break
        transitions, draw, assignments = sample_transitions(stashed, remaining, config)
        action = action_fn(transitions)
        stashed, remaining = transitions[action]

        if verbose:
            draw_str = " ".join(str(v + 1) for v in draw)
            stash_str = "  ".join(f"{c:>2}" for c in stashed)
            remaining_str = "  ".join(f"{c:>2}" for c in remaining)
            print(f"  Round {round_num+1}: drew [{draw_str}]")
            print(_format_bundles(draw, assignments, action, config))
            print(f"  Stash:     [{stash_str}]")
            print(f"  Remaining: [{remaining_str}]")
            print()

    score = total_score_from_table(np.array(stashed, dtype=np.int64), score_table)
    if verbose:
        print(f"  Score: {score:.0f}")
    return score


def play_games_batched(n, engine, config, collect_data=False):
    """Play n games in lockstep using Engine.search_roots_batch.

    Returns dict with:
        scores: (n,) float array of final scores
        total_leaves: int, total leaves evaluated
    If collect_data=True, also includes:
        stashed: (S, num_types) int32 array of all root transitions
        remaining: (S, num_types) int32 array
        values: (S,) float32 array of search values per transition
    """
    B = config.num_bundles
    T = config.num_types
    score_table = config.make_score_table()
    total_leaves = 0

    stashed = np.zeros((n, T), dtype=np.int64)
    remaining = np.tile(np.array(config.init_pool, dtype=np.int64), (n, 1))

    if collect_data:
        all_stashed = []
        all_remaining = []
        all_values = []

    for round_idx in range(config.num_rounds):
        deck_size = int(remaining[0].sum())
        if deck_size < config.draw_size:
            break

        root_s = np.empty((n * B, T), dtype=np.int64)
        root_r = np.empty((n * B, T), dtype=np.int64)
        for g in range(n):
            transitions, _, _ = sample_transitions(
                tuple(stashed[g]), tuple(remaining[g]), config
            )
            for b, (s, r) in enumerate(transitions):
                idx = g * B + b
                root_s[idx] = s
                root_r[idx] = r

        root_vals, num_leaves = engine.search_roots_batch(root_s, root_r)
        total_leaves += num_leaves
        game_vals = root_vals.reshape(n, B)

        if collect_data:
            all_stashed.append(root_s.astype(np.int32))
            all_remaining.append(root_r.astype(np.int32))
            all_values.append(game_vals.ravel())

        best = game_vals.argmax(axis=1)
        for g in range(n):
            idx = g * B + best[g]
            stashed[g] = root_s[idx]
            remaining[g] = root_r[idx]

    scores = np.array([
        total_score_from_table(stashed[g], score_table) for g in range(n)
    ])

    result = {"scores": scores, "total_leaves": total_leaves}
    if collect_data:
        result["stashed"] = np.concatenate(all_stashed)
        result["remaining"] = np.concatenate(all_remaining)
        result["values"] = np.concatenate(all_values).astype(np.float32)
    return result
