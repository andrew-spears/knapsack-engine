"""
Expectimax search engine.

Tree alternates chance nodes (random draw, averaged) and decision nodes
(pick bundle, maxed). Heuristic at leaves is the current score via lookup table.

Supports pluggable leaf evaluation via `leaf_fn` for batched NN evaluation.
"""

import numpy as np
import numba
from numba import int64, float64
from game import GameConfig, total_score_from_table, generate_assignments, sample_transitions


@numba.jit(nopython=True, cache=True)
def sample_draw(remaining, deck_size, draw_size, num_types):
    """Sample draw_size goods without replacement via cumulative sum lookup.
    Modifies remaining in place — caller must copy first."""
    draw = np.empty(draw_size, dtype=int64)
    size = deck_size
    for j in range(draw_size):
        r = np.random.randint(0, size)
        cumsum = 0
        for i in range(num_types):
            cumsum += remaining[i]
            if r < cumsum:
                draw[j] = i
                remaining[i] -= 1
                size -= 1
                break
    return draw


@numba.jit(nopython=True, cache=True)
def _copy_array(src, n):
    dst = np.empty(n, dtype=int64)
    for i in range(n):
        dst[i] = src[i]
    return dst


@numba.jit(nopython=True, cache=True)
def _apply_bundle(stashed, draw, assignments, bundle_idx, draw_size, overlap_degree, num_types):
    """Copy stashed and add goods assigned to bundle_idx."""
    new = np.empty(num_types, dtype=int64)
    for i in range(num_types):
        new[i] = stashed[i]
    for j in range(draw_size):
        for r in range(overlap_degree):
            if assignments[j, r] == bundle_idx:
                new[draw[j]] += 1
                break
    return new


@numba.jit(nopython=True, cache=True)
def _search(stashed, remaining, deck_size, depth, fanout,
            num_types, draw_size, num_bundles, overlap_degree, score_table):
    """Expectimax search. Returns (value, nodes_visited)."""
    if depth == 0 or deck_size < draw_size:
        return total_score_from_table(stashed, score_table), 1

    sum_value = 0.0
    total_nodes = 0

    for _ in range(fanout):
        rem_copy = _copy_array(remaining, num_types)
        draw = sample_draw(rem_copy, deck_size, draw_size, num_types)
        assignments = generate_assignments(draw_size, num_bundles, overlap_degree)

        best = -1e9
        for b in range(num_bundles):
            new_stash = _apply_bundle(stashed, draw, assignments, b,
                                      draw_size, overlap_degree, num_types)
            val, nodes = _search(new_stash, rem_copy, deck_size - draw_size,
                                 depth - 1, fanout,
                                 num_types, draw_size, num_bundles, overlap_degree, score_table)
            total_nodes += nodes
            if val > best:
                best = val

        sum_value += best

    return sum_value / fanout, total_nodes


# --- Array-based tree expansion (numba) + numpy backup ---

@numba.jit(nopython=True, cache=True)
def expand_level(stashed, remaining, fanout, num_bundles, draw_size, overlap_degree, num_types):
    """Expand N states by one tree level (chance + decision) via numba.

    Each state gets `fanout` random draws, each with `num_bundles` bundle choices.

    Input:  stashed (N, num_types), remaining (N, num_types)
    Output: (N * fanout * num_bundles, num_types) for both stashed and remaining.
    """
    N = stashed.shape[0]
    M = N * fanout * num_bundles
    out_stashed = np.empty((M, num_types), dtype=int64)
    out_remaining = np.empty((M, num_types), dtype=int64)

    for n in range(N):
        deck_size = 0
        for t in range(num_types):
            deck_size += remaining[n, t]

        for f in range(fanout):
            rem = np.empty(num_types, dtype=int64)
            for t in range(num_types):
                rem[t] = remaining[n, t]
            draw = sample_draw(rem, deck_size, draw_size, num_types)
            assignments = generate_assignments(draw_size, num_bundles, overlap_degree)

            for b in range(num_bundles):
                idx = (n * fanout + f) * num_bundles + b
                new_s = _apply_bundle(stashed[n], draw, assignments, b,
                                      draw_size, overlap_degree, num_types)
                for t in range(num_types):
                    out_stashed[idx, t] = new_s[t]
                    out_remaining[idx, t] = rem[t]

    return out_stashed, out_remaining


@numba.jit(nopython=True, cache=True)
def batch_score_from_table(stashed, score_table):
    """Score N states using the precomputed score table."""
    N = stashed.shape[0]
    num_types = stashed.shape[1]
    values = np.empty(N, dtype=float64)
    for n in range(N):
        total = 0.0
        for t in range(num_types):
            total += score_table[t, stashed[n, t]]
        values[n] = total
    return values


def expand_to_leaves(stashed, remaining, depth, fanout, config):
    """Expand states through `depth` tree levels via numba.

    All states at a given level have the same remaining total, so terminality
    is uniform per level. Stops early if leaves become terminal.

    Returns (leaf_stashed, leaf_remaining, actual_depth).
    """
    s, r = stashed, remaining
    actual_depth = 0
    for d in range(depth):
        deck_size = int(r[0].sum())
        if deck_size < config.draw_size:
            break
        s, r = expand_level(s, r, fanout, config.num_bundles,
                            config.draw_size, config.overlap_degree, config.num_types)
        actual_depth += 1
    return s, r, actual_depth


def propagate_leaf_values(values, num_roots, actual_depth, fanout, num_bundles):
    """Propagate leaf values to roots: max over bundles, mean over draws, per level.

    values: flat array of length num_roots * (fanout * num_bundles)^actual_depth
    Returns: array of length num_roots.
    """
    if actual_depth == 0:
        return values
    v = values.reshape(num_roots, *([fanout, num_bundles] * actual_depth))
    for _ in range(actual_depth):
        v = v.max(axis=-1)   # decision: max over bundles
        v = v.mean(axis=-1)  # chance: mean over draws
    return v


# --- Closure-based tree expansion (Python, used by Engine.search_value with leaf_fn) ---

def expand_tree(stashed, remaining, depth, fanout, config, score_table):
    """Expand expectimax tree in Python, collecting leaves for batched evaluation.

    Returns (leaves, backup_fn):
      - leaves: list of (stashed, remaining, is_terminal) tuples
      - backup_fn(values): takes a list/array of float values (one per leaf),
        propagates max-over-bundles then mean-over-fanout back to root, returns scalar.
    """
    leaves = []

    def _expand(stashed, remaining, depth):
        deck_size = sum(remaining)
        is_terminal = deck_size < config.draw_size

        if depth == 0 or is_terminal:
            idx = len(leaves)
            leaves.append((stashed, remaining, is_terminal))
            return lambda values: values[idx]

        # Chance node: sample `fanout` draws, average results
        sample_backups = []
        for _ in range(fanout):
            transitions, _, _ = sample_transitions(stashed, remaining, config)

            # Decision node: max over bundles
            bundle_backups = []
            for s, r in transitions:
                bundle_backups.append(_expand(s, r, depth - 1))

            # Capture bundle_backups in closure
            def _max_backup(values, _bbs=bundle_backups):
                best = -1e9
                for bb in _bbs:
                    v = bb(values)
                    if v > best:
                        best = v
                return best

            sample_backups.append(_max_backup)

        def _mean_backup(values, _sbs=sample_backups):
            total = 0.0
            for sb in _sbs:
                total += sb(values)
            return total / len(_sbs)

        return _mean_backup

    backup_fn = _expand(stashed, remaining, depth)
    return leaves, backup_fn


class Engine:
    def __init__(self, depth, fanout, config=None, leaf_fn=None, array_leaf_fn=None):
        """
        leaf_fn: optional callable that takes a list of (stashed, remaining, is_terminal)
                 tuples and returns a list of float values. If provided, uses
                 expand_tree + leaf_fn for batched evaluation instead of numba _search.
        array_leaf_fn: optional callable (stashed_arr, remaining_arr) -> values_arr
                 for array-based batched evaluation via search_roots_batch.
        """
        if config is None:
            config = GameConfig()
        self.depth = depth
        self.fanout = fanout
        self.config = config
        self.score_table = config.make_score_table()
        self.node_count = 0
        self.leaf_fn = leaf_fn
        self.array_leaf_fn = array_leaf_fn

    def search_value(self, stashed, remaining):
        cfg = self.config
        deck_size = sum(remaining)

        if self.leaf_fn is not None:
            # Batched leaf evaluation path
            if self.depth == 0 or deck_size < cfg.draw_size:
                self.node_count += 1
                # Single leaf — just call leaf_fn directly
                is_terminal = deck_size < cfg.draw_size
                values = self.leaf_fn([(stashed, remaining, is_terminal)])
                return values[0]

            leaves, backup_fn = expand_tree(
                stashed, remaining, self.depth, self.fanout, cfg, self.score_table
            )
            self.node_count += len(leaves)
            values = self.leaf_fn(leaves)
            return backup_fn(values)
        else:
            # Numba path
            stashed_arr = np.array(stashed, dtype=np.int64)
            remaining_arr = np.array(remaining, dtype=np.int64)

            if self.depth == 0 or deck_size < cfg.draw_size:
                self.node_count += 1
                return total_score_from_table(stashed_arr, self.score_table)

            val, nodes = _search(
                stashed_arr, remaining_arr,
                deck_size, self.depth, self.fanout,
                cfg.num_types, cfg.draw_size, cfg.num_bundles,
                cfg.overlap_degree, self.score_table,
            )
            self.node_count += nodes
            return val

    def search_roots_batch(self, root_s, root_r):
        """Batch search over multiple root states using array-based expansion.

        root_s, root_r: (N, num_types) int64 arrays.
        Returns: (values, num_leaves) where values is (N,) float32.
        Requires array_leaf_fn for non-terminal leaves.
        """
        cfg = self.config
        leaf_s, leaf_r, actual_depth = expand_to_leaves(
            root_s, root_r, self.depth, self.fanout, cfg
        )
        num_leaves = len(leaf_s)

        is_terminal = int(leaf_r[0].sum()) < cfg.draw_size
        if is_terminal or self.array_leaf_fn is None:
            leaf_vals = batch_score_from_table(leaf_s, self.score_table).astype(np.float32)
        else:
            leaf_vals = self.array_leaf_fn(leaf_s, leaf_r)

        num_roots = len(root_s)
        root_vals = propagate_leaf_values(
            leaf_vals, num_roots, actual_depth, self.fanout, cfg.num_bundles
        )
        return root_vals, num_leaves

    def get_action(self, transitions):
        best_action = 0
        best_value = -1e9
        for idx, (stashed, remaining) in enumerate(transitions):
            v = self.search_value(stashed, remaining)
            if v > best_value:
                best_value = v
                best_action = idx
        return best_action
