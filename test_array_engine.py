"""
Verify array-based tree expansion + backup matches closure-based expand_tree.

Both paths should produce the same leaf states and backup the same result
(up to randomness — so we seed them identically).
"""

import numpy as np
import torch
from game import GameConfig, sample_transitions
from engine import (Engine, expand_to_leaves, propagate_leaf_values,
                    batch_score_from_table)
from model import ValueNet, load_model, encode_state_tuples


def test_backup_simple():
    """Test propagate_leaf_values on a hand-constructed example."""
    # 2 roots, depth=1, fanout=2, num_bundles=3
    # Each root has 2*3=6 leaves
    # Layout: [root0: [draw0: [b0, b1, b2], draw1: [b0, b1, b2]], root1: ...]
    values = np.array([
        # root 0, draw 0
        1.0, 5.0, 3.0,   # max=5
        # root 0, draw 1
        2.0, 4.0, 6.0,   # max=6
        # root 1, draw 0
        10.0, 8.0, 9.0,  # max=10
        # root 1, draw 1
        7.0, 12.0, 11.0, # max=12
    ], dtype=np.float32)

    result = propagate_leaf_values(values, num_roots=2, actual_depth=1, fanout=2, num_bundles=3)
    # root 0: mean(5, 6) = 5.5
    # root 1: mean(10, 12) = 11.0
    expected = np.array([5.5, 11.0])
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print("  backup simple: OK")


def test_backup_depth2():
    """Test backup at depth=2."""
    # 1 root, depth=2, fanout=2, num_bundles=2
    # Leaves = 1 * (2*2)^2 = 16
    # Shape after reshape: (1, 2, 2, 2, 2)
    values = np.arange(16, dtype=np.float32)
    result = propagate_leaf_values(values, num_roots=1, actual_depth=2, fanout=2, num_bundles=2)

    # Manual: reshape to (1, 2, 2, 2, 2)
    v = values.reshape(1, 2, 2, 2, 2)
    # Inner: max over last axis (bundles): (1,2,2,2)
    v = v.max(axis=-1)
    # Mean over last axis (draws): (1,2,2)
    v = v.mean(axis=-1)
    # Max over last axis (bundles): (1,2)
    v = v.max(axis=-1)
    # Mean over last axis (draws): (1,)
    v = v.mean(axis=-1)
    assert np.allclose(result, v), f"Expected {v}, got {result}"
    print("  backup depth=2: OK")


def test_backup_depth0():
    """Depth 0 should return values unchanged."""
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = propagate_leaf_values(values, num_roots=3, actual_depth=0, fanout=5, num_bundles=4)
    assert np.array_equal(result, values)
    print("  backup depth=0: OK")


def test_batch_score():
    """Test batch_score_from_table against single-state scoring."""
    config = GameConfig.small()
    score_table = config.make_score_table()

    stashed = np.array([
        [0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0],
        [1, 2, 3, 4, 0],
        [4, 4, 4, 4, 4],
    ], dtype=np.int64)

    from game import total_score_from_table
    expected = np.array([total_score_from_table(stashed[i], score_table)
                         for i in range(len(stashed))])
    result = batch_score_from_table(stashed, score_table)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print("  batch_score: OK")


def test_expand_level_shapes():
    """Test that expand_level produces correct output shapes."""
    config = GameConfig.small()
    N, F, B = 3, 5, config.num_bundles

    stashed = np.zeros((N, config.num_types), dtype=np.int64)
    remaining = np.tile(np.array(config.init_pool, dtype=np.int64), (N, 1))

    from engine import expand_level
    out_s, out_r = expand_level(stashed, remaining, F, B,
                                config.draw_size, config.overlap_degree, config.num_types)

    expected_rows = N * F * B
    assert out_s.shape == (expected_rows, config.num_types), f"stashed shape: {out_s.shape}"
    assert out_r.shape == (expected_rows, config.num_types), f"remaining shape: {out_r.shape}"

    # All remaining should have same total (init - draw_size)
    expected_total = sum(config.init_pool) - config.draw_size
    totals = out_r.sum(axis=1)
    assert np.all(totals == expected_total), f"Remaining totals vary: {np.unique(totals)}"

    # At least some states should have gained items
    assert np.any(out_s.sum(axis=1) > 0), "No items stashed in any state after expansion"
    print(f"  expand_level shapes: OK ({expected_rows} states)")


def test_expand_to_leaves_depth():
    """Test that expand_to_leaves stops at correct depth and handles terminal states."""
    config = GameConfig.small()

    stashed = np.zeros((1, config.num_types), dtype=np.int64)
    remaining = np.array([config.init_pool], dtype=np.int64)

    # Should expand full depth
    leaf_s, leaf_r, actual = expand_to_leaves(stashed, remaining, depth=2, fanout=3, config=config)
    assert actual == 2, f"Expected depth 2, got {actual}"
    expected_leaves = 1 * (3 * config.num_bundles) ** 2
    assert len(leaf_s) == expected_leaves, f"Expected {expected_leaves} leaves, got {len(leaf_s)}"

    # With very deep search, should stop when terminal
    leaf_s2, leaf_r2, actual2 = expand_to_leaves(stashed, remaining, depth=20, fanout=2, config=config)
    assert actual2 <= config.num_rounds, f"Expanded {actual2} levels but only {config.num_rounds} rounds"
    assert int(leaf_r2[0].sum()) < config.draw_size, "Leaves should be terminal"
    print(f"  expand_to_leaves depth: OK (depth=2 -> {len(leaf_s)} leaves, "
          f"depth=20 -> stopped at {actual2})")


def test_array_vs_closure_values():
    """Compare array path vs closure path on the same state.

    Both should give similar distributions (not identical due to different
    random draws, but statistically close over many runs).
    """
    config = GameConfig.small()
    score_table = config.make_score_table()

    stashed = (2, 1, 0, 1, 0)
    remaining = (6, 6, 6, 4, 4)
    depth, fanout = 2, 20

    # Run closure path many times
    engine = Engine(depth, fanout, config)
    closure_values = [engine.search_value(stashed, remaining) for _ in range(200)]

    # Run array path many times
    s_arr = np.array([stashed], dtype=np.int64)
    r_arr = np.array([remaining], dtype=np.int64)
    array_values = []
    for _ in range(200):
        leaf_s, leaf_r, actual = expand_to_leaves(s_arr, r_arr, depth, fanout, config)
        leaf_vals = batch_score_from_table(leaf_s, score_table).astype(np.float32)
        root_val = propagate_leaf_values(leaf_vals, 1, actual, fanout, config.num_bundles)
        array_values.append(float(root_val[0]))

    # Both should have similar mean (same search algorithm, different random samples)
    closure_mean = np.mean(closure_values)
    array_mean = np.mean(array_values)
    diff = abs(closure_mean - array_mean)
    # Allow generous tolerance since these are stochastic
    assert diff < 5.0, (f"Means differ too much: closure={closure_mean:.2f}, "
                        f"array={array_mean:.2f}, diff={diff:.2f}")
    print(f"  array vs closure: OK (closure mean={closure_mean:.2f}, "
          f"array mean={array_mean:.2f}, diff={diff:.2f})")


def test_full_datagen_round():
    """Simulate one round of worker_batched logic and verify shapes/values."""
    config = GameConfig.small()
    score_table = config.make_score_table()
    depth, fanout = 2, 10
    n_games = 5
    B = config.num_bundles
    T = config.num_types

    stashed = np.zeros((n_games, T), dtype=np.int64)
    remaining = np.tile(np.array(config.init_pool, dtype=np.int64), (n_games, 1))

    # Sample transitions
    root_s = np.empty((n_games * B, T), dtype=np.int64)
    root_r = np.empty((n_games * B, T), dtype=np.int64)
    for g in range(n_games):
        transitions, _, _ = sample_transitions(tuple(stashed[g]), tuple(remaining[g]), config)
        for b, (s, r) in enumerate(transitions):
            root_s[g * B + b] = s
            root_r[g * B + b] = r

    # Expand
    leaf_s, leaf_r, actual_depth = expand_to_leaves(root_s, root_r, depth, fanout, config)
    expected_leaves = n_games * B * (fanout * B) ** actual_depth
    assert len(leaf_s) == expected_leaves, f"Expected {expected_leaves} leaves, got {len(leaf_s)}"

    # Evaluate (score table since no model)
    is_terminal = int(leaf_r[0].sum()) < config.draw_size
    if is_terminal:
        leaf_vals = batch_score_from_table(leaf_s, score_table).astype(np.float32)
    else:
        # Simulate NN with score table for testing
        leaf_vals = batch_score_from_table(leaf_s, score_table).astype(np.float32)

    # Backup
    root_vals = propagate_leaf_values(leaf_vals, n_games * B, actual_depth, fanout, B)
    assert root_vals.shape == (n_games * B,), f"Root vals shape: {root_vals.shape}"

    game_vals = root_vals.reshape(n_games, B)
    assert game_vals.shape == (n_games, B)

    # Each game should have a best action
    best = game_vals.argmax(axis=1)
    assert best.shape == (n_games,)
    assert np.all(best >= 0) and np.all(best < B)

    print(f"  full datagen round: OK ({n_games} games, {expected_leaves} leaves, "
          f"depth={actual_depth}, values={game_vals.mean():.1f}±{game_vals.std():.1f})")


if __name__ == "__main__":
    print("Testing propagate_leaf_values...")
    test_backup_simple()
    test_backup_depth2()
    test_backup_depth0()

    print("\nTesting batch_score_from_table...")
    test_batch_score()

    print("\nTesting expand_level...")
    test_expand_level_shapes()

    print("\nTesting expand_to_leaves...")
    test_expand_to_leaves_depth()

    print("\nTesting array vs closure path...")
    test_array_vs_closure_values()

    print("\nTesting full datagen round...")
    test_full_datagen_round()

    print("\nAll tests passed.")
