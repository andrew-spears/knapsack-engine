"""
Neural network value function for the collection game.

Approximates expected score under optimal play for a given game state.
"""

import torch
import torch.nn as nn
import numpy as np
from game import GameConfig, total_score_from_table


class ValueNet(nn.Module):
    """Small fully-connected network: state -> predicted value."""

    def __init__(self, num_types, hidden_size=32, num_layers=3):
        super().__init__()
        input_size = 2 * num_types  # stashed + remaining, both normalized
        layers = []
        in_dim = input_size
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            in_dim = hidden_size
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _pool_norm(config):
    """Normalization divisor: init_pool clamped to avoid div-by-zero."""
    return np.maximum(np.array(config.init_pool, dtype=np.float32), 1.0)


def encode_state_arrays(stashed, remaining, config):
    """Encode numpy arrays of states as a normalized float tensor.

    stashed, remaining: (N, num_types) int arrays.
    Returns: tensor of shape (N, 2 * num_types).
    """
    pool = _pool_norm(config)
    s_norm = stashed.astype(np.float32) / pool
    r_norm = remaining.astype(np.float32) / pool
    return torch.tensor(np.concatenate([s_norm, r_norm], axis=1), dtype=torch.float32)


def encode_state_tuples(states, config):
    """Encode list of (stashed, remaining) tuples as a normalized float tensor.

    Returns: tensor of shape (len(states), 2 * num_types).
    """
    pool = _pool_norm(config)
    rows = []
    for stashed, remaining in states:
        s = np.array(stashed, dtype=np.float32) / pool
        r = np.array(remaining, dtype=np.float32) / pool
        rows.append(np.concatenate([s, r]))
    return torch.tensor(np.array(rows), dtype=torch.float32)


def make_leaf_fn(model, config):
    """Create a leaf_fn that uses the NN for non-terminal states.

    Returns a callable: list of (stashed, remaining, is_terminal) -> list of floats.
    Terminal states are evaluated with score_table; non-terminal states get a
    single batched forward pass through the model.
    """
    score_table = config.make_score_table()

    def leaf_fn(leaves):
        n = len(leaves)
        values = [0.0] * n
        nn_indices = []
        nn_states = []

        for i, (stashed, remaining, is_terminal) in enumerate(leaves):
            if is_terminal:
                stashed_arr = np.array(stashed, dtype=np.int64)
                values[i] = float(total_score_from_table(stashed_arr, score_table))
            else:
                nn_indices.append(i)
                nn_states.append((stashed, remaining))

        if nn_states:
            X = encode_state_tuples(nn_states, config)
            with torch.no_grad():
                preds = model(X).cpu().numpy()
            for j, idx in enumerate(nn_indices):
                values[idx] = float(preds[j])

        return values

    return leaf_fn


def make_heuristic_leaf_fn(config):
    """Create a leaf_fn that uses score_table for all states (terminal or not).

    Useful as a baseline: equivalent logic to the numba _search path but
    executed through the batched expand_tree infrastructure.
    """
    score_table = config.make_score_table()

    def leaf_fn(leaves):
        values = []
        for stashed, remaining, is_terminal in leaves:
            stashed_arr = np.array(stashed, dtype=np.int64)
            values.append(float(total_score_from_table(stashed_arr, score_table)))
        return values

    return leaf_fn


def make_array_leaf_fn(model, config):
    """Create a leaf_fn for array-based batched evaluation.

    Returns callable: (stashed_arr, remaining_arr) -> values_arr
    where inputs are (N, num_types) int64 arrays and output is (N,) float32.
    Terminal detection is handled by the caller (Engine).
    """
    pool_norm = _pool_norm(config)

    def leaf_fn(stashed, remaining):
        s_norm = stashed.astype(np.float32) / pool_norm
        r_norm = remaining.astype(np.float32) / pool_norm
        X = torch.tensor(np.concatenate([s_norm, r_norm], axis=1), dtype=torch.float32)
        with torch.no_grad():
            return model(X).numpy()

    return leaf_fn


def greedy_nn_action(model, transitions, config):
    """Pick best action using NN value predictions (no search)."""
    X = encode_state_tuples(transitions, config)
    with torch.no_grad():
        values = model(X)
    return values.argmax().item()


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(path, config, hidden_size=32, num_layers=3):
    model = ValueNet(config.num_types, hidden_size, num_layers)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model
