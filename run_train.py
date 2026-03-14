"""
Train model from pre-generated .npz data files.

Usage:
    python run_train.py data.npz                       # train on one file
    python run_train.py data1.npz data2.npz            # combine multiple
    python run_train.py data/*.npz --epochs 200        # glob works too
    python run_train.py data.npz --resume value_net.pt # continue training

Outputs:
    value_net.pt — trained model weights
"""

import argparse
import os
import re
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from game import GameConfig, play_game, play_games_batched
from engine import Engine
from model import (ValueNet, make_array_leaf_fn,
                   save_model, load_model, greedy_nn_action, encode_state_arrays)


def auto_model_name(data_paths, epochs, hidden, layers):
    """Derive model name from data files. e.g. data/d2_f10_r1.npz -> models/d2_f10_24k_h128_l4_e50.pt"""
    basename = os.path.basename(data_paths[0])
    m = re.match(r'(d\d+_f\d+)', basename)
    prefix = m.group(1) if m else "model"
    n_samples = sum(len(np.load(p)["values"]) for p in data_paths)
    return f"models/{prefix}_{n_samples // 1000}k_h{hidden}_l{layers}_e{epochs}.pt"


def load_data(paths):
    """Load and concatenate multiple .npz data files."""
    all_stashed, all_remaining, all_values = [], [], []
    for p in paths:
        d = np.load(p)
        all_stashed.append(d["stashed"])
        all_remaining.append(d["remaining"])
        all_values.append(d["values"])
        print(f"  {p}: {len(d['values'])} samples")
    stashed = np.concatenate(all_stashed)
    remaining = np.concatenate(all_remaining)
    values = np.concatenate(all_values)
    print(f"  Total: {len(values)} samples")
    print(f"  Value range: [{values.min():.1f}, {values.max():.1f}], mean: {values.mean():.1f}")
    return stashed, remaining, values


def train(model, stashed, remaining, values, config, epochs, batch_size, lr, device="cpu"):
    """Train model on numpy arrays. Returns (final_loss, loss_history)."""
    X = encode_state_arrays(stashed, remaining, config).to(device)
    y = torch.tensor(values, dtype=torch.float32).to(device)
    model = model.to(device)

    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    losses = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}")

    model = model.cpu()
    model.eval()
    return avg_loss, losses


def evaluate(config, model, n_games):
    array_leaf_fn = make_array_leaf_fn(model, config)
    engine_nn = Engine(2, 10, config, array_leaf_fn=array_leaf_fn)
    engine_heur = Engine(2, 10, config)

    search_nn = play_games_batched(n_games, engine_nn, config)["scores"]
    search_heur = play_games_batched(n_games, engine_heur, config)["scores"]
    nn_only = [play_game(config, lambda t: greedy_nn_action(model, t, config)) for _ in range(n_games)]
    random_scores = [play_game(config, lambda t: np.random.randint(len(t))) for _ in range(n_games)]

    print(f"\n{'Agent':<25s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s}")
    print("-" * 58)
    for name, scores in [
        ("Search + NN leaf", search_nn),
        ("Search + heuristic leaf", search_heur),
        ("NN only (no search)", nn_only),
        ("Random", random_scores),
    ]:
        print(f"{name:<25s} {np.mean(scores):8.1f} {np.std(scores):8.1f} {min(scores):8.1f} {max(scores):8.1f}")


def main():
    p = argparse.ArgumentParser(description="Train NN value function from .npz data")
    p.add_argument("data", nargs="+", help=".npz data files")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--eval-games", type=int, default=50)
    p.add_argument("--resume", type=str, default=None, help="Model to continue from")
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    config = GameConfig.small()

    if args.output is None:
        args.output = auto_model_name(args.data, args.epochs, args.hidden, args.layers)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print("Loading data:")
    stashed, remaining, values = load_data(args.data)

    if args.resume:
        print(f"\nResuming from {args.resume}")
        model = load_model(args.resume, config, args.hidden, args.layers)
    else:
        model = ValueNet(config.num_types, args.hidden, args.layers)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Device selection
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    print(f"\nTraining ({args.epochs} epochs):")
    final_loss, losses = train(model, stashed, remaining, values, config,
                               args.epochs, args.batch_size, args.lr, device=device)
    print(f"Final loss: {final_loss:.4f}")

    save_model(model, args.output)
    print(f"\nSaved to {args.output}")

    # Save loss plot
    plot_path = args.output.replace(".pt", "_loss.png")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(losses) + 1), losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"Training Loss ({len(values)} samples, lr={args.lr})")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"Loss plot: {plot_path}")

    print(f"\nEvaluating ({args.eval_games} games):")
    evaluate(config, model, args.eval_games)


if __name__ == "__main__":
    main()
