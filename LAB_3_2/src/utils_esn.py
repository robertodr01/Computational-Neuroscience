"""Utility functions for ESN assignment notebook."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch


def load_narma10(csv_path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    frame = pd.read_csv(csv_path, header=None)
    x = torch.tensor(frame.iloc[0].values, dtype=torch.float32)
    y = torch.tensor(frame.iloc[1].values, dtype=torch.float32)
    return x, y


def split_series(x: torch.Tensor, y: torch.Tensor, train_size: int, val_size: int):
    x_train = x[:train_size]
    y_train = y[:train_size]

    x_val = x[train_size : train_size + val_size]
    y_val = y[train_size : train_size + val_size]

    x_test = x[train_size + val_size :]
    y_test = y[train_size + val_size :]

    return x_train, x_val, x_test, y_train, y_val, y_test


def reshape_for_esn(x_split, y_split):
    x_train, x_val, x_test = x_split
    y_train, y_val, y_test = y_split

    return (
        x_train.reshape(-1, 1),
        x_val.reshape(-1, 1),
        x_test.reshape(-1, 1),
        y_train.reshape(-1, 1),
        y_val.reshape(-1, 1),
        y_test.reshape(-1, 1),
    )


def plot_series(series: torch.Tensor, title: str, limit: int = 200):
    plt.figure(figsize=(12, 3.5))
    plt.plot(series[:limit].detach().cpu().numpy())
    plt.title(title)
    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.grid(alpha=0.25)
    plt.show()


def plot_predictions(y_true: torch.Tensor, y_pred: torch.Tensor, title: str, path: str | Path, start: int = 0, end: int = 400):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    yt = y_true.reshape(-1).detach().cpu().numpy()[start:end]
    yp = y_pred.reshape(-1).detach().cpu().numpy()[start:end]

    plt.figure(figsize=(12, 3.5))
    plt.plot(yt, label="True", alpha=0.75)
    plt.plot(yp, label="Predicted", alpha=0.75, linestyle=":")
    plt.title(title)
    plt.xlabel("Time step")
    plt.ylabel("Output")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.show()
