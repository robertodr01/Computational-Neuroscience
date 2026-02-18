"""Data loading and plotting utilities for NARMA10 experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def load_narma10(csv_path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load NARMA10 as (input, target) torch tensors."""

    frame = pd.read_csv(csv_path, header=None)
    x = torch.tensor(frame.iloc[0].values, dtype=torch.float32)
    y = torch.tensor(frame.iloc[1].values, dtype=torch.float32)
    return x, y


def split_series(x: torch.Tensor, y: torch.Tensor, train_size: int, val_size: int):
    """Split contiguous data into train, validation and test partitions."""

    x_train = x[:train_size]
    y_train = y[:train_size]

    x_val = x[train_size : train_size + val_size]
    y_val = y[train_size : train_size + val_size]

    x_test = x[train_size + val_size :]
    y_test = y[train_size + val_size :]

    return x_train, x_val, x_test, y_train, y_val, y_test


def make_tdnn_tensors(x_split: tuple[torch.Tensor, torch.Tensor, torch.Tensor], y_split):
    """Convert split vectors into TDNN tensors."""

    x_train, x_val, x_test = x_split
    y_train, y_val, y_test = y_split

    x_train = x_train.reshape(1, 1, -1)
    x_val = x_val.reshape(1, 1, -1)
    x_test = x_test.reshape(1, 1, -1)

    y_train = y_train.reshape(1, -1, 1)
    y_val = y_val.reshape(1, -1, 1)
    y_test = y_test.reshape(1, -1, 1)

    return x_train, x_val, x_test, y_train, y_val, y_test


def make_rnn_tensors(x_split: tuple[torch.Tensor, torch.Tensor, torch.Tensor], y_split):
    """Convert split vectors into RNN tensors (single batch)."""

    x_train, x_val, x_test = x_split
    y_train, y_val, y_test = y_split

    x_train = x_train.reshape(1, -1, 1)
    x_val = x_val.reshape(1, -1, 1)
    x_test = x_test.reshape(1, -1, 1)

    y_train = y_train.reshape(1, -1, 1)
    y_val = y_val.reshape(1, -1, 1)
    y_test = y_test.reshape(1, -1, 1)

    return x_train, x_val, x_test, y_train, y_val, y_test


def plot_series_segment(series: torch.Tensor, title: str, limit: int = 200):
    plt.figure(figsize=(12, 3.5))
    plt.plot(series[:limit].detach().cpu().numpy())
    plt.title(title)
    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.grid(alpha=0.25)
    plt.show()


def plot_loss_curves(train_history: list[float], val_history: list[float], title: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4.2))
    plt.plot(train_history, label="Train")
    plt.plot(val_history, label="Validation")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.show()


def plot_prediction_slice(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    title: str,
    path: str | Path,
    limit: int = 250,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    yt = y_true.reshape(-1).detach().cpu().numpy()[:limit]
    yp = y_pred.reshape(-1).detach().cpu().numpy()[:limit]

    plt.figure(figsize=(12, 3.5))
    plt.plot(yt, label="True", alpha=0.8)
    plt.plot(yp, label="Predicted", alpha=0.8, linestyle=":")
    plt.title(title)
    plt.xlabel("Time step")
    plt.ylabel("Output")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.show()


def save_metrics(path: str | Path, metrics: dict[str, float], header: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as fp:
        fp.write(f"{header}\n")
        fp.write("-" * len(header) + "\n\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                fp.write(f"{key}: {value:.8f}\n")
            else:
                fp.write(f"{key}: {value}\n")
