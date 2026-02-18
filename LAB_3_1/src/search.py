"""Random-search utilities for TDNN and RNN model selection."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch

try:
    from .models import RNNRegressor, TDNNRegressor
except ImportError:  # pragma: no cover - supports notebook-style imports
    from models import RNNRegressor, TDNNRegressor


def _sample(space: dict[str, list[Any]], rng: random.Random) -> dict[str, Any]:
    return {key: rng.choice(values) for key, values in space.items()}


def tdnn_random_search(
    spaces: list[dict[str, list[Any]]],
    n_iter: int,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    seed: int = 42,
):
    """Random search over TDNN hyperparameters."""

    rng = random.Random(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    best = None

    for trial in range(1, n_iter + 1):
        chosen_space = rng.choice(spaces)
        params = _sample(chosen_space, rng)

        model = TDNNRegressor(
            input_channels=1,
            conv_channels=params["conv_channels"],
            kernel_sizes=params["kernel_sizes"],
            dilations=params["dilations"],
            strides=params["strides"],
            activations=params["activations"],
            fc_hidden=params["fc_hidden"],
        )

        tr_hist, va_hist = model.fit(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            epochs=int(params["epochs"]),
            lr=float(params["lr"]),
            weight_decay=float(params["weight_decay"]),
            patience=int(params["patience"]),
            min_delta=float(params["min_delta"]),
        )

        score = float(va_hist[-1])
        if best is None or score < best["val_loss"]:
            best = {
                "trial": trial,
                "params": params,
                "model": model,
                "train_history": tr_hist,
                "val_history": va_hist,
                "val_loss": score,
            }

    return best


def rnn_random_search(
    space: dict[str, list[Any]],
    n_iter: int,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    seed: int = 42,
):
    """Random search over RNN hyperparameters."""

    rng = random.Random(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    best = None

    for trial in range(1, n_iter + 1):
        params = _sample(space, rng)

        model = RNNRegressor(
            input_size=1,
            hidden_size=int(params["hidden_size"]),
            num_layers=int(params["num_layers"]),
            nonlinearity=str(params["nonlinearity"]),
            dropout=float(params["dropout"]),
            bidirectional=bool(params["bidirectional"]),
        )

        tr_hist, va_hist = model.fit(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            epochs=int(params["epochs"]),
            lr=float(params["lr"]),
            weight_decay=float(params["weight_decay"]),
            patience=int(params["patience"]),
            min_delta=float(params["min_delta"]),
        )

        score = float(va_hist[-1])
        if best is None or score < best["val_loss"]:
            best = {
                "trial": trial,
                "params": params,
                "model": model,
                "train_history": tr_hist,
                "val_history": va_hist,
                "val_loss": score,
            }

    return best
