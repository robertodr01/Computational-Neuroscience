"""Random search for Echo State Network hyperparameters."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch

try:
    from .esn import ESNParams, EchoStateNetwork
except ImportError:  # pragma: no cover - supports notebook-style imports
    from esn import ESNParams, EchoStateNetwork


def _sample(space: dict[str, list[Any]], rng: random.Random) -> dict[str, Any]:
    return {key: rng.choice(values) for key, values in space.items()}


def random_search_esn(
    space: dict[str, list[Any]],
    random_iter: int,
    n_runs: int,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    seed: int = 42,
):
    """Evaluate random ESN configurations and keep the best by mean val MSE."""

    rng = random.Random(seed)
    best = None

    for trial in range(1, random_iter + 1):
        params = _sample(space, rng)
        train_losses: list[float] = []
        val_losses: list[float] = []
        best_trial_model = None

        for run in range(n_runs):
            esn_params = ESNParams(
                input_dim=x_train.shape[1],
                output_dim=y_train.shape[1],
                hidden_dim=int(params["hidden_dim"]),
                spectral_radius=float(params["spectral_radius"]),
                input_scale=float(params["input_scale"]),
                bias_scale=float(params["bias_scale"]),
                scale_mode=str(params["scale_mode"]),
                washout=int(params["washout"]),
                seed=seed + trial * 100 + run,
            )

            model = EchoStateNetwork(esn_params)
            model.fit(x_train, y_train, l2=float(params["l2"]))

            y_train_hat = model.predict(x_train)
            y_val_hat = model.predict(x_val)
            train_loss = float(model.mse(y_train, y_train_hat).item())
            val_loss = float(model.mse(y_val, y_val_hat).item())

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            best_trial_model = model

        summary = {
            "trial": trial,
            "params": params,
            "model": best_trial_model,
            "train_mean": float(np.mean(train_losses)),
            "train_std": float(np.std(train_losses)),
            "val_mean": float(np.mean(val_losses)),
            "val_std": float(np.std(val_losses)),
        }

        if best is None or summary["val_mean"] < best["val_mean"]:
            best = summary

    return best
