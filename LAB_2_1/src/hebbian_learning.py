"""Hebbian-family learning rules for 2D firing-rate neurons.

The implementation mirrors the behavior of the original LAB2_1 reference,
while exposing a cleaner API for the rewritten notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

RuleName = Literal["hebb", "oja", "subtractive", "bcm", "covariance", "sub-norm", "BCM", "cov"]


@dataclass
class FitResult:
    final_w: np.ndarray
    history: np.ndarray
    convergence_epoch: int
    final_theta: float | None = None


def load_dataset(csv_path: str | Path) -> np.ndarray:
    """Load a 2xN dataset from CSV."""

    data = np.loadtxt(csv_path, delimiter=",", dtype=float)
    if data.ndim != 2:
        raise ValueError("Dataset must be 2D")
    return data


def zscore_per_feature(data: np.ndarray) -> np.ndarray:
    """Apply the same normalization used in the original notebook (StandardScaler)."""

    scaler = StandardScaler()
    return scaler.fit_transform(data.T).T


def principal_eigenvector(data: np.ndarray) -> np.ndarray:
    """Compute principal eigenvector exactly as in the original notebook."""

    cov = np.cov(data)
    eigvals, eigvecs = np.linalg.eig(cov)
    return eigvecs.T[np.argmax(eigvals)]


def _normalize_rule_name(rule: str) -> str:
    mapping = {
        "subtractive": "sub-norm",
        "bcm": "BCM",
        "covariance": "cov",
    }
    return mapping.get(rule, rule)


def fit_rule(
    data: np.ndarray,
    rule: RuleName,
    epochs: int,
    lr: float,
    tol: float,
    seed: int = 42,
    alpha: float = 0.01,
    theta0: float | None = None,
) -> FitResult:
    """Train a two-weight neuron with one of the supported rules.

    Behavior matches the reference implementation from the original repository.
    """

    if data.shape[0] != 2:
        raise ValueError("The implementation expects a 2xN matrix")

    rule_name = _normalize_rule_name(str(rule))

    np.random.seed(seed)
    w = np.random.uniform(-1, 1, size=2)
    w_old = w.copy()

    theta = float(np.random.uniform(0, 1) if theta0 is None else theta0)

    history: list[np.ndarray] = []
    convergence = int(epochs)

    for epoch in range(1, epochs + 1):
        samples = shuffle(data.T)

        for x in samples:
            v = w @ x

            if rule_name == "hebb":
                w = w + lr * (v * x)
            elif rule_name == "oja":
                w = w + lr * ((v * x) - alpha * (v**2) * w)
            elif rule_name == "sub-norm":
                nu = data.shape[0]
                n = np.ones(nu)
                w = w + lr * (v * x - (v * (n @ x) * n) / nu)
            elif rule_name == "BCM":
                w = w + lr * (v * x * (v - theta))
                theta = theta + lr * 10 * (v**2 - theta)
            elif rule_name == "cov":
                c = np.cov(data)
                # Keep this update inside sample loop to mirror the original behavior.
                w = w + lr * (c @ w)
            else:
                raise ValueError(f"Unsupported rule: {rule}")

        if np.linalg.norm(w - w_old) < tol:
            history.append(w.copy())
            convergence = epoch
            break

        # Intentionally keep assignment semantics similar to the old implementation.
        w_old = w
        history.append(w.copy())

    return FitResult(
        final_w=w,
        history=np.asarray(history),
        convergence_epoch=convergence,
        final_theta=theta,
    )


def save_history(history: np.ndarray, path: str | Path) -> None:
    """Save weight history using the same npz style as the reference notebook."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, *history)


def plot_points_with_vectors(
    data: np.ndarray,
    learned_w: np.ndarray,
    principal_vec: np.ndarray,
    title: str,
    output_path: str | Path,
    anchor: tuple[float, float] = (0.0, 0.0),
) -> None:
    """Plot samples, learned weights and principal eigenvector (reference style)."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(data[0], data[1], "o", markersize=3, label="Data points")
    ax.set_xlabel("u[0]")
    ax.set_ylabel("u[1]")
    ax.set_title(title)

    width = 0.003 if anchor == (0.1, 0.2) else 0.002
    ax.quiver(anchor[0], anchor[1], learned_w[0], learned_w[1], color="green", width=width, label="Weight vector (w)")
    ax.quiver(
        anchor[0],
        anchor[1],
        principal_vec[0],
        principal_vec[1],
        color="red",
        width=width,
        label="Principal eigenvector",
    )

    ax.legend()

    if output_path.exists():
        output_path.unlink()
    plt.savefig(output_path)
    plt.show()


def plot_weight_trajectory(history: np.ndarray, title: str, output_path: str | Path) -> None:
    """Plot w[0], w[1], and ||w|| with the same layout used in the reference."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tspan = np.arange(0, history.shape[0], 1)
    norm_hist = np.zeros(history.shape[0])
    for i, w in enumerate(history):
        norm_hist[i] = np.linalg.norm(w)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title, fontsize=16)

    axs[0].plot(tspan, history.T[0])
    axs[0].set_title("w[0] over time")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("w[0]")

    axs[1].plot(tspan, history.T[1])
    axs[1].set_title("w[1] over time")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("w[1]")

    axs[2].plot(tspan, norm_hist)
    axs[2].set_title("Norm of W over time")
    axs[2].set_xlabel("Epochs")
    axs[2].set_ylabel("Norm of W")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path.exists():
        output_path.unlink()
    plt.savefig(output_path)
    plt.show()
