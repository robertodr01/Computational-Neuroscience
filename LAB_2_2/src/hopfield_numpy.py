"""Hopfield network utilities for binary image reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


@dataclass
class RecallResult:
    recalled: np.ndarray
    energies: np.ndarray
    overlaps: np.ndarray
    epochs_run: int


def load_patterns(folder: str | Path, pattern_ids: list[int] | None = None) -> np.ndarray:
    """Load flattened pattern vectors from CSV files."""

    folder = Path(folder)
    if pattern_ids is None:
        files = sorted(folder.glob("p*.csv"))
    else:
        files = [folder / f"p{idx}.csv" for idx in pattern_ids]

    vectors = []
    for file_path in files:
        # Mirror the original lab pipeline: transpose each 32x32 pattern before flattening.
        mat = np.loadtxt(file_path, delimiter=",", dtype=float).reshape(32, 32)
        vec = mat.T.reshape(-1)
        vec = np.where(vec >= 0, 1.0, -1.0)
        vectors.append(vec)

    if not vectors:
        raise ValueError("No patterns found")

    return np.stack(vectors, axis=0)


class HopfieldNetwork:
    """Classic Hopfield network with asynchronous updates."""

    def __init__(self, patterns: np.ndarray):
        if patterns.ndim != 2:
            raise ValueError("patterns must be a matrix [num_patterns, num_units]")
        self.patterns = patterns.astype(float)
        self.num_patterns, self.size = self.patterns.shape
        self.weights = self._build_weights(self.patterns)

    @staticmethod
    def _build_weights(patterns: np.ndarray) -> np.ndarray:
        n_units = patterns.shape[1]
        w = patterns.T @ patterns
        np.fill_diagonal(w, 0.0)
        return w / n_units

    def incremental_store(self, pattern: np.ndarray) -> None:
        pattern = np.where(pattern.reshape(-1) >= 0, 1.0, -1.0)
        if pattern.size != self.size:
            raise ValueError("Pattern size mismatch")
        self.patterns = np.vstack([self.patterns, pattern])
        self.num_patterns += 1
        self.weights += np.outer(pattern, pattern) / self.size
        np.fill_diagonal(self.weights, 0.0)

    def energy(self, state: np.ndarray) -> float:
        x = state.reshape(-1)
        return float(-0.5 * x @ self.weights @ x)

    def overlap(self, state: np.ndarray, target: np.ndarray) -> float:
        x = state.reshape(-1)
        y = target.reshape(-1)
        return float(np.dot(x, y) / self.size)

    def recall(
        self,
        initial_state: np.ndarray,
        target: np.ndarray,
        bias: float = 0.5,
        max_epochs: int = 10,
        seed: int | None = None,
    ) -> RecallResult:
        """Run asynchronous recall until convergence or max epochs."""

        if seed is not None:
            torch.manual_seed(seed)
        x = np.where(initial_state.reshape(-1) >= 0, 1.0, -1.0)
        target = np.where(target.reshape(-1) >= 0, 1.0, -1.0)

        energies: list[float] = []
        overlaps: list[float] = []

        for epoch in range(max_epochs):
            before = x.copy()
            for idx in torch.randperm(self.size).tolist():
                field = self.weights[idx] @ x + bias
                x[idx] = 1.0 if field > 0 else -1.0

                energies.append(self.energy(x))
                overlaps.append(self.overlap(x, target))

            if np.array_equal(before, x):
                return RecallResult(x.copy(), np.asarray(energies), np.asarray(overlaps), epoch + 1)

        return RecallResult(x.copy(), np.asarray(energies), np.asarray(overlaps), epoch + 1)


def distort_pattern(pattern: np.ndarray, ratio: float, seed: int = 42) -> np.ndarray:
    """Flip a percentage of bits."""

    if not (0 <= ratio <= 1):
        raise ValueError("ratio must be in [0, 1]")

    x = np.where(pattern.reshape(-1) >= 0, 1.0, -1.0).copy()

    k = int(ratio * x.size)
    if k == 0:
        return x

    if seed is not None:
        torch.manual_seed(seed)
    idx = torch.randperm(x.size)[:k].numpy()
    x[idx] *= -1.0
    return x


def plot_reconstruction(original: np.ndarray, noisy: np.ndarray, recalled: np.ndarray, title: str, path: str | Path) -> None:
    """Visual comparison between original, noisy and recalled pattern."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.3))
    for ax, data, name in [
        (axes[0], original, "Original"),
        (axes[1], noisy, "Noisy"),
        (axes[2], recalled, "Recalled"),
    ]:
        ax.imshow(data.reshape(32, 32), cmap="gray")
        ax.set_title(name)
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.show()


def plot_dynamics(energies: np.ndarray, overlaps: np.ndarray, title: str, path: str | Path) -> None:
    """Plot Hopfield energy and overlap trajectories."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.1))
    axes[0].plot(energies, color="#bcbd22")
    axes[0].set_title("Energy")
    axes[0].set_xlabel("Update step")
    axes[0].grid(alpha=0.25)

    axes[1].plot(overlaps, color="#9467bd")
    axes[1].set_title("Overlap")
    axes[1].set_xlabel("Update step")
    axes[1].grid(alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.show()
