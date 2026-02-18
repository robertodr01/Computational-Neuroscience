"""Echo State Network implementation for sequence regression."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ESNParams:
    input_dim: int
    hidden_dim: int
    output_dim: int
    spectral_radius: float
    input_scale: float
    bias_scale: float
    scale_mode: str = "norm"
    washout: int = 0
    seed: int = 42


class EchoStateNetwork:
    """Reservoir computing model with closed-form linear readout."""

    def __init__(self, params: ESNParams):
        self.params = params
        g = torch.Generator().manual_seed(params.seed)

        self.Win = torch.empty(params.hidden_dim, params.input_dim).uniform_(-1, 1, generator=g)
        self.W = torch.empty(params.hidden_dim, params.hidden_dim).uniform_(-1, 1, generator=g)
        self.b = torch.empty(params.hidden_dim).uniform_(-1, 1, generator=g)

        self.Win = self._rescale(self.Win, params.input_scale, params.scale_mode)
        self.b = self._rescale(self.b, params.bias_scale, params.scale_mode)

        eigs = torch.linalg.eigvals(self.W).abs()
        radius = torch.max(eigs).real
        self.W = self.W * (params.spectral_radius / radius)

        self.Wout = torch.zeros(params.hidden_dim + 1, params.output_dim)

    @staticmethod
    def _rescale(tensor: torch.Tensor, scale: float, mode: str) -> torch.Tensor:
        if mode == "range":
            return tensor * scale
        if mode == "norm":
            norm = torch.norm(tensor)
            if norm == 0:
                return tensor
            return tensor * (scale / norm)
        raise ValueError(f"Unsupported scale_mode: {mode}")

    def _collect_states(self, x: torch.Tensor) -> torch.Tensor:
        """Compute reservoir states for an input sequence x[T, D]."""

        x = x.reshape(-1, self.params.input_dim)
        states = []
        h = torch.zeros(self.params.hidden_dim)

        with torch.no_grad():
            for t in range(x.shape[0]):
                h = torch.tanh(self.W @ h + self.Win @ x[t] + self.b)
                states.append(h.clone())

        return torch.stack(states, dim=0)

    def fit(self, x_train: torch.Tensor, y_train: torch.Tensor, l2: float = 0.0) -> None:
        """Fit readout weights with ridge regression."""

        states = self._collect_states(x_train)
        y_train = y_train.reshape(-1, self.params.output_dim)

        states = states[self.params.washout :]
        y_train = y_train[self.params.washout :]

        design = torch.cat([states, torch.ones(states.shape[0], 1)], dim=1)

        if l2 <= 0:
            self.Wout = torch.linalg.pinv(design) @ y_train
        else:
            identity = torch.eye(design.shape[1])
            self.Wout = torch.linalg.solve(design.T @ design + l2 * identity, design.T @ y_train)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        states = self._collect_states(x)
        design = torch.cat([states, torch.ones(states.shape[0], 1)], dim=1)
        return design @ self.Wout

    @staticmethod
    def mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return torch.mean((y_true - y_pred) ** 2)

    def export_parameters(self) -> dict[str, torch.Tensor]:
        return {
            "Win": self.Win,
            "W": self.W,
            "b": self.b,
            "Wout": self.Wout,
        }
