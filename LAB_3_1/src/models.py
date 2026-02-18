"""TDNN and RNN models for sequence-to-sequence regression."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

try:
    from .early_stopping import EarlyStopping
except ImportError:  # pragma: no cover - supports notebook-style imports
    from early_stopping import EarlyStopping


def _activation_from_name(name: str) -> nn.Module:
    if not hasattr(nn, name):
        raise ValueError(f"Unknown activation: {name}")
    return getattr(nn, name)()


class TDNNRegressor(nn.Module):
    """Configurable temporal convolutional regressor."""

    def __init__(
        self,
        input_channels: int,
        conv_channels: Sequence[int],
        kernel_sizes: Sequence[int],
        dilations: Sequence[int],
        strides: Sequence[int],
        activations: Sequence[str],
        fc_hidden: int = 32,
    ):
        super().__init__()

        lengths = [len(conv_channels), len(kernel_sizes), len(dilations), len(strides), len(activations)]
        if len(set(lengths)) != 1:
            raise ValueError("All TDNN parameter lists must have the same length")

        self.convs = nn.ModuleList()
        self.acts = nn.ModuleList()

        in_ch = input_channels
        for out_ch, k, d, s, act_name in zip(conv_channels, kernel_sizes, dilations, strides, activations):
            pad = (d * (k - 1)) // 2
            self.convs.append(nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=k, dilation=d, stride=s, padding=pad))
            self.acts.append(_activation_from_name(act_name))
            in_ch = out_ch

        self.fc1 = nn.Linear(in_ch, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for conv, act in zip(self.convs, self.acts):
            out = act(conv(out))
        out = out.transpose(1, 2)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out

    def fit(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_val: torch.Tensor,
        y_val: torch.Tensor,
        epochs: int,
        lr: float,
        weight_decay: float,
        patience: int,
        min_delta: float,
    ) -> tuple[list[float], list[float]]:
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()
        stopper = EarlyStopping(patience=patience, min_delta=min_delta)

        train_history: list[float] = []
        val_history: list[float] = []

        for _ in range(epochs):
            self.train()
            optimizer.zero_grad()
            pred = self(x_train)
            train_loss = loss_fn(pred, y_train)
            train_loss.backward()
            optimizer.step()

            self.eval()
            with torch.no_grad():
                val_loss = loss_fn(self(x_val), y_val)

            train_history.append(float(train_loss.item()))
            val_history.append(float(val_loss.item()))

            if stopper.step(float(val_loss.item()), self):
                break

        stopper.restore(self)
        return train_history, val_history


class RNNRegressor(nn.Module):
    """Vanilla recurrent neural network for seq2seq regression."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        nonlinearity: str,
        dropout: float,
        bidirectional: bool,
    ):
        super().__init__()

        if dropout > 0.0 and num_layers == 1:
            num_layers = 2

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

        factor = 2 if bidirectional else 1
        self.out = nn.Linear(hidden_size * factor, 1)

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        out, h = self.rnn(x, h0)
        out = self.out(out)
        return out, h

    def fit(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_val: torch.Tensor,
        y_val: torch.Tensor,
        epochs: int,
        lr: float,
        weight_decay: float,
        patience: int,
        min_delta: float,
    ) -> tuple[list[float], list[float]]:
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()
        stopper = EarlyStopping(patience=patience, min_delta=min_delta)

        train_history: list[float] = []
        val_history: list[float] = []

        for _ in range(epochs):
            self.train()
            optimizer.zero_grad()
            pred, _ = self(x_train)
            train_loss = loss_fn(pred, y_train)
            train_loss.backward()
            optimizer.step()

            self.eval()
            with torch.no_grad():
                val_pred, _ = self(x_val)
                val_loss = loss_fn(val_pred, y_val)

            train_history.append(float(train_loss.item()))
            val_history.append(float(val_loss.item()))

            if stopper.step(float(val_loss.item()), self):
                break

        stopper.restore(self)
        return train_history, val_history
