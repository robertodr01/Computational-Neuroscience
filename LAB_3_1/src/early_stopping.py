"""Early stopping helper for PyTorch models."""

from __future__ import annotations

import copy


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 30, min_delta: float = 1e-8):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state_dict = None

    def step(self, val_loss: float, model) -> bool:
        """Return True when training should stop."""

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = float(val_loss)
            self.counter = 0
            self.best_state_dict = copy.deepcopy(model.state_dict())
            return False

        self.counter += 1
        return self.counter >= self.patience

    def restore(self, model) -> None:
        """Load the best observed model state."""

        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
