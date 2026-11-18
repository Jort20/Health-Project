"""Lightweight Multi-Layer Perceptron (MLP) regressor backend for EyeTrax gaze estimation.

This module implements `TinyMLPModel`, wrapping scikit-learn's `MLPRegressor`
with a compact two-layer neural network architecture (default `(64, 32)`) to
learn non-linear mappings from standardized eye and head-pose features to 2D
screen coordinates. It registers the model under `"tiny_mlp"`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.neural_network import MLPRegressor

from . import register_model
from .base import BaseModel


class TinyMLPModel(BaseModel):
    """Compact feed-forward neural network regressor for non-linear 2D gaze prediction."""

    def __init__(
        self,
        *,
        hidden_layer_sizes: tuple[int, ...] = (64, 32),
        activation: str = "relu",
        alpha: float = 1e-4,
        learning_rate_init: float = 1e-3,
        max_iter: int = 500,
        early_stopping: bool = True,
    ) -> None:
        """Initialize the TinyMLP gaze estimator.

        Args:
            hidden_layer_sizes: Tuple specifying neuron counts per hidden layer.
            activation: Activation function for hidden layers ("relu", "tanh", "logistic").
            alpha: L2 weight regularization strength.
            learning_rate_init: Initial learning rate for the Adam optimizer.
            max_iter: Maximum number of training epochs.
            early_stopping: Whether to hold out a validation split to halt training
                when validation score stops improving.
        """
        super().__init__()
        self._init_native(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=early_stopping,
        )

    def _init_native(self, **kw: Any) -> None:
        """Instantiate the underlying scikit-learn MLPRegressor with deterministic seeding."""
        self.model = MLPRegressor(
            solver="adam",
            batch_size="auto",
            random_state=0,
            verbose=False,
            **kw,
        )

    def _native_train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the MLP network on standardized features `X` and target coordinates `y`."""
        self.model.fit(X, y)

    def _native_predict(self, X: np.ndarray) -> np.ndarray:
        """Predict 2D screen coordinates from standardized feature matrix `X`."""
        return self.model.predict(X)


# Register backend for lookup via create_model("tiny_mlp")
register_model("tiny_mlp", TinyMLPModel)
