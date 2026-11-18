"""Abstract class for EyeTrax gaze estimation regression backends.

This module defines `BaseModel`, which standardizes feature normalization via
scikit-learn's `StandardScaler`, optional per-feature weighting, model training
and inference hooks, and pickle-based model serialization.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler


class BaseModel(ABC):
    """Common interface and preprocessing pipeline every gaze-prediction model must implement."""

    def __init__(self) -> None:
        """Initialize the model with a feature standardizer."""
        self.scaler = StandardScaler()

    @abstractmethod
    def _init_native(self, **kwargs: Any) -> None:
        """Instantiate the underlying backend estimator (e.g., Ridge, SVR, MLP)."""
        ...

    @abstractmethod
    def _native_train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the underlying estimator on standardized feature matrix `X` and targets `y`."""
        ...

    @abstractmethod
    def _native_predict(self, X: np.ndarray) -> np.ndarray:
        """Predict 2D gaze coordinates from standardized feature matrix `X`."""
        ...

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_scaling: np.ndarray | None = None,
    ) -> None:
        """Standardize input features, apply optional feature weights, and train the model.

        Args:
            X: Training feature matrix of shape (n_samples, n_features).
            y: Target screen coordinates of shape (n_samples, 2).
            variable_scaling: Optional 1D multiplicative weights per feature column.
        """
        self.variable_scaling = variable_scaling
        Xs = self.scaler.fit_transform(X)
        if variable_scaling is not None:
            Xs *= variable_scaling
        self._native_train(Xs, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Transform input features using the fitted scaler and predict gaze coordinates.

        Args:
            X: Input feature matrix of shape (n_samples, n_features).

        Returns:
            Predicted screen coordinates of shape (n_samples, 2).
        """
        Xs = self.scaler.transform(X)
        # Re-apply feature weights learned during training if present
        if getattr(self, "variable_scaling", None) is not None:
            Xs *= self.variable_scaling
        return self._native_predict(Xs)

    def save(self, path: str | Path) -> None:
        """Serialize the complete model instance (including scaler state) to disk.

        Args:
            path: Destination file path for the pickled model.
        """
        with Path(path).open("wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, path: str | Path) -> "BaseModel":
        """Deserialize and return a previously saved `BaseModel` instance.

        Args:
            path: Path to the saved `.pkl` model file.

        Returns:
            Loaded `BaseModel` instance ready for inference.
        """
        with Path(path).open("rb") as fh:
            return pickle.load(fh)
