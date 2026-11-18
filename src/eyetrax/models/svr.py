"""Linear Support Vector Regression (SVR) model backend for EyeTrax gaze estimation.

This module implements `LinearSVRModel`, fitting two independent scikit-learn
`LinearSVR` regressors—one for horizontal (`x`) and one for vertical (`y`)
screen coordinates—to map standardized eye and head-pose features to 2D gaze
positions. It registers the model under both `"linear_svr"` and `"svr"`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.svm import LinearSVR

from . import register_model
from .base import BaseModel


class LinearSVRModel(BaseModel):
    """Dual-output Linear Support Vector Regressor for 2D gaze coordinate prediction."""

    def __init__(
        self,
        *,
        C: float = 5.0,
        epsilon: float = 5.0,
        loss: str = "epsilon_insensitive",
        fit_intercept: bool = True,
    ) -> None:
        """Initialize the LinearSVR gaze estimator.

        Args:
            C: Regularization parameter controlling the penalty for errors beyond epsilon.
            epsilon: Epsilon-tube margin within which no penalty is associated in the loss.
            loss: Loss function type ("epsilon_insensitive" or "squared_epsilon_insensitive").
            fit_intercept: Whether to calculate an intercept term for each coordinate axis.
        """
        super().__init__()
        self._init_native(
            C=C,
            epsilon=epsilon,
            loss=loss,
            fit_intercept=fit_intercept,
        )

    def _init_native(self, **kwargs: Any) -> None:
        """Store a template `LinearSVR` instance holding the configured hyperparameters."""
        self._template = LinearSVR(**kwargs)

    def _native_train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit separate LinearSVR models for X and Y screen coordinates.

        Args:
            X: Standardized feature matrix of shape (n_samples, n_features).
            y: Target screen coordinates of shape (n_samples, 2).
        """
        y = y.reshape(-1, 2)

        # LinearSVR supports 1D targets only, so clone the template per screen axis
        self.model_x = LinearSVR(**self._template.get_params())
        self.model_y = LinearSVR(**self._template.get_params())

        self.model_x.fit(X, y[:, 0])
        self.model_y.fit(X, y[:, 1])

    def _native_predict(self, X: np.ndarray) -> np.ndarray:
        """Predict 2D screen coordinates by stacking X and Y regressor outputs.

        Args:
            X: Standardized feature matrix of shape (n_samples, n_features).

        Returns:
            Predicted screen coordinates of shape (n_samples, 2).
        """
        x_pred = self.model_x.predict(X)
        y_pred = self.model_y.predict(X)
        return np.column_stack((x_pred, y_pred))


# Register backend under both primary and alias keys
register_model("linear_svr", LinearSVRModel)
register_model("svr", LinearSVRModel)
