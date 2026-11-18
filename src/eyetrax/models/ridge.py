"""Ridge regression model backend for EyeTrax gaze estimation.

This module implements `RidgeModel`, wrapping scikit-learn's L2-regularized
linear least squares regressor (`Ridge`) to map standardized eye and head-pose
features to 2D screen coordinates, and registers it under `"ridge"`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

from . import register_model
from .base import BaseModel


class RidgeModel(BaseModel):
    """L2-regularized linear regression model for 2D gaze coordinate prediction."""

    def __init__(self, alpha: float = 1.0) -> None:
        """Initialize the Ridge gaze estimator.

        Args:
            alpha: Regularization strength; larger values specify stronger L2 penalty.
        """
        super().__init__()
        self._init_native(alpha=alpha)

    def _init_native(self, **kw: Any) -> None:
        """Instantiate the underlying scikit-learn Ridge regressor."""
        self.model = Ridge(**kw)

    def _native_train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the Ridge model on standardized features `X` and target coordinates `y`."""
        self.model.fit(X, y)

    def _native_predict(self, X: np.ndarray) -> np.ndarray:
        """Predict 2D screen coordinates from standardized feature matrix `X`."""
        return self.model.predict(X)


# Register backend as the default lookup via create_model("ridge")
register_model("ridge", RidgeModel)
