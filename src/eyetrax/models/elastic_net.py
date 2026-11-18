"""ElasticNet regression model backend for EyeTrax gaze estimation.

This module implements `ElasticNetModel`, wrapping scikit-learn's ElasticNet
(combined L1 and L2 regularization) to map standardized eye and head-pose
features to 2D screen coordinates, and registers it under `"elastic_net"`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import ElasticNet

from . import register_model
from .base import BaseModel


class ElasticNetModel(BaseModel):
    """Linear gaze regression model regularized with both L1 and L2 penalties."""

    def __init__(self, *, alpha: float = 1.0, l1_ratio: float = 0.5) -> None:
        """Initialize the ElasticNet gaze estimator.

        Args:
            alpha: Constant multiplier for the combined L1/L2 penalty terms.
            l1_ratio: Mixing parameter in [0, 1] (0 = pure Ridge/L2, 1 = pure Lasso/L1).
        """
        super().__init__()
        self._init_native(alpha=alpha, l1_ratio=l1_ratio)

    def _init_native(self, **kw: Any) -> None:
        """Instantiate the underlying scikit-learn ElasticNet regressor."""
        self.model = ElasticNet(**kw)

    def _native_train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the ElasticNet model on standardized features `X` and target coordinates `y`."""
        self.model.fit(X, y)

    def _native_predict(self, X: np.ndarray) -> np.ndarray:
        """Predict 2D screen coordinates from standardized feature matrix `X`."""
        return self.model.predict(X)


# Register backend for lookup via create_model("elastic_net")
register_model("elastic_net", ElasticNetModel)
