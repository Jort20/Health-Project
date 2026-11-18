"""Hybrid Kalman and Exponential Moving Average (EMA) gaze-coordinate smoother.

This module provides `KalmanEMASmoother`, which first filters raw gaze predictions
through a 2D Kalman filter and subsequently applies an Exponential Moving Average
pass to further suppress high-frequency cursor jitter during fixations.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2

from .kalman import KalmanSmoother


class KalmanEMASmoother(KalmanSmoother):
    """Apply an Exponential Moving Average (EMA) pass on top of Kalman filter output.

    `ema_alpha` controls the smoothing weight applied to the previous EMA state:
        - 0.0: EMA disabled (returns pure Kalman filter output).
        - Closer to 1.0: Stronger jitter suppression with increased cursor latency.
    """

    def __init__(
        self, kf: Optional[cv2.KalmanFilter] = None, ema_alpha: float = 0.25
    ) -> None:
        """Initialize the hybrid Kalman-EMA smoother.

        Args:
            kf: Optional pre-configured OpenCV KalmanFilter instance.
            ema_alpha: Smoothing retention factor in [0.0, 1.0].

        Raises:
            ValueError: If `ema_alpha` falls outside [0.0, 1.0].
        """
        super().__init__(kf=kf)

        if not 0.0 <= ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha must be in [0.0, 1.0], got {ema_alpha}")

        self.ema_alpha = float(ema_alpha)
        self.ema_x: float | None = None
        self.ema_y: float | None = None

    def step(self, x: int, y: int) -> Tuple[int, int]:
        """Filter raw (x, y) gaze coordinates via Kalman prediction followed by EMA.

        Args:
            x: Raw predicted horizontal screen coordinate in pixels.
            y: Raw predicted vertical screen coordinate in pixels.

        Returns:
            Tuple of smoothed (x, y) screen pixel coordinates.
        """
        # Run base Kalman predict/correct step first
        kx, ky = super().step(x, y)

        a = self.ema_alpha
        if a == 0.0:
            return kx, ky

        # Seed EMA state on the first frame, then blend previous and current estimates
        if self.ema_x is None or self.ema_y is None:
            self.ema_x = float(kx)
            self.ema_y = float(ky)
        else:
            self.ema_x = a * self.ema_x + (1.0 - a) * float(kx)
            self.ema_y = a * self.ema_y + (1.0 - a) * float(ky)

        return int(self.ema_x), int(self.ema_y)
