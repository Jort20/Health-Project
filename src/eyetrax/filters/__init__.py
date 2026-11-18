"""Gaze coordinate smoothing filters and Kalman filter factory utilities.

This module uses EyeTrax's spatial-temporal smoothing implementations
(Kalman, Kalman-EMA, KDE, and no-op pass-through) along with `make_kalman`, a
factory function that builds a constant-velocity 2D OpenCV Kalman filter for
tracking screen gaze position and velocity.
"""

from __future__ import annotations

import cv2
import numpy as np


def make_kalman(
    state_dim: int = 4,
    meas_dim: int = 2,
    dt: float = 1.0,
    process_var: float = 50.0,
    measurement_var: float = 0.2,
    init_state: np.ndarray | None = None,
) -> cv2.KalmanFilter:
    """Create and configure a 2D constant-velocity OpenCV KalmanFilter.

    The default state vector tracks [x, y, vx, vy] while measurements observe
    raw gaze screen coordinates [x, y].

    Args:
        state_dim: Dimensionality of the internal state vector (default 4).
        meas_dim: Dimensionality of the measurement vector (default 2).
        dt: Time step between consecutive frames.
        process_var: Diagonal variance scaling the process noise covariance.
        measurement_var: Diagonal variance scaling the measurement noise covariance.
        init_state: Optional initial state array of shape (state_dim,).

    Returns:
        Configured cv2.KalmanFilter instance ready for predict/correct steps.
    """
    kf = cv2.KalmanFilter(state_dim, meas_dim)

    # Constant-velocity state transition model: pos_next = pos + vel * dt
    kf.transitionMatrix = np.array(
        [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
    )
    # Observe only position (x, y) from the 4D state vector
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)

    # Initialize process, measurement, and posterior error covariance matrices
    kf.processNoiseCov = np.eye(state_dim, dtype=np.float32) * process_var
    kf.measurementNoiseCov = np.eye(meas_dim, dtype=np.float32) * measurement_var
    kf.errorCovPost = np.eye(state_dim, dtype=np.float32)

    kf.statePre = np.zeros((state_dim, 1), np.float32)
    kf.statePost = np.zeros((state_dim, 1), np.float32)

    # override default zero initial state
    if init_state is not None:
        init_state = np.asarray(init_state, np.float32).reshape(state_dim, 1)
        kf.statePre[:] = init_state
        kf.statePost[:] = init_state

    return kf


from .base import BaseSmoother  
from .kalman import KalmanSmoother 
from .kalman_ema import KalmanEMASmoother  
from .kde import KDESmoother 
from .noop import NoSmoother  

__all__ = [
    "make_kalman",
    "BaseSmoother",
    "KalmanSmoother",
    "KalmanEMASmoother",
    "KDESmoother",
    "NoSmoother",
]
