"""Dense grid gaze calibration routine for high spatial accuracy.

This module implements a configurable N x M grid calibration workflow (defaulting
to a 5x5 serpentine layout). It verifies face detection, presents sequential target
dots across the screen, records eye features at each point, and trains the
GazeEstimator on the collected samples.
"""

import cv2
import numpy as np

from eyetrax.calibration.common import (
    _pulse_and_capture,
    compute_grid_points_from_shape,
    wait_for_face_and_countdown,
)
from eyetrax.gaze import GazeEstimator
from eyetrax.utils.screen import get_screen_size
from eyetrax.utils.video import open_camera


def run_dense_grid_calibration(
    gaze_estimator: GazeEstimator,
    *,
    rows: int = 5,
    cols: int = 5,
    margin_ratio: float = 0.10,
    order: str = "serpentine",
    pulse_d: float = 0.9,
    cd_d: float = 0.9,
    camera_index: int = 0,
) -> None:
    """Calibrate a GazeEstimator using a dense rows x cols grid of target points.

    Args:
        gaze_estimator: GazeEstimator instance to train in-place.
        rows: Number of vertical grid rows.
        cols: Number of horizontal grid columns.
        margin_ratio: Fractional edge padding keeping points away from screen borders.
        order: Grid traversal pattern ("default" row-major or "serpentine" snake path).
        pulse_d: Duration in seconds of the initial fixation pulse per point.
        cd_d: Duration in seconds of the feature-capture phase per point.
        camera_index: Webcam device index.
    """
    sw, sh = get_screen_size()

    # Open webcam and wait for stable face/eye detection before starting
    cap = open_camera(camera_index)
    if not wait_for_face_and_countdown(cap, gaze_estimator, sw, sh, 2):
        cap.release()
        cv2.destroyAllWindows()
        return

    # Generate grid target coordinates and run the visual pulse-and-capture loop
    pts = compute_grid_points_from_shape(
        rows, cols, sw, sh, margin_ratio=margin_ratio, order=order
    )
    res = _pulse_and_capture(
        gaze_estimator, cap, pts, sw, sh, pulse_d=pulse_d, cd_d=cd_d
    )
    cap.release()
    cv2.destroyAllWindows()

    # Abort if user cancelled early via ESC
    if res is None:
        return

    # Fit the regression model on all valid captured eye features
    feats, targs = res
    if feats:
        gaze_estimator.train(np.array(feats), np.array(targs))
        print(f"[dense_grid] Calibrated with {len(feats)} samples from {rows}x{cols} grid")
