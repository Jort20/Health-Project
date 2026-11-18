"""Standard 9-point gaze calibration routine covering a full 3x3 screen grid.

This module implements the classic nine-point calibration workflow starting at
the screen center, followed by the four corners and four edge midpoints. It
verifies face detection, records eye features at each target, and trains the
GazeEstimator on the collected samples.
"""

import cv2
import numpy as np

from eyetrax.calibration.common import (
    _pulse_and_capture,
    compute_grid_points,
    wait_for_face_and_countdown,
)
from eyetrax.gaze import GazeEstimator
from eyetrax.utils.screen import get_screen_size
from eyetrax.utils.video import open_camera


def run_9_point_calibration(
    gaze_estimator: GazeEstimator, camera_index: int = 0
) -> None:
    """Calibrate a GazeEstimator using a standard 3x3 (9-point) screen grid.

    Args:
        gaze_estimator: GazeEstimator instance to train in-place.
        camera_index: Webcam device index.
    """
    sw, sh = get_screen_size()

    # Open webcam and wait for stable face/eye detection before starting
    cap = open_camera(camera_index)
    if not wait_for_face_and_countdown(cap, gaze_estimator, sw, sh, 2):
        cap.release()
        cv2.destroyAllWindows()
        return

    # Define 3x3 grid order: center, four corners, then four edge midpoints
    order = [
        (1, 1),
        (0, 0),
        (2, 0),
        (0, 2),
        (2, 2),
        (1, 0),
        (0, 1),
        (2, 1),
        (1, 2),
    ]
    pts = compute_grid_points(order, sw, sh)

    # Display sequential targets and capture eye features at each point
    res = _pulse_and_capture(gaze_estimator, cap, pts, sw, sh)
    cap.release()
    cv2.destroyAllWindows()

    if res is None:
        return

    # Fit the regression model on all valid captured feature-target pairs
    feats, targs = res
    if feats:
        gaze_estimator.train(np.array(feats), np.array(targs))
