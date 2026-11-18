"""Fast 5-point gaze calibration routine covering screen center and corners.

This module implements a lightweight calibration workflow using five anchor
points (center plus the four corners of a 3x3 grid). It checks for stable face
detection, presents sequential visual targets, records eye features, and trains
the GazeEstimator on the collected samples.
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


def run_5_point_calibration(
    gaze_estimator: GazeEstimator, camera_index: int = 0
) -> None:
    """Calibrate a GazeEstimator using a quick 5-point screen pattern.

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

    # Define center and four corner coordinates on a 3x3 logical grid
    order = [(1, 1), (0, 0), (2, 0), (0, 2), (2, 2)]
    pts = compute_grid_points(order, sw, sh)

    # Display targets and collect eye features at each point
    res = _pulse_and_capture(gaze_estimator, cap, pts, sw, sh)
    cap.release()
    cv2.destroyAllWindows()

    # Abort if user cancelled early via ESC
    if res is None:
        return

    # Train the regression model on all valid captured feature-target pairs
    feats, targs = res
    if feats:
        gaze_estimator.train(np.array(feats), np.array(targs))
