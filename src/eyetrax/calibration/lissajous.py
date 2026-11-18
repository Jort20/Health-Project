"""Continuous smooth-pursuit gaze calibration along a Lissajous curve.

This module animates a single moving target along a 3:2 Lissajous figure-eight
trajectory with sinusoidal velocity easing. It continuously extracts eye features
while the user tracks the moving dot and trains the GazeEstimator on the
collected samples.
"""

from typing import Tuple

import cv2
import numpy as np

from eyetrax.calibration.common import wait_for_face_and_countdown
from eyetrax.gaze import GazeEstimator
from eyetrax.utils.screen import get_screen_size
from eyetrax.utils.video import open_camera


def run_lissajous_calibration(
    gaze_estimator: GazeEstimator, camera_index: int = 0
) -> None:
    """Calibrate a GazeEstimator by tracking a target moving along a Lissajous curve.

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

    # Define Lissajous amplitudes (40% of screen) and frequency ratio (a=3, b=2)
    A, B, a, b, d = sw * 0.4, sh * 0.4, 3, 2, 0

    def curve(t: float) -> Tuple[float, float]:
        """Compute (x, y) screen coordinates on the Lissajous trajectory at phase t."""
        return (A * np.sin(a * t + d) + sw / 2, B * np.sin(b * t) + sh / 2)

    total_time = 5.0
    fps = 60
    frames = int(total_time * fps)
    feats, targs = [], []

    # Pre-integrate sinusoidal speed profile to normalize phase progression
    acc = 0.0
    for i in range(frames):
        frac = i / (frames - 1)
        spd = 0.3 + 0.7 * np.sin(np.pi * frac)
        acc += spd / fps
    end = acc if acc >= 1e-6 else 1e-6
    acc = 0.0

    # Animate target along the curve and record eye features per frame
    for i in range(frames):
        frac = i / (frames - 1)
        spd = 0.3 + 0.7 * np.sin(np.pi * frac)
        acc += spd / fps
        t = (acc / end) * (2 * np.pi)

        ret, f = cap.read()
        if not ret:
            continue

        x, y = curve(t)
        c = np.zeros((sh, sw, 3), dtype=np.uint8)
        cv2.circle(c, (int(x), int(y)), 20, (0, 255, 0), -1)
        cv2.imshow("Calibration", c)

        # Exit if ESC key is pressed
        if cv2.waitKey(1) == 27:
            break

        # Store valid non-blink features paired with continuous target coordinates
        ft, blink = gaze_estimator.extract_features(f)
        if ft is not None and not blink:
            feats.append(ft)
            targs.append([x, y])

    cap.release()
    cv2.destroyAllWindows()

    # Fit the regression model on all collected smooth-pursuit samples
    if feats:
        gaze_estimator.train(np.array(feats), np.array(targs))
