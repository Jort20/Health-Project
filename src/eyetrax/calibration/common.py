"""Shared calibration utilities for grid generation, face readiness, and data capture.

This module provides helper functions used across EyeTrax calibration routines:
computing pixel coordinates for screen calibration grids, waiting until a user's
face and open eyes are stably detected, and executing the visual pulse-and-capture
loop to record eye features at target screen positions.
"""

import time
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

from eyetrax.gaze import GazeEstimator


def compute_grid_points(
    order: Iterable[Tuple[int, int]],
    sw: int,
    sh: int,
    margin_ratio: float = 0.10,
) -> List[Tuple[int, int]]:
    """Translate grid (row, col) indices into absolute screen pixel locations.

    Args:
        order: Sequence of (row, col) grid coordinates.
        sw: Screen width in pixels.
        sh: Screen height in pixels.
        margin_ratio: Fractional edge padding applied to all screen borders.

    Returns:
        List of (x, y) screen pixel coordinates corresponding to the input order.
    """
    order = list(order)
    if not order:
        return []

    max_r = max(r for r, _ in order)
    max_c = max(c for _, c in order)

    # Calculate inset pixel margins and usable grid dimensions
    mx, my = int(sw * margin_ratio), int(sh * margin_ratio)
    gw, gh = sw - 2 * mx, sh - 2 * my

    step_x = 0 if max_c == 0 else gw / max_c
    step_y = 0 if max_r == 0 else gh / max_r

    return [(mx + int(c * step_x), my + int(r * step_y)) for r, c in order]


def compute_grid_points_from_shape(
    rows: int,
    cols: int,
    sw: int,
    sh: int,
    margin_ratio: float = 0.10,
    order: str = "default",
) -> list[tuple[int, int]]:
    """Generate (x, y) pixel coordinates for a rows x cols calibration grid.

    Args:
        rows: Number of grid rows (must be positive).
        cols: Number of grid columns (must be positive).
        sw: Screen width in pixels.
        sh: Screen height in pixels.
        margin_ratio: Fractional edge padding in the range [0.0, 0.5).
        order: Traversal pattern — "default" (row-major) or "serpentine" (snake
            pattern to minimize long saccadic jumps between rows).

    Returns:
        Ordered list of (x, y) screen coordinates.

    Raises:
        ValueError: If grid dimensions, margin_ratio, or traversal order are invalid.
    """
    if rows <= 0 or cols <= 0:
        raise ValueError(f"rows and cols must be positive (got rows={rows}, cols={cols})")
    if not 0.0 <= margin_ratio < 0.5:
        raise ValueError(f"margin_ratio must be in [0.0, 0.5) (got {margin_ratio})")

    if order == "default":
        indices = [(r, c) for r in range(rows) for c in range(cols)]
    elif order == "serpentine":
        # Reverse column direction on odd rows for smooth visual tracking
        indices = []
        for r in range(rows):
            cols_range = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
            indices.extend((r, c) for c in cols_range)
    else:
        raise ValueError(f"unknown order '{order}' (expected 'default' or 'serpentine')")

    return compute_grid_points(indices, sw, sh, margin_ratio)


def wait_for_face_and_countdown(
    cap: cv2.VideoCapture,
    gaze_estimator: GazeEstimator,
    sw: int,
    sh: int,
    dur: int = 2,
) -> bool:
    """Wait until a non-blinking face is detected, then display a countdown timer.

    Args:
        cap: Active OpenCV webcam capture object.
        gaze_estimator: Estimator used to verify face and open-eye feature extraction.
        sw: Screen width in pixels.
        sh: Screen height in pixels.
        dur: Required continuous face-detection duration in seconds before starting.

    Returns:
        True once the countdown completes, or False if aborted via ESC.
    """
    cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    fd_start = None
    countdown = False

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        f, blink = gaze_estimator.extract_features(frame)
        face = f is not None and not blink
        canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
        now = time.time()

        if face:
            if not countdown:
                fd_start = now
                countdown = True
            elapsed = now - fd_start
            if elapsed >= dur:
                return True

            # Render shrinking green pie countdown with smoothstep easing
            t = elapsed / dur
            e = t * t * (3 - 2 * t)
            ang = 360 * (1 - e)
            cv2.ellipse(
                canvas,
                (sw // 2, sh // 2),
                (50, 50),
                0,
                -90,
                -90 + ang,
                (0, 255, 0),
                -1,
            )
        else:
            # Reset timer and warn user if face tracking is lost or eyes are closed
            countdown = False
            fd_start = None
            txt = "Face not detected"
            fs = 2
            thick = 3
            size, _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)
            tx = (sw - size[0]) // 2
            ty = (sh + size[1]) // 2
            cv2.putText(
                canvas, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 255), thick
            )

        cv2.imshow("Calibration", canvas)
        if cv2.waitKey(1) == 27:
            return False


def _pulse_and_capture(
    gaze_estimator: GazeEstimator,
    cap: cv2.VideoCapture,
    pts: Iterable[Tuple[int, int]],
    sw: int,
    sh: int,
    pulse_d: float = 1.0,
    cd_d: float = 1.0,
) -> Optional[Tuple[List[np.ndarray], List[List[int]]]]:
    """Run the sequential pulse-and-capture animation for a list of target points.

    Args:
        gaze_estimator: Estimator used to extract eye features from webcam frames.
        cap: Active OpenCV webcam capture object.
        pts: Iterable of (x, y) pixel targets to display.
        sw: Screen width in pixels.
        sh: Screen height in pixels.
        pulse_d: Duration in seconds of the attention-grabbing pulse phase.
        cd_d: Duration in seconds of the feature-recording countdown phase.

    Returns:
        Tuple of (extracted_features, target_xy_coords), or None if aborted via ESC.
    """
    feats, targs = [], []

    for x, y in pts:
        # Pulsing green target dot to attract gaze fixation
        ps = time.time()
        final_radius = 20
        while True:
            e = time.time() - ps
            if e > pulse_d:
                break
            ok, frame = cap.read()
            if not ok:
                continue
            canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
            radius = 15 + int(15 * abs(np.sin(2 * np.pi * e)))
            final_radius = radius
            cv2.circle(canvas, (x, y), radius, (0, 255, 0), -1)
            cv2.imshow("Calibration", canvas)
            if cv2.waitKey(1) == 27:
                return None

        # Static dot with shrinking progress ring while recording features
        cs = time.time()
        while True:
            e = time.time() - cs
            if e > cd_d:
                break
            ok, frame = cap.read()
            if not ok:
                continue
            canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
            cv2.circle(canvas, (x, y), final_radius, (0, 255, 0), -1)
            t = e / cd_d
            ease = t * t * (3 - 2 * t)
            ang = 360 * (1 - ease)
            cv2.ellipse(canvas, (x, y), (40, 40), 0, -90, -90 + ang, (255, 255, 255), 4)
            cv2.imshow("Calibration", canvas)
            if cv2.waitKey(1) == 27:
                return None

            # Store valid non-blink feature vectors paired with screen coordinates
            ft, blink = gaze_estimator.extract_features(frame)
            if ft is not None and not blink:
                feats.append(ft)
                targs.append([x, y])

    return feats, targs
