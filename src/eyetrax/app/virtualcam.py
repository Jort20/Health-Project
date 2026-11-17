import csv
import os
import time

import cv2
import numpy as np

from eyetrax.calibration import (
    run_5_point_calibration,
    run_9_point_calibration,
    run_dense_grid_calibration,
    run_lissajous_calibration,
)
from eyetrax.cli import parse_common_args
from eyetrax.filters import (
    KDESmoother,
    KalmanEMASmoother,
    KalmanSmoother,
    NoSmoother,
    make_kalman,
)
from eyetrax.gaze import GazeEstimator
from eyetrax.utils.draw import draw_cursor
from eyetrax.utils.screen import get_screen_size
from eyetrax.utils.video import camera, iter_frames


def run_virtualcam():
    args = parse_common_args()

    filter_method = args.filter
    camera_index = args.camera
    calibration_method = args.calibration
    confidence_level = args.confidence
    ema_alpha = args.ema_alpha

    gaze_estimator = GazeEstimator(model_name=args.model)

    if args.model_file and os.path.isfile(args.model_file):
        gaze_estimator.load_model(args.model_file)
        print(f"[virtualcam] Loaded gaze model from {args.model_file}")
    else:
        if calibration_method == "9p":
            run_9_point_calibration(gaze_estimator, camera_index=camera_index)
        elif calibration_method == "5p":
            run_5_point_calibration(gaze_estimator, camera_index=camera_index)
        elif calibration_method == "dense":
            run_dense_grid_calibration(
                gaze_estimator,
                camera_index=camera_index,
                rows=args.grid_rows,
                cols=args.grid_cols,
                margin_ratio=args.grid_margin,
            )
        else:
            run_lissajous_calibration(gaze_estimator, camera_index=camera_index)

    screen_width, screen_height = get_screen_size()

    if filter_method == "kalman":
        kalman = make_kalman()
        smoother = KalmanSmoother(kalman)
        smoother.tune(gaze_estimator, camera_index=camera_index)
    elif filter_method == "kalman_ema":
        kalman = make_kalman()
        smoother = KalmanEMASmoother(kalman, ema_alpha=ema_alpha)
        smoother.tune(gaze_estimator, camera_index=camera_index)
    elif filter_method == "kde":
        kalman = None
        smoother = KDESmoother(screen_width, screen_height, confidence=confidence_level)
    else:
        kalman = None
        smoother = NoSmoother()

    with camera(camera_index) as cap, open("gaze_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "x_raw", "y_raw", "x_smooth", "y_smooth", "blink", "ear"])
        print("Tracking gestart -> gaze_data.csv (Ctrl+C om te stoppen)")

        try:
            for frame in iter_frames(cap):
                features, blink_detected = gaze_estimator.extract_features(frame)

                if features is not None and len(gaze_estimator._ear_history) > 0:
                    ear = round(float(gaze_estimator._ear_history[-1]), 4)
                else:
                    ear = ""

                if features is not None and not blink_detected:
                    gaze_point = gaze_estimator.predict(np.array([features]))[0]
                    x, y = map(int, gaze_point)
                    x_pred, y_pred = smoother.step(x, y)
                else:
                    x = y = x_pred = y_pred = ""

                writer.writerow([time.time(), x, y, x_pred, y_pred, int(blink_detected), ear])
        except KeyboardInterrupt:
            print("\nGestopt en opgeslagen in gaze_data.csv")


if __name__ == "__main__":
    run_virtualcam()
