"""
Module for running an eyetracking session combined with audio monitoring.
The collected data (raw and smoothed gaze direction, blink detection, and sound levels) 
is continuously written to a CSV file for later analysis.
"""

import csv
from datetime import datetime
import os
import threading
import time

import cv2
import numpy as np
import sounddevice as sd

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


class AmbientAudioMonitor:
    """
    Class for monitoring ambient noise (in decibels) in the background.
    Reads the microphone asynchronously without blocking the main eyetracker thread.
    """
    def __init__(self, samplerate: int = 16000, db_offset: float = 90.0):
        self.samplerate = samplerate
        self.db_offset = db_offset
        self._lock = threading.Lock()
        self._rms_buffer = []
        self._peak_buffer = []
        self._stream = None

    def _callback(self, indata, frames, time_info, status):
        """Internal callback function called by sounddevice when new audio data arrives."""
        mono = indata[:, 0]
        rms = float(np.sqrt(np.mean(np.square(mono))))
        peak = float(np.max(np.abs(mono)))
        with self._lock:
            self._rms_buffer.append(rms)
            self._peak_buffer.append(peak)

    def start(self):
        """Starts the microphone stream and begins listening to the environment."""
        try:
            self._stream = sd.InputStream(
                channels=1,
                samplerate=self.samplerate,
                callback=self._callback,
            )
            self._stream.start()
        except Exception as e:
            print(f"[warning] Microphone could not be started: {e}")
            self._stream = None

    def read_interval(self):
        """
        Processes the collected audio data since the last time this function was called.

        Returns:
            tuple: (mean_sound_level_db, peak_sound_level_db). 
                   Returns ("", "") if no data is available.
        """
        if self._stream is None:
            return "", ""
        with self._lock:
            if not self._rms_buffer:
                return "", ""
            mean_rms = float(np.sqrt(np.mean(np.square(self._rms_buffer))))
            max_peak = float(np.max(self._peak_buffer))
            self._rms_buffer.clear()
            self._peak_buffer.clear()

        db_mean = round(20.0 * np.log10(max(mean_rms, 1e-9)) + self.db_offset, 1)
        db_peak = round(20.0 * np.log10(max(max_peak, 1e-9)) + self.db_offset, 1)
        return db_mean, db_peak

    def stop(self):
        """Stops the audio stream and closes the microphone."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()


def run_virtualcam():
    """
    Main function to initialize and run the application.
    
    Process:
    1. Parses command-line arguments.
    2. Loads or calibrates the GazeEstimator model.
    3. Sets up the chosen filter/smoother (e.g., Kalman) for stable coordinates.
    4. Starts audio monitoring and the webcam.
    5. Loop per frame: calculates gaze direction (x, y), audio DB, and writes everything to a CSV file.
    """
    args = parse_common_args()

    filter_method = args.filter
    camera_index = args.camera
    calibration_method = args.calibration
    confidence_level = args.confidence
    ema_alpha = args.ema_alpha

    gaze_estimator = GazeEstimator(model_name=args.model)

    # Initialize or calibrate the model
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

    # Set up the chosen smoothing filter
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

    audio_monitor = AmbientAudioMonitor(db_offset=90.0)
    audio_monitor.start()

    os.makedirs("Data", exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
    csv_path = os.path.join("Data", f"gaze_data_{timestamp_str}.csv")

    with camera(camera_index) as cap, open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "x_raw",
            "y_raw",
            "x_smooth",
            "y_smooth",
            "blink",
            "ear",
            "sound_db",
            "sound_peak_db",
        ])
        print(f"Tracking started -> {csv_path} (Ctrl+C to stop)")

        try:
            for frame in iter_frames(cap):
                features, blink_detected = gaze_estimator.extract_features(frame)
                sound_db, sound_peak_db = audio_monitor.read_interval()

                # Retrieve Eye Aspect Ratio (EAR)
                if features is not None and len(gaze_estimator._ear_history) > 0:
                    ear = round(float(gaze_estimator._ear_history[-1]), 4)
                else:
                    ear = ""

                # Calculate gaze location if eyes are open
                if features is not None and not blink_detected:
                    gaze_point = gaze_estimator.predict(np.array([features]))[0]
                    x, y = map(int, gaze_point)
                    x_pred, y_pred = smoother.step(x, y)
                else:
                    x = y = x_pred = y_pred = ""

                writer.writerow([
                    time.time(),
                    x,
                    y,
                    x_pred,
                    y_pred,
                    int(blink_detected),
                    ear,
                    sound_db,
                    sound_peak_db,
                ])
        except KeyboardInterrupt:
            print(f"\nStopped and saved to {csv_path}")
        finally:
            audio_monitor.stop()


if __name__ == "__main__":
    run_virtualcam()
