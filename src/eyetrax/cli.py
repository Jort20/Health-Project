"""Shared command-line argument parser for EyeTrax gaze estimation scripts.

This module defines `parse_common_args`, standardizing CLI flags across EyeTrax
applications for selecting webcam indices, calibration routines, smoothing filters,
regression model backends, pre-trained model paths, and display backgrounds.
"""

import argparse


def parse_common_args() -> argparse.Namespace:
    """Parse common command-line arguments for EyeTrax calibration and inference.

    Returns:
        argparse.Namespace: Parsed configuration options including filter type,
            camera index, calibration method, grid dimensions, regression model
            backend, and optional pre-trained model file path.
    """
    parser = argparse.ArgumentParser(description="Common Gaze Estimation Arguments")

    # Temporal/spatial gaze coordinate smoothing options
    parser.add_argument(
        "--filter",
        choices=["kalman", "kalman_ema", "kde", "none"],
        default="none",
        help="Select the filter to apply to gaze estimation",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.25,
        help="EMA smoothing strength for kalman_ema (0 disables EMA, closer to 1 is smoother)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence level for KDE smoothing, range 0 to 1",
    )

    # Hardware and calibration routine settings
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index for video capture, default is 0 (first camera)",
    )
    parser.add_argument(
        "--calibration",
        choices=["9p", "5p", "lissajous", "dense"],
        default="9p",
        help="Calibration method for gaze estimation",
    )

    # Dense grid calibration layout parameters
    parser.add_argument(
        "--grid-rows",
        type=int,
        default=5,
        help="Grid rows for dense calibration (default 5)",
    )
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=5,
        help="Grid columns for dense calibration (default 5)",
    )
    parser.add_argument(
        "--grid-margin",
        type=float,
        default=0.10,
        help="Margin ratio for dense calibration (default 0.10 = 10%)",
    )

    # Display and regression model options
    parser.add_argument(
        "--background",
        type=str,
        default=None,
        help="Path to a custom background image (optional)",
    )
    parser.add_argument(
        "--model",
        default="ridge",
        help="The machine learning model to use for gaze estimation, default is 'ridge'",
    )
    parser.add_argument(
        "--model-file",
        type=str,
        default=None,
        help="Path to a previously-trained gaze model",
    )

    return parser.parse_args()
