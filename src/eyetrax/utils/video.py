"""OpenCV webcam capture and fullscreen display window utilities for EyeTrax.

This module provides helper functions and context managers for opening webcam
devices (with automatic fallback from index 0 to 1), managing fullscreen OpenCV
windows safely, and iterating continuously over incoming video frames.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import cv2
import numpy as np


def open_camera(index: int = 0) -> cv2.VideoCapture:
    """Open an OpenCV VideoCapture device by index with fallback support.

    If opening device `0` fails (common when virtual or secondary cameras shift
    indices), this function automatically attempts to open device `1`.

    Args:
        index: Camera device index to open (default 0).

    Returns:
        An opened `cv2.VideoCapture` instance.

    Raises:
        RuntimeError: If the requested camera (and fallback, when applicable) cannot be opened.
    """
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        return cap

    cap.release()
    # Fall back to device 1 when default primary webcam index 0 is unavailable
    if index == 0:
        cap = cv2.VideoCapture(1)
        if cap.isOpened():
            return cap
        cap.release()
        raise RuntimeError("cannot open camera 0 (fallback to camera 1 also failed)")

    raise RuntimeError(f"cannot open camera {index}")


@contextmanager
def fullscreen(name: str) -> Iterator[None]:
    """Context manager that creates and automatically destroys a fullscreen OpenCV window.

    Args:
        name: Window title identifier used by `cv2.imshow`.

    Yields:
        Control to the caller while the fullscreen window remains active.
    """
    cv2.namedWindow(name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    try:
        yield
    finally:
        cv2.destroyWindow(name)


@contextmanager
def camera(index: int = 0) -> Iterator[cv2.VideoCapture]:
    """Context manager yielding an opened `cv2.VideoCapture` and releasing it on exit.

    Args:
        index: Camera device index passed to `open_camera`.

    Yields:
        Active `cv2.VideoCapture` instance.
    """
    cap = open_camera(index)
    try:
        yield cap
    finally:
        cap.release()


def iter_frames(cap: cv2.VideoCapture) -> Iterator[np.ndarray]:
    """Infinite generator that continuously reads and yields valid BGR frames.

    Args:
        cap: Active `cv2.VideoCapture` instance.

    Yields:
        Successive non-empty BGR video frames as NumPy arrays.
    """
    while True:
        ok, frame = cap.read()
        # Skip dropped or empty frames without terminating the stream loop
        if not ok:
            continue
        yield frame
