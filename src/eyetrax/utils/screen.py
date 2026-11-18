"""Display resolution utilities for EyeTrax calibration and overlay rendering.

This module queries connected display monitors via `screeninfo` to determine
the primary screen dimensions in pixels, enabling accurate mapping between
normalized gaze estimates and full-screen pixel coordinates.
"""

from typing import Tuple

from screeninfo import get_monitors


def get_screen_size() -> Tuple[int, int]:
    """Retrieve the width and height in pixels of the primary display monitor.

    Returns:
        Tuple of (width, height) in pixels for the first detected monitor.
    """
    # Query the primary monitor's pixel resolution
    m = get_monitors()[0]
    return m.width, m.height
