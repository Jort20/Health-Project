"""2D OpenCV drawing utilities for rendering gaze cursors and webcam thumbnails.

This module provides helper functions for compositing visual elements onto
EyeTrax display canvases, including a semi-transparent concentric gaze cursor
and a bordered webcam preview thumbnail.
"""

from __future__ import annotations

import cv2
import numpy as np


def draw_cursor(
    canvas: np.ndarray,
    x: int,
    y: int,
    alpha: float,
    *,
    radius_outer: int = 30,
    radius_inner: int = 25,
    color_outer: tuple[int, int, int] = (0, 0, 255),
    color_inner: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Render a semi-transparent concentric gaze cursor onto the target canvas in-place.

    Args:
        canvas: BGR image array to draw the cursor onto.
        x: Horizontal cursor center coordinate in pixels.
        y: Vertical cursor center coordinate in pixels.
        alpha: Visibility factor in [0.0, 1.0]; skips drawing if less than or equal to 0.
        radius_outer: Radius in pixels of the outer ring.
        radius_inner: Radius in pixels of the inner filled circle (disabled if <= 0).
        color_outer: BGR color tuple for the outer circle.
        color_inner: BGR color tuple for the inner circle.

    Returns:
        The modified `canvas` array with the blended cursor overlay.
    """
    if alpha <= 0.0:
        return canvas

    # Draw concentric circles on a temporary copy for alpha blending
    overlay = canvas.copy()
    cv2.circle(overlay, (int(x), int(y)), radius_outer, color_outer, -1)
    if radius_inner > 0:
        cv2.circle(overlay, (int(x), int(y)), radius_inner, color_inner, -1)

    # Blend overlay onto the main canvas with scaled opacity
    cv2.addWeighted(overlay, alpha * 0.6, canvas, 1 - alpha * 0.6, 0, canvas)
    return canvas


def make_thumbnail(
    frame: np.ndarray,
    *,
    size: tuple[int, int] = (320, 240),
    border: int = 2,
    border_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Resize a webcam frame into a bordered picture-in-picture thumbnail.

    Args:
        frame: Raw BGR webcam frame.
        size: Target inner (width, height) in pixels before adding the border.
        border: Border thickness in pixels around all four edges.
        border_color: BGR color tuple for the solid border frame.

    Returns:
        Resized BGR image array with the surrounding border applied.
    """
    img = cv2.resize(frame, size)
    return cv2.copyMakeBorder(
        img,
        border,
        border,
        border,
        border,
        cv2.BORDER_CONSTANT,
        value=border_color,
    )
