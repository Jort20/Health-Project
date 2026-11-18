"""Kernel Density Estimation (KDE) spatial-temporal gaze smoother and contour generator.

This module implements `KDESmoother`, which maintains a sliding time window of
recent gaze points, estimates a 2D Gaussian KDE over the screen, extracts
highest-density confidence contours for visual overlays, and returns the
windowed mean gaze position.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Tuple

import cv2
import numpy as np
from scipy.stats import gaussian_kde

from .base import BaseSmoother


class KDESmoother(BaseSmoother):
    """Smooths 2D gaze coordinates using a sliding-window Gaussian KDE."""

    def __init__(
        self,
        screen_w: int,
        screen_h: int,
        *,
        time_window: float = 0.5,
        confidence: float = 0.5,
        grid: Tuple[int, int] = (320, 200),
    ) -> None:
        """Initialize the KDE smoother and history buffer.

        Args:
            screen_w: Screen width in pixels.
            screen_h: Screen height in pixels.
            time_window: Sliding retention window in seconds for recent gaze points.
            confidence: Cumulative probability mass threshold in (0, 1) used to
                extract highest-density contour regions.
            grid: Low-resolution (width, height) evaluation grid for fast KDE density
                computation before upscaling to full screen size.
        """
        super().__init__()
        self.sw, self.sh = screen_w, screen_h
        self.window = time_window
        self.conf = confidence
        self.grid = grid
        self.hist: Deque[Tuple[float, int, int]] = deque()

    def step(self, x: int, y: int) -> Tuple[int, int]:
        """Record a new (x, y) gaze point, compute KDE confidence contours, and return smoothed coordinates.

        Args:
            x: Raw predicted horizontal screen coordinate in pixels.
            y: Raw predicted vertical screen coordinate in pixels.

        Returns:
            Tuple of smoothed (x, y) screen pixel coordinates (windowed mean).
        """
        now = time.time()

        # Append current sample and evict points older than the sliding time window
        self.hist.append((now, x, y))
        while self.hist and now - self.hist[0][0] > self.window:
            self.hist.popleft()

        pts = np.asarray([(hx, hy) for (_, hx, hy) in self.hist])
        if pts.shape[0] < 2:
            self.debug.clear()
            return x, y

        try:
            # Evaluate 2D Gaussian KDE across the downsampled screen grid
            kde = gaussian_kde(pts.T)
            xi, yi = np.mgrid[
                0 : self.sw : complex(self.grid[0]),
                0 : self.sh : complex(self.grid[1]),
            ]
            zi = kde(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape).T

            # Find density threshold capturing the desired cumulative confidence mass
            flat = zi.ravel()
            idx = np.argsort(flat)[::-1]
            cdf = np.cumsum(flat[idx]) / flat.sum()
            thr = flat[idx[np.searchsorted(cdf, self.conf)]]

            # Upscale binary confidence mask to screen resolution and extract contours
            mask = (zi >= thr).astype(np.uint8)
            mask = cv2.resize(mask, (self.sw, self.sh))

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Expose mask and contours for downstream visualization (e.g. demo overlay)
            self.debug["mask"] = mask
            self.debug["contours"] = contours

            sx, sy = pts.mean(axis=0).astype(int)
            return int(sx), int(sy)

        except np.linalg.LinAlgError:
            # Fallback when points are collinear and covariance matrix is singular
            self.debug.clear()
            return x, y
