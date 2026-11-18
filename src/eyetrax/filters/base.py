"""Abstract base interface for EyeTrax gaze-coordinate smoothing filters.

This module defines `BaseSmoother`, the contract that all temporal and spatial
gaze smoothing filters must implement to filter raw predicted screen coordinates
frame by frame.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple


class BaseSmoother(ABC):
    """Abstract base class for sequential 2D gaze-coordinate smoothers."""

    def __init__(self) -> None:
        """Initialize the smoother with an empty dictionary for optional debug visuals."""
        # Stores optional visualization metadata (e.g., KDE confidence contours)
        self.debug: dict = {}

    @abstractmethod
    def step(self, x: int, y: int) -> Tuple[int, int]:
        """Filter a single raw (x, y) gaze prediction and return smoothed coordinates.

        Args:
            x: Raw predicted horizontal screen coordinate in pixels.
            y: Raw predicted vertical screen coordinate in pixels.

        Returns:
            Tuple of smoothed (x, y) screen pixel coordinates.
        """
        ...
