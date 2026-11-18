"""Top-level package entry point for EyeTrax with lazy symbol loading.

This module exposes the package version (`__version__`) and provides,
on-demand imports for core EyeTrax classes and calibration functions via PEP 562
(`__getattr__`), avoiding heavy dependency initialization (such as MediaPipe
and OpenCV) until those symbols are actually accessed.
"""

from typing import Any, List

from ._version import __version__

# Map public top-level symbols to their (submodule_path, attribute_name) targets
_lazy_map = {
    "GazeEstimator": ("eyetrax.gaze", "GazeEstimator"),
    "make_kalman": ("eyetrax.filters", "make_kalman"),
    "run_9_point_calibration": ("eyetrax.calibration", "run_9_point_calibration"),
    "run_5_point_calibration": ("eyetrax.calibration", "run_5_point_calibration"),
    "run_lissajous_calibration": ("eyetrax.calibration", "run_lissajous_calibration"),
}


def __getattr__(name: str) -> Any:
    """Lazily import and cache top-level EyeTrax symbols upon first attribute access.

    Args:
        name: Requested attribute name on the `eyetrax` package.

    Returns:
        The resolved class or function from its respective submodule.

    Raises:
        AttributeError: If `name` is not part of the public lazy export map.
    """
    try:
        module_name, symbol = _lazy_map[name]
    except KeyError:
        raise AttributeError(name) from None

    import importlib

    # Import submodule dynamically and cache symbol in module globals for future lookups
    module = importlib.import_module(module_name)
    value = getattr(module, symbol)
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    """Return sorted public attributes including lazily-loaded symbols for introspection."""
    std_attrs = set(globals()) | {"__getattr__", "__dir__"}
    return sorted(std_attrs | _lazy_map.keys())


__all__ = list(_lazy_map) + ["__version__"]
