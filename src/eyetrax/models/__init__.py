"""Model registry and factory utilities for EyeTrax gaze regression backends.

This module maintains a registry (`AVAILABLE_MODELS`) of available gaze
estimation model classes, automatically discovers model implementations within
the `eyetrax.models` package, and exposes `create_model` to instantiate them
by name.
"""

from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Type

from .base import BaseModel

__all__ = ["BaseModel", "create_model", "AVAILABLE_MODELS"]

AVAILABLE_MODELS: Dict[str, Type[BaseModel]] = {}


def register_model(name: str, cls: Type[BaseModel]) -> None:
    """Register a gaze model class under a unique identifier name.

    Args:
        name: Lookup key used to reference the model (e.g., "ridge", "mlp").
        cls: Subclass of `BaseModel` to associate with `name`.

    Raises:
        ValueError: If `name` is already registered in `AVAILABLE_MODELS`.
    """
    if name in AVAILABLE_MODELS:
        raise ValueError(f"Model name '{name}' already registered")
    AVAILABLE_MODELS[name] = cls


def _auto_discover() -> None:
    """Dynamically import all sibling model modules to trigger their registration."""
    pkg_dir = Path(__file__).resolve().parent
    for f in pkg_dir.iterdir():
        # Skip package init, abstract base, and non-Python files
        if f.name in {"__init__.py", "base.py"} or f.suffix != ".py":
            continue
        mod_name = f"{__name__}.{f.stem}"
        import_module(mod_name)


def create_model(name: str, **kwargs: Any) -> BaseModel:
    """Instantiate a registered gaze estimation model by name.

    Args:
        name: Registered model identifier.
        **kwargs: Keyword arguments forwarded to the model constructor.

    Returns:
        An initialized instance of the requested `BaseModel` subclass.

    Raises:
        ValueError: If `name` does not match any discovered model.
    """
    # Populate registry on first lookup
    if not AVAILABLE_MODELS:
        _auto_discover()
    try:
        cls = AVAILABLE_MODELS[name]
    except KeyError as e:
        raise ValueError(
            f"Unknown model '{name}'. Available: {sorted(AVAILABLE_MODELS)}"
        ) from e
    return cls(**kwargs)
