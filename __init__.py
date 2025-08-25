# src/__init__.py
"""
intraday-volume-ml | Top-level package initializer.

Usage examples:
    from src import __version__, load_config, ensure_dirs
    from src.utils import set_seed
    from src.models import DeepLOBv
"""

from __future__ import annotations
from importlib.metadata import version as _pkg_version, PackageNotFoundError

__all__ = [
    "__version__",           # package version string
    "load_config", "ensure_dirs",
    "set_seed", "device_auto",
    # Optional top-level exposure of a commonly used model:
    "DeepLOBv",
]

# --- version ---
try:
    __version__ = _pkg_version("intraday-volume-ml")  # if installed as a package
except PackageNotFoundError:
    __version__ = "0.0.0"

# --- light-weight re-exports (keep imports cheap & side-effect-free) ---
from .config import load_config, ensure_dirs               # noqa: E402
from .utils.common import set_seed, device_auto            # noqa: E402

# --- optional: model at top level (guarded to avoid hard fail if torch missing) ---
try:
    from .models.deeplobv import DeepLOBv                 # noqa: E402
except Exception:  # e.g., torch not installed yet
    DeepLOBv = None  # type: ignore
