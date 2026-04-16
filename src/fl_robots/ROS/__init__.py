"""Compatibility package that mirrors ``fl_robots`` under the ``ROS`` name."""

from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

_fl_package = Path(__file__).resolve().parent.parent / "fl_robots"
if _fl_package.is_dir():
    __path__.append(str(_fl_package))

from fl_robots import __author__, __version__

__all__ = ["__author__", "__version__"]
