"""Compatibility shim for editable installs.

The actual Python package lives in ``src/fl_robots/fl_robots`` because the ROS2
workspace also stores ``setup.py``, ``package.xml``, ``launch/``, and ``config/``
under ``src/fl_robots``. Some editable-install workflows can resolve this outer
directory as the ``fl_robots`` package root, which breaks imports such as
``fl_robots.message_bus``.

This shim extends the package path to include the inner implementation package so
both local source runs and editable installs resolve the same modules.
"""

from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

_inner_package = Path(__file__).resolve().parent / "fl_robots"
if _inner_package.is_dir():
	__path__.append(str(_inner_package))

try:
	from .fl_robots import __author__, __version__
except ImportError:  # pragma: no cover - only relevant during partial installs
	__author__ = "Developer"
	__version__ = "0.0.0"

__all__ = ["__author__", "__version__"]


