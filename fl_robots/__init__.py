"""Source-checkout compatibility package for ``fl_robots`` imports."""

from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

_impl_package = Path(__file__).resolve().parent.parent / "src" / "fl_robots" / "fl_robots"
if _impl_package.is_dir():
    __path__.append(str(_impl_package))

__author__ = "Developer"
__version__ = "1.0.0"

__all__ = ["__author__", "__version__"]
