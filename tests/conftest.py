"""Pytest configuration — ensure fl_robots package is importable."""

import sys
from pathlib import Path

_src = str(Path(__file__).resolve().parent.parent / "src" / "fl_robots")
if _src not in sys.path:
    sys.path.insert(0, _src)

