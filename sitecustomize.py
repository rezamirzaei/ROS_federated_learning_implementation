"""Make local source-checkout imports resolve consistently.

Python imports ``sitecustomize`` automatically during startup when this
repository root is on ``sys.path``. Adding ``src/fl_robots`` here keeps direct
``python`` runs aligned with the editable install layout used by ``uv``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_src_dir = Path(__file__).resolve().parent / "src" / "fl_robots"
if _src_dir.is_dir():
    _src_dir_str = str(_src_dir)
    if _src_dir_str not in sys.path:
        sys.path.insert(0, _src_dir_str)
