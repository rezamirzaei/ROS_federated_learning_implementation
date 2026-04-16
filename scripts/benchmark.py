#!/usr/bin/env python3
"""Repository-level shim so `python scripts/benchmark.py …` works from the root.

All real logic lives in :mod:`fl_robots.scripts.benchmark`.
"""

from __future__ import annotations

from fl_robots.scripts.benchmark import main

if __name__ == "__main__":
    raise SystemExit(main())
