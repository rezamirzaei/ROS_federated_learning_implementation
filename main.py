#!/usr/bin/env python3
"""Repository entrypoint for local source runs.

This thin wrapper keeps ``python main.py`` working from the repository root while
routing all real CLI behavior through the packaged ``fl_robots.cli`` module.
"""

from __future__ import annotations

from fl_robots.cli import main


if __name__ == "__main__":
    main()
