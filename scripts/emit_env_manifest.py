#!/usr/bin/env python3
"""Emit an environment manifest as a JSON sidecar for benchmark reproducibility.

Usage::

    python scripts/emit_env_manifest.py -o results/env_manifest.json

The manifest captures Python version, installed packages, OS, git SHA, and
basic hardware info so any benchmark result can be traced back to the exact
environment that produced it.
"""

from __future__ import annotations

import argparse
import importlib.metadata as _md
import json
import logging
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


def _git_sha() -> str:
    """Read the current git SHA from .git/HEAD without subprocess."""
    try:
        head = Path(".git/HEAD").read_text().strip()
        if head.startswith("ref:"):
            ref_path = Path(".git") / head.split("ref: ", 1)[1]
            return ref_path.read_text().strip()
        return head
    except Exception:
        return "unknown"


def _installed_packages() -> dict[str, str]:
    try:
        return {d.metadata["Name"]: d.version for d in _md.distributions()}
    except Exception:
        return {}


def build_manifest() -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "git_sha": _git_sha(),
        "packages": _installed_packages(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit environment manifest")
    parser.add_argument("-o", "--output", default="results/env_manifest.json")
    args = parser.parse_args()

    manifest = build_manifest()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, default=str))
    logging.getLogger(__name__).info("Wrote environment manifest to %s", args.output)


if __name__ == "__main__":
    main()
