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
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _installed_packages() -> dict[str, str]:
    try:
        import importlib.metadata as md

        return {d.metadata["Name"]: d.version for d in md.distributions()}
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
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"Wrote environment manifest to {args.output}")


if __name__ == "__main__":
    main()

