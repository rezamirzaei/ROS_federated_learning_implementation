#!/usr/bin/env python3
"""Compare two benchmark JSON reports side-by-side.

Usage
-----

    python scripts/compare.py results/benchmark_iid.json results/benchmark_skewed.json
    python scripts/compare.py --markdown results/a.json results/b.json  # emit table
    python scripts/compare.py --baseline results/baseline_ci.json results/ci_benchmark.json \\
        --fail-on-regression --max-accuracy-drop 5.0

The ``--fail-on-regression`` mode is wired into CI: it exits 1 when the
"new" report's final accuracy drops more than ``--max-accuracy-drop`` pp
below the baseline. Used to gate the smoke-benchmark run so silent model
regressions don't sneak in.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text())


def _extract_final_accuracy(report: dict[str, Any]) -> float | None:
    """Return final accuracy as a scalar, collapsing multi-seed reports."""
    summary = report.get("summary") or {}
    final = summary.get("final_test_accuracy")
    if isinstance(final, dict):  # multi-seed summary
        return float(final.get("mean"))
    if final is None:
        return None
    return float(final)


def _extract_bytes(report: dict[str, Any]) -> int | None:
    return report.get("summary", {}).get("total_bytes")


def _extract_wall(report: dict[str, Any]) -> float | None:
    summary = report.get("summary") or {}
    wall = summary.get("total_wall_seconds")
    if isinstance(wall, dict):
        return float(wall.get("mean"))
    return None if wall is None else float(wall)


def _fmt_bytes(n: int | None) -> str:
    if n is None:
        return "—"
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} TiB"


def render_markdown(a: dict, b: dict, *, name_a: str, name_b: str) -> str:
    acc_a = _extract_final_accuracy(a)
    acc_b = _extract_final_accuracy(b)
    bytes_a = _extract_bytes(a)
    bytes_b = _extract_bytes(b)
    wall_a = _extract_wall(a)
    wall_b = _extract_wall(b)
    alg_a = a.get("config", {}).get("algorithm", "?")
    alg_b = b.get("config", {}).get("algorithm", "?")

    lines = [
        f"| Metric | {name_a} | {name_b} | Δ |",
        "|---|---:|---:|---:|",
        f"| Algorithm | `{alg_a}` | `{alg_b}` | — |",
        f"| Final accuracy (%) | {acc_a} | {acc_b} | "
        f"{f'{acc_b - acc_a:+.3f}' if acc_a is not None and acc_b is not None else '—'} |",
        f"| Wall-clock (s) | {wall_a} | {wall_b} | "
        f"{f'{wall_b - wall_a:+.2f}' if wall_a is not None and wall_b is not None else '—'} |",
        f"| Total bytes | {_fmt_bytes(bytes_a)} | {_fmt_bytes(bytes_b)} | "
        f"{_fmt_bytes(bytes_b - bytes_a) if bytes_a is not None and bytes_b is not None else '—'} |",
    ]
    return "\n".join(lines)


def render_plain(a: dict, b: dict, *, name_a: str, name_b: str) -> str:
    acc_a = _extract_final_accuracy(a)
    acc_b = _extract_final_accuracy(b)
    return (
        f"{name_a}: final={acc_a}%\n{name_b}: final={acc_b}%\nΔ = {acc_b - acc_a:+.3f} pp"
        if acc_a is not None and acc_b is not None
        else f"{name_a}: {acc_a}\n{name_b}: {acc_b}"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("baseline", type=Path, help="Baseline (reference) JSON")
    p.add_argument("candidate", type=Path, help="Candidate (new) JSON")
    p.add_argument("--markdown", action="store_true", help="Emit a markdown table")
    p.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero if candidate regresses versus baseline.",
    )
    p.add_argument(
        "--max-accuracy-drop",
        type=float,
        default=5.0,
        help="Percentage-points drop tolerated before regression fails CI.",
    )
    args = p.parse_args(argv)

    a = _load(args.baseline)
    b = _load(args.candidate)
    name_a = args.baseline.stem
    name_b = args.candidate.stem

    print(
        render_markdown(a, b, name_a=name_a, name_b=name_b)
        if args.markdown
        else render_plain(a, b, name_a=name_a, name_b=name_b)
    )

    if args.fail_on_regression:
        acc_a = _extract_final_accuracy(a)
        acc_b = _extract_final_accuracy(b)
        if acc_a is None or acc_b is None:
            print("\nERROR: missing final_test_accuracy in one of the reports", file=sys.stderr)
            return 2
        drop = acc_a - acc_b
        if drop > args.max_accuracy_drop:
            print(
                f"\nREGRESSION: final accuracy dropped {drop:.3f} pp "
                f"(tolerance {args.max_accuracy_drop:.3f} pp)",
                file=sys.stderr,
            )
            return 1
        print(f"\nOK: accuracy delta {-drop:+.3f} pp (tolerance ±{args.max_accuracy_drop:.3f} pp)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
