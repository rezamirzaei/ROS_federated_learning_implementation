#!/usr/bin/env python3
"""Add type annotations to all functions missing them for mypy compliance."""

import re
from pathlib import Path


def fix_file(path: Path) -> int:
    text = path.read_text()
    lines = text.split("\n")
    changed = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        # Detect def line
        if re.match(r"^(\s*)def \w+\(", line):
            # Collect full signature
            sig_start = i
            paren_depth = line.count("(") - line.count(")")
            while paren_depth > 0 and i + 1 < len(lines):
                i += 1
                paren_depth += lines[i].count("(") - lines[i].count(")")
            sig_end = i
            full_sig = "\n".join(lines[sig_start : sig_end + 1])

            if "->" not in full_sig:
                # Need to add return type. Find the closing ): line
                last_line = lines[sig_end]
                new_last = re.sub(r"\)\s*:\s*$", ") -> None:", last_line)
                if new_last == last_line:
                    # Maybe ): is followed by something, or pattern didn't match
                    # Try more aggressive
                    new_last = re.sub(r"\)\s*:", ") -> None:", last_line, count=1)
                if new_last != last_line:
                    lines[sig_end] = new_last
                    changed += 1

            # Now check for untyped parameters in the (possibly updated) signature
            # We only handle simple cases: (self), (self, x), fixtures, etc.
        i += 1

    if changed:
        path.write_text("\n".join(lines))
    return changed


def fix_untyped_params(path: Path) -> int:
    """Fix 'Function is missing a type annotation for one or more parameters' by adding Any."""
    text = path.read_text()

    # Add 'from typing import Any' if not present and we have untyped params
    needs_any = False
    lines = text.split("\n")

    # Find functions with untyped non-self/cls params
    i = 0
    changed = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^(\s*)def (\w+)\(", line)
        if m:
            # Collect full sig
            sig_start = i
            paren_depth = line.count("(") - line.count(")")
            while paren_depth > 0 and i + 1 < len(lines):
                i += 1
                paren_depth += lines[i].count("(") - lines[i].count(")")
            sig_end = i

            full_sig = "\n".join(lines[sig_start : sig_end + 1])
            # Extract params between first ( and last )
            paren_content_match = re.search(r"\((.*)\)", full_sig, re.DOTALL)
            if paren_content_match:
                params_str = paren_content_match.group(1)
                params = [p.strip() for p in params_str.split(",")]
                new_params = []
                param_changed = False
                for p in params:
                    stripped = p.strip()
                    if not stripped or stripped in ("self", "cls"):
                        new_params.append(p)
                    elif ":" not in stripped and "=" not in stripped and "*" not in stripped:
                        # bare param like `client` -> `client: Any`
                        new_params.append(p + ": Any")
                        param_changed = True
                        needs_any = True
                    elif "=" in stripped and ":" not in stripped.split("=")[0]:
                        # param like `x=5` -> `x: Any = 5`
                        name_part, default = stripped.split("=", 1)
                        new_params.append(name_part.rstrip() + ": Any = " + default.lstrip())
                        param_changed = True
                        needs_any = True
                    else:
                        new_params.append(p)

                if param_changed:
                    new_params_str = ", ".join(new_params)
                    # Reconstruct - for single-line defs
                    if sig_start == sig_end:
                        new_line = re.sub(
                            r"\(.*\)", "(" + new_params_str + ")", lines[sig_start], count=1
                        )
                        lines[sig_start] = new_line
                        changed += 1
                    # For multi-line, just skip (too complex to restructure)
        i += 1

    if needs_any and changed:
        # Add 'from __future__ import annotations' and Any import if needed
        has_any_import = "from typing import Any" in text or (
            "from typing import" in text and "Any" in text
        )
        if not has_any_import:
            # Find a good place to insert
            insert_idx = 0
            for idx, line in enumerate(lines):
                if line.startswith(("import ", "from ")):
                    insert_idx = idx
                    break
            if insert_idx == 0:
                for idx, line in enumerate(lines):
                    if (
                        line.strip()
                        and not line.startswith("#")
                        and not line.startswith('"""')
                        and not line.startswith("'''")
                    ):
                        insert_idx = idx
                        break
            lines.insert(insert_idx, "from typing import Any")
            changed += 1

    if changed:
        path.write_text("\n".join(lines))
    return changed


files = [
    "scripts/visualize.py",
    "tests/test_ros_dashboard_security.py",
    "tests/test_showcase_web.py",
    "tests/test_security_headers.py",
    "tests/test_e2e_command_flow.py",
    "tests/test_coverage_boost.py",
    "tests/test_ros_integration.py",
    "tests/test_new_modules.py",
    "tests/test_localization.py",
    "tests/conftest.py",
    "src/fl_robots/test/test_aggregator_launch.py",
    "tests/test_properties.py",
    "tests/test_aggregation.py",
    "tests/test_mpc_observability.py",
    "tests/test_web_and_qp.py",
    "tests/test_ros_fake_env.py",
    "tests/test_production_endpoints.py",
    "tests/test_history_endpoints.py",
    "tests/test_fedprox_runtime.py",
    "tests/test_fedprox_benchmark.py",
    "tests/test_smoke_modules.py",
]

total = 0
for f in files:
    p = Path(f)
    if p.exists():
        c = fix_file(p)
        total += c
        if c:
            print(f"  return types: {f} ({c} fixes)")

# Second pass for param annotations
for f in files:
    p = Path(f)
    if p.exists():
        c = fix_untyped_params(p)
        total += c
        if c:
            print(f"  param types:  {f} ({c} fixes)")

print(f"\nTotal fixes: {total}")
