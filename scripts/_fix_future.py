#!/usr/bin/env python3
"""Fix from __future__ import annotations ordering - must be first."""

from pathlib import Path

files = (
    list(Path("tests").glob("*.py"))
    + list(Path("scripts").glob("*.py"))
    + list(Path("src/fl_robots/test").glob("*.py"))
)

for p in files:
    text = p.read_text()
    lines = text.split("\n")

    # Check if `from typing import Any` appears BEFORE `from __future__`
    any_idx = None
    future_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "from typing import Any" and any_idx is None:
            any_idx = i
        if line.strip() == "from __future__ import annotations" and future_idx is None:
            future_idx = i

    if any_idx is not None and future_idx is not None and any_idx < future_idx:
        # Remove the Any import and put it after __future__
        lines.pop(any_idx)
        # future_idx shifted down by 1 since we removed a line before it
        new_future_idx = future_idx - 1
        lines.insert(new_future_idx + 1, "from typing import Any")
        p.write_text("\n".join(lines))
        print(f"Fixed __future__ ordering in {p}")
    elif any_idx is not None and future_idx is None:
        # Has Any but no __future__ - add __future__ before it
        lines.insert(any_idx, "from __future__ import annotations")
        p.write_text("\n".join(lines))
        print(f"Added __future__ before Any in {p}")
