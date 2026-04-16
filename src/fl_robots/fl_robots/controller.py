"""
Shared command controller.

Both dashboards (``standalone_web`` and the ROS2 ``web_dashboard``) previously
defined their own ``VALID_COMMANDS`` set and validated payloads inline. Keeping
them in one place means a new command (or a typo fix) only needs to land once.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

__all__ = [
    "COMMAND_NAMES",
    "CommandLiteral",
    "CommandRequest",
    "is_valid_command",
    "validate_command",
]

#: Canonical command vocabulary. Keep alphabetised so diffs stay minimal.
COMMAND_NAMES: tuple[str, ...] = (
    "disturbance",
    "reset",
    "start_training",
    "step",
    "stop_training",
    "toggle_autopilot",
)

CommandLiteral = Literal[
    "disturbance",
    "reset",
    "start_training",
    "step",
    "stop_training",
    "toggle_autopilot",
]


class CommandRequest(BaseModel):
    """Pydantic payload for ``POST /api/command``.

    Rejects unknown commands at parse time, so callers get a 400 with a clear
    Pydantic error message instead of silently forwarding garbage to the
    simulation engine.
    """

    model_config = ConfigDict(frozen=True, slots=True, extra="forbid")

    command: CommandLiteral


def is_valid_command(name: str) -> bool:
    """Pure predicate — useful inside the simulation engine's own dispatch."""
    return name in COMMAND_NAMES


def validate_command(name: str) -> CommandLiteral:
    """Validate *name* and return it typed, or raise ``ValueError``."""
    if not is_valid_command(name):
        raise ValueError(f"Unknown command {name!r}. Valid: {', '.join(COMMAND_NAMES)}")
    return name  # type: ignore[return-value]
