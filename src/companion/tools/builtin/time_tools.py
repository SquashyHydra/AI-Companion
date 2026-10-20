"""Time tools."""

from __future__ import annotations

from datetime import datetime

from companion.tools.registry import tool


@tool
def get_time() -> str:
    """Return the current local date and time."""
    now = datetime.now().astimezone()
    return now.strftime("%A, %Y-%m-%d %H:%M:%S") + f" (UTC{now:%z})"
