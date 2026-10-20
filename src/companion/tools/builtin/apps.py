"""App tools: list and open applications from the allowlist in data/apps.json."""

from __future__ import annotations

import os
from pathlib import Path

from companion.config import get_config
from companion.safety.allowlist import load_allowlist
from companion.tools.registry import tool


def _apps_file() -> Path:
    return get_config().tools.apps_file


def _launch(target: str) -> None:
    """Hand the target to the Windows shell. Steam URLs, programs, and shortcuts all work."""
    if not hasattr(os, "startfile"):
        raise RuntimeError("opening apps is only supported on Windows")
    os.startfile(target)  # type: ignore[attr-defined]


@tool
def list_apps() -> str:
    """List the applications and games the companion is allowed to open."""
    names = sorted(load_allowlist(_apps_file()))
    if not names:
        return "No apps are on the allowlist yet. Add entries to data/apps.json."
    return "I can open: " + ", ".join(names) + "."


@tool
def open_app(name: str) -> str:
    """Open an application or game by its friendly name. Only names from list_apps work.

    Args:
        name: The friendly name of the app, as listed by list_apps.
    """
    apps = load_allowlist(_apps_file())
    key = name.strip().lower()
    target = apps.get(key)
    if target is None:
        return f"Error: '{name}' is not on the app allowlist. Use list_apps to see what I can open."
    if os.path.isabs(target) and not Path(target).is_file():
        return f"Error: the file for '{key}' was not found at {target}."
    _launch(target)
    return f"Opened {key}."
