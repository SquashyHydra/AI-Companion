"""Allowlist of applications the companion may open, loaded from a JSON file."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_STEAM_RE = re.compile(r"^steam://rungameid/\d+$")
_BARE_PROGRAM_RE = re.compile(r"^[A-Za-z0-9_.-]+\.(exe|lnk)$", re.IGNORECASE)
_ALLOWED_SUFFIXES = (".exe", ".lnk")


def is_valid_target(target: str) -> bool:
    """Accept a Steam game URL, a bare program name, or an absolute .exe or .lnk path."""
    if _STEAM_RE.match(target):
        return True
    if _BARE_PROGRAM_RE.match(target):
        return True
    path = Path(target)
    return path.is_absolute() and path.suffix.lower() in _ALLOWED_SUFFIXES


def load_allowlist(path: Path) -> dict[str, str]:
    """Return {friendly name: target} from the JSON file. Invalid entries are skipped with a warning.

    A missing file means an empty allowlist. Malformed JSON raises ValueError.
    """
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object of name: target pairs")

    apps: dict[str, str] = {}
    for name, target in data.items():
        key = str(name).strip().lower()
        value = str(target).strip() if isinstance(target, str) else ""
        if not key or not is_valid_target(value):
            logger.warning("Skipping invalid app allowlist entry '%s'", name)
            continue
        apps[key] = value
    return apps
