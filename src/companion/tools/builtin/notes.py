"""Notes tools: append to and read back markdown notes, one file per topic."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from companion.config import get_config
from companion.tools.registry import tool

MAX_NOTE_CHARS = 2000
MAX_READ_NOTES = 100
_TOPIC_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
_WINDOWS_RESERVED = {"con", "prn", "aux", "nul"} | {f"{p}{i}" for p in ("com", "lpt") for i in range(1, 10)}


def _notes_dir() -> Path:
    return get_config().tools.notes_dir


def _topic_error(topic: str) -> str | None:
    if not _TOPIC_RE.fullmatch(topic) or topic.lower() in _WINDOWS_RESERVED:
        return "Error: topic must be 1-40 letters, digits, '-' or '_' (not a reserved Windows name)."
    return None


@tool
def take_note(text: str, topic: str = "general") -> str:
    """Save a note under a topic so it can be read back later.

    Args:
        text: The note content to save.
        topic: Topic name made of letters, digits, '-' or '_'.
    """
    error = _topic_error(topic)
    if error:
        return error
    note = " ".join(text.split())
    if not note:
        return "Error: the note is empty."
    if len(note) > MAX_NOTE_CHARS:
        return f"Error: the note is longer than {MAX_NOTE_CHARS} characters."

    path = _notes_dir() / f"{topic}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", encoding="utf-8") as handle:
        if is_new:
            handle.write(f"# {topic}\n\n")
        handle.write(f"- [{datetime.now():%Y-%m-%d %H:%M}] {note}\n")
    return f"Saved note to topic '{topic}'."


@tool
def read_notes(topic: str = "general", limit: int = 20) -> str:
    """Read the most recent notes saved under a topic.

    Args:
        topic: Topic name to read from.
        limit: How many of the most recent notes to return, from 1 to 100.
    """
    error = _topic_error(topic)
    if error:
        return error
    if not 1 <= limit <= MAX_READ_NOTES:
        return f"Error: limit must be between 1 and {MAX_READ_NOTES}."

    entries = _read_entries(_notes_dir() / f"{topic}.md")
    if not entries:
        return f"No notes found for topic '{topic}'."
    return "\n".join(entries[-limit:])


@tool(requires_confirmation=True)
def clear_notes(topic: str = "general") -> str:
    """Permanently delete every note saved under a topic. The user must confirm first.

    Args:
        topic: Topic name whose notes will be deleted.
    """
    error = _topic_error(topic)
    if error:
        return error
    path = _notes_dir() / f"{topic}.md"
    if not path.is_file():
        return f"No notes found for topic '{topic}'. Nothing was cleared."
    count = len(_read_entries(path))
    path.unlink()
    return f"Cleared {count} note(s) from topic '{topic}'."


def _read_entries(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("- [")]
