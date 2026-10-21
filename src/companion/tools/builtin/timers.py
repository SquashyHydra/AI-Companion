"""Timers and reminders.

They run in memory. When a store file is configured (see configure_store), they are also saved
there, restored at startup, and a timer that came due while the companion was closed is announced once.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import NamedTuple

from companion.notices import notices
from companion.tools.registry import tool

MAX_MINUTES = 24 * 60
MIN_MINUTES = 0.1
MAX_ACTIVE = 20
MAX_LABEL_CHARS = 60
_CLOCK_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*$")


logger = logging.getLogger(__name__)
STORE_VERSION = 1


@dataclass
class _Scheduled:
    timer_id: int
    kind: str  # "timer" or "reminder"
    label: str
    due: float  # time.time() value when it fires. Wall clock, so it survives a restart.
    handle: threading.Timer


class _Saved(NamedTuple):
    timer_id: int
    kind: str
    label: str
    due: float


_lock = threading.Lock()
_scheduled: dict[int, _Scheduled] = {}
_next_id = 1
_store_path: Path | None = None
_announce_missed = True


def configure_store(path: Path | None, *, announce_missed: bool = True) -> int:
    """Save timers to a JSON file and restore the ones saved there. Returns how many were restored.

    Pass None to keep timers in memory only. That is the default, and what the tests use.
    """
    global _store_path, _announce_missed
    with _lock:
        _store_path = path
        _announce_missed = announce_missed
    if path is None:
        return 0
    return _restore(time.time())


def _restore(now: float) -> int:
    global _next_id
    next_id, saved = _load_store()
    restored = 0
    missed: list[_Saved] = []
    with _lock:
        _next_id = max(_next_id, next_id)
        for item in saved:
            if item.due <= now:
                missed.append(item)
            elif len(_scheduled) < MAX_ACTIVE:
                _start_locked(item.timer_id, item.kind, item.label, item.due)
                restored += 1
            else:
                logger.warning("timer %d not restored: limit of %d reached", item.timer_id, MAX_ACTIVE)
        _save_locked()
    for item in missed:
        if _announce_missed:
            notices.post(_missed_text(item))
        else:
            logger.info("timer %d came due while closed and was not announced", item.timer_id)
    return restored


def _missed_text(item: _Saved) -> str:
    kind = "Timer" if item.kind == "timer" else "Reminder"
    due = datetime.fromtimestamp(item.due)
    return f"{kind} missed while the companion was closed: {item.label} (due {due:%H:%M})."


def _load_store() -> tuple[int, list[_Saved]]:
    path = _store_path
    if path is None or not path.is_file():
        return 1, []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        next_id = int(data.get("next_id", 1))
        raw_items = data.get("items", [])
        if not isinstance(raw_items, list):
            raise ValueError("items must be a list")
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        logger.warning("timers store %s is unreadable, starting empty: %s", path, exc)
        return 1, []
    saved: list[_Saved] = []
    for entry in raw_items:
        try:
            kind = str(entry["kind"])
            if kind not in ("timer", "reminder"):
                raise ValueError(f"unknown kind {kind!r}")
            saved.append(
                _Saved(
                    int(entry["id"]),
                    kind,
                    _clean_label(str(entry["label"]), kind),
                    float(entry["due"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            logger.warning("skipping a bad timer entry in %s: %r", path, entry)
    return next_id, saved


def _save_locked() -> None:
    """Write the running timers to the store file. The caller holds _lock."""
    path = _store_path
    if path is None:
        return
    payload = {
        "version": STORE_VERSION,
        "next_id": _next_id,
        "items": [
            {"id": item.timer_id, "kind": item.kind, "label": item.label, "due": item.due}
            for item in sorted(_scheduled.values(), key=lambda item: item.due)
        ],
    }
    tmp_name: str | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".timers-", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp_name, path)
    except OSError:
        logger.exception("could not save timers to %s", path)
        if tmp_name is not None:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)


def _clean_label(label: str, default: str) -> str:
    return " ".join(label.split())[:MAX_LABEL_CHARS] or default


def _fire(timer_id: int) -> None:
    with _lock:
        item = _scheduled.pop(timer_id, None)
        if item is None:
            return
        _save_locked()
    prefix = "Timer done" if item.kind == "timer" else "Reminder"
    notices.post(f"{prefix}: {item.label}.")


def _schedule(seconds: float, kind: str, label: str) -> int:
    """Start a background timer, save it, and return its number."""
    global _next_id
    with _lock:
        timer_id = _next_id
        _next_id += 1
        _start_locked(timer_id, kind, label, time.time() + seconds)
        _save_locked()
    return timer_id


def _start_locked(timer_id: int, kind: str, label: str, due: float) -> None:
    handle = threading.Timer(max(0.0, due - time.time()), _fire, args=(timer_id,))
    handle.daemon = True
    _scheduled[timer_id] = _Scheduled(timer_id, kind, label, due, handle)
    handle.start()


def _room_left() -> bool:
    with _lock:
        return len(_scheduled) < MAX_ACTIVE


def _describe_minutes(minutes: float) -> str:
    if minutes < 1:
        return f"{round(minutes * 60)} seconds"
    return f"{minutes:g} minutes"


@tool
def set_timer(minutes: float, label: str = "timer") -> str:
    """Set a countdown timer. The companion announces it when the time is up.

    Args:
        minutes: How long to wait, in minutes. Between 0.1 and 1440.
        label: A short name for the timer, such as "tea" or "laundry".
    """
    if not MIN_MINUTES <= minutes <= MAX_MINUTES:
        return "Error: minutes must be between 0.1 and 1440."
    name = _clean_label(label, "timer")
    if not _room_left():
        return f"Error: {MAX_ACTIVE} timers and reminders are already running. Cancel one first."
    timer_id = _schedule(minutes * 60, "timer", name)
    return f"Timer {timer_id} set for {_describe_minutes(minutes)}: {name}."


@tool
def set_reminder(at: str, label: str = "reminder") -> str:
    """Set a reminder for a clock time today, or tomorrow if that time has passed.

    Args:
        at: The time in 24-hour format, such as "18:30".
        label: What to remind you about, such as "call mum".
    """
    match = _CLOCK_RE.match(at)
    if not match or int(match.group(1)) > 23 or int(match.group(2)) > 59:
        return "Error: at must be a 24-hour time such as 18:30."
    name = _clean_label(label, "reminder")
    if not _room_left():
        return f"Error: {MAX_ACTIVE} timers and reminders are already running. Cancel one first."
    now = datetime.now()
    target = now.replace(hour=int(match.group(1)), minute=int(match.group(2)), second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    seconds = (target - now).total_seconds()
    timer_id = _schedule(seconds, "reminder", name)
    return f"Reminder {timer_id} set for {target:%H:%M}: {name}."


@tool
def list_timers() -> str:
    """List the timers and reminders that are still running, with the time left."""
    with _lock:
        items = sorted(_scheduled.values(), key=lambda item: item.due)
        now = time.time()
        lines = [
            f"  {item.timer_id}: {item.kind} '{item.label}', {max(0.0, item.due - now) / 60:.1f} min left"
            for item in items
        ]
    return "\n".join(lines) or "No timers or reminders are running."


@tool
def cancel_timer(timer_id: int) -> str:
    """Cancel a running timer or reminder by its number, as shown by list_timers.

    Args:
        timer_id: The number of the timer or reminder to cancel.
    """
    with _lock:
        item = _scheduled.pop(timer_id, None)
        if item is not None:
            _save_locked()
    if item is None:
        return f"Error: no timer or reminder with number {timer_id}."
    item.handle.cancel()
    return f"Cancelled {item.kind} {timer_id}: {item.label}."
