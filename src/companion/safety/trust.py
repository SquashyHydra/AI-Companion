"""Per-turn trust. Web results are untrusted, so state-changing tools need a yes after one is read.

The flag is reset at the start of each user turn and set as soon as a web search is sent. While
it is set, every tool in STATE_CHANGING_TOOLS asks for confirmation, even if it normally would not.
"""

from __future__ import annotations

import threading

STATE_CHANGING_TOOLS: frozenset[str] = frozenset(
    {
        "open_file",
        "open_app",
        "volume",
        "set_timer",
        "set_reminder",
        "cancel_timer",
        "remember",
        "forget",
        "clear_notes",
    }
)


class TurnTrust:
    """Thread-safe flag: has untrusted text (web results) entered the current turn."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._untrusted = False

    def begin(self) -> None:
        """Start a new user turn with no untrusted text yet."""
        with self._lock:
            self._untrusted = False

    def mark_untrusted(self) -> None:
        with self._lock:
            self._untrusted = True

    @property
    def untrusted(self) -> bool:
        with self._lock:
            return self._untrusted


turn_trust = TurnTrust()
