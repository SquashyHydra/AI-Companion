"""Plain-text record of each conversation, one file per conversation, under logs/conversations/.

A file is created on the first completed turn after a new conversation starts, so an empty
conversation leaves no file. Turns are appended as they finish. Nothing here is sent to the model.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class ConversationLog:
    def __init__(
        self,
        directory: Path,
        *,
        enabled: bool = True,
        clock: Callable[[], datetime] = datetime.now,
    ) -> None:
        self._dir = Path(directory)
        self._enabled = enabled
        self._clock = clock
        self._lock = threading.Lock()
        self._path: Path | None = None

    @property
    def path(self) -> Path | None:
        """The file for the current conversation, or None until the first turn is written."""
        return self._path

    def start_new(self) -> None:
        """Begin a new conversation. The next turn opens a new file."""
        with self._lock:
            self._path = None

    def write_turn(self, user_text: str, reply: str) -> None:
        """Append one completed turn. Failures are logged and never stop the conversation."""
        if not self._enabled:
            return
        try:
            with self._lock:
                if self._path is None:
                    self._path = self._new_file()
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(f"You: {user_text.strip()}\n")
                    handle.write(f"Companion: {reply.strip()}\n\n")
        except OSError:
            logger.warning("could not write the conversation log", exc_info=True)

    def _new_file(self) -> Path:
        self._dir.mkdir(parents=True, exist_ok=True)
        stamp = self._clock().strftime("%Y%m%d-%H%M%S")
        candidate = self._dir / f"conversation-{stamp}.txt"
        counter = 2
        while candidate.exists():
            candidate = self._dir / f"conversation-{stamp}-{counter}.txt"
            counter += 1
        return candidate
