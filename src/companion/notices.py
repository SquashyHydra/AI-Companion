"""Thread-safe queue for messages raised off the main loop, such as timers firing.

Background threads call post(). The voice and text loops call drain() between turns and
show or speak whatever is waiting.
"""

from __future__ import annotations

import queue


class NoticeQueue:
    def __init__(self) -> None:
        self._items: queue.Queue[str] = queue.Queue()

    def post(self, text: str) -> None:
        self._items.put(text)

    def drain(self) -> list[str]:
        items: list[str] = []
        while True:
            try:
                items.append(self._items.get_nowait())
            except queue.Empty:
                return items


notices = NoticeQueue()
