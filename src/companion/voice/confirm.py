"""Confirmation for the continuous loop. The question is shown by the main thread, not read with input()."""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable

from companion.voice.machine import ConfirmRequest, ConfirmTimedOut, Event

logger = logging.getLogger("companion.confirm")

DEFAULT_TIMEOUT_S = 30.0
POLL_S = 0.1
"""How often the waiting thread checks the stop flag. Keeps shutdown prompt."""


class EventConfirmer:
    """Implements the Confirmer protocol using the voice loop's event queue.

    confirm() runs on the reply worker thread. It posts a ConfirmRequest, then waits for an
    answer that the main thread puts on the answers queue. No answer in time counts as no.
    Stale answers from earlier questions are drained first, so an old reply cannot approve
    a new action.

    The wait is sliced into short polls instead of one blocking get(), so setting `stop`
    (on shutdown) ends a pending confirmation at once, declined and logged.
    """

    def __init__(
        self,
        post: Callable[[Event], None],
        answers: queue.Queue[bool],
        timeout_s: float = DEFAULT_TIMEOUT_S,
        stop: threading.Event | None = None,
    ) -> None:
        self._post = post
        self._answers = answers
        self._timeout_s = timeout_s
        self._stop = stop or threading.Event()

    def confirm(self, question: str) -> bool:
        _drain(self._answers)
        self._post(ConfirmRequest(question, timeout_s=self._timeout_s))
        deadline = time.monotonic() + self._timeout_s
        while True:
            if self._stop.is_set():
                logger.info("confirmation cancelled by shutdown; declined")
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.info("confirmation timed out after %g s; declined", self._timeout_s)
                self._post(ConfirmTimedOut())
                return False
            try:
                answer = bool(self._answers.get(timeout=min(POLL_S, remaining)))
            except queue.Empty:
                continue
            logger.info("confirmation answered %s", "yes" if answer else "no")
            return answer


def _drain(answers: queue.Queue[bool]) -> None:
    while True:
        try:
            answers.get_nowait()
        except queue.Empty:
            return
