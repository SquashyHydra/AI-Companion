"""Confirmation for risky tool calls. The registry depends only on the Confirmer protocol."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Protocol

logger = logging.getLogger("companion.confirm")

MAX_ATTEMPTS = 3
YES_ANSWERS = frozenset({"y", "yes"})
NO_ANSWERS = frozenset({"n", "no"})


class Confirmer(Protocol):
    """Asks the user a yes/no question. Returns True only on a clear yes."""

    def confirm(self, question: str) -> bool: ...


class TerminalConfirmer:
    """Asks in the terminal and re-asks on unclear answers. No valid answer counts as no."""

    def __init__(self, ask: Callable[[str], str] = input, max_attempts: int = MAX_ATTEMPTS) -> None:
        self._ask = ask
        self._max_attempts = max_attempts

    def confirm(self, question: str) -> bool:
        for attempt in range(1, self._max_attempts + 1):
            try:
                answer = self._ask(f"{question} [y/n] ")
            except (EOFError, KeyboardInterrupt):
                logger.info("confirmation cancelled; declined")
                return False
            normalized = answer.strip().lower()
            if normalized in YES_ANSWERS:
                logger.info("confirmation answered yes")
                return True
            if normalized in NO_ANSWERS:
                logger.info("confirmation answered no")
                return False
            logger.info(
                "confirmation invalid answer %r (attempt %d of %d)",
                answer,
                attempt,
                self._max_attempts,
            )
            print("Please answer y or n.")
        logger.info("confirmation gave no valid answer; declined")
        return False
