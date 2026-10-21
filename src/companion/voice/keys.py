"""Keyboard input for the voice loop. Polled from the main thread so Ctrl+C always works.

Windows only in practice: msvcrt supplies the key source. Both functions are injectable for tests.
"""

from __future__ import annotations

from collections.abc import Callable

from companion.voice.machine import Answer, Cancel, Command, Event, Quit, Toggle, Typed

PROMPT = "You> "
_SPECIAL_PREFIXES = ("\x00", "\xe0")
"""Arrow and function keys send a prefix byte followed by a second code. Both are ignored."""


class LineEditor:
    """A small line buffer with echo. Turns key presses into machine events."""

    def __init__(self, write: Callable[[str], None], prompt: str = PROMPT) -> None:
        self._write = write
        self._prompt = prompt
        self._buffer = ""
        self._skip_next = False

    @property
    def text(self) -> str:
        return self._buffer

    def press(self, key: str) -> list[Event]:
        if self._skip_next:
            self._skip_next = False
            return []
        if key in _SPECIAL_PREFIXES:
            self._skip_next = True
            return []
        if key == "\x03":
            return [Quit()]
        if key in ("\r", "\n"):
            return self._submit()
        if key in ("\b", "\x7f"):
            if self._buffer:
                self._buffer = self._buffer[:-1]
                self._write("\b \b")
            return []
        if key == "\x1b":
            self.clear_line()
            self._buffer = ""
            return []
        if len(key) != 1 or not key.isprintable():
            return []
        self._buffer += key
        self._write(key)
        return []

    def redraw(self) -> None:
        self._write(self._prompt + self._buffer)

    def show(self, text: str) -> None:
        """Print a line above the prompt and keep what the user has typed so far."""
        self.clear_line()
        self._write(text + "\n")
        self.redraw()

    def _submit(self) -> list[Event]:
        text = self._buffer
        self._buffer = ""
        self._write("\n")
        if not text.strip():
            return [Toggle()]
        if text.strip().startswith("/"):
            return [Command(text)]
        return [Typed(text)]

    def clear_line(self) -> None:
        """Erase the prompt line so other output can be written there."""
        width = len(self._prompt) + len(self._buffer)
        self._write("\r" + " " * width + "\r")


class ConfirmKeys:
    """Keys while a confirmation is pending. Nothing is echoed and nothing reaches the editor.

    y or n answers. Ctrl+C cancels the confirmation (declined). Enter and every other key are
    ignored, so Enter cannot pause listening or answer for the user.
    """

    def press(self, key: str) -> list[Event]:
        if key == "\x03":
            return [Cancel()]
        if key in ("y", "Y"):
            return [Answer(True)]
        if key in ("n", "N"):
            return [Answer(False)]
        return []


class KeyReader:
    """The single keyboard reader. Reads every waiting key without blocking.

    While a confirmation is pending (confirming=True) keys go to ConfirmKeys. Otherwise they
    go to the line editor, which echoes them. Special-key prefix bytes are skipped in both
    modes, so an arrow key never leaks a second code into an answer.
    """

    def __init__(
        self,
        editor: LineEditor,
        kbhit: Callable[[], bool] | None = None,
        getwch: Callable[[], str] | None = None,
        confirm_keys: ConfirmKeys | None = None,
    ) -> None:
        self._editor = editor
        self._confirm_keys = confirm_keys or ConfirmKeys()
        self._kbhit = kbhit or _default_kbhit
        self._getwch = getwch or _default_getwch
        self._skip_next = False

    def poll(self, confirming: bool = False) -> list[Event]:
        events: list[Event] = []
        while self._kbhit():
            key = self._getwch()
            if self._skip_next:
                self._skip_next = False
                continue
            if key in _SPECIAL_PREFIXES:
                self._skip_next = True
                continue
            if confirming:
                events += self._confirm_keys.press(key)
            else:
                events += self._editor.press(key)
        return events


class NoKeys:
    """Stands in for KeyReader in background mode. There is no console, so no keys ever arrive."""

    def poll(self, confirming: bool = False) -> list[Event]:
        return []


def _default_kbhit() -> bool:
    import msvcrt

    return bool(msvcrt.kbhit())


def _default_getwch() -> str:
    import msvcrt

    return msvcrt.getwch()
