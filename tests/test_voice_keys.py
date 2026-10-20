"""Tests for the keyboard line editor, key reader, and the event-based confirmer."""

import logging
import queue
import threading
import time

from companion.voice.confirm import EventConfirmer
from companion.voice.keys import KeyReader, LineEditor
from companion.voice.machine import (
    Answer,
    Cancel,
    Command,
    ConfirmRequest,
    ConfirmTimedOut,
    Quit,
    Toggle,
    Typed,
)


def _editor() -> tuple[LineEditor, list[str]]:
    written: list[str] = []
    return LineEditor(written.append), written


def _type(editor: LineEditor, keys: str) -> list:
    events = []
    for key in keys:
        events += editor.press(key)
    return events


def test_typing_echoes_and_enter_submits_text():
    editor, written = _editor()
    assert _type(editor, "hi") == []
    assert editor.text == "hi"
    assert "".join(written) == "hi"
    assert _type(editor, "\r") == [Typed("hi")]
    assert editor.text == ""


def test_empty_enter_toggles_listening():
    editor, _ = _editor()
    assert _type(editor, "\r") == [Toggle()]


def test_slash_text_is_a_command():
    editor, _ = _editor()
    assert _type(editor, "/tools\r") == [Command("/tools")]


def test_backspace_removes_the_last_character():
    editor, _ = _editor()
    _type(editor, "ab")
    _type(editor, "\b")
    assert editor.text == "a"


def test_escape_clears_the_line():
    editor, _ = _editor()
    _type(editor, "abc")
    _type(editor, "\x1b")
    assert editor.text == ""


def test_ctrl_c_quits():
    editor, _ = _editor()
    assert _type(editor, "\x03") == [Quit()]


def test_arrow_keys_are_ignored():
    editor, _ = _editor()
    # Arrow keys send a prefix byte, then a code. The code must not become text.
    assert _type(editor, "\xe0H") == []
    assert _type(editor, "\x00K") == []
    assert editor.text == ""


def test_non_printable_keys_are_ignored():
    editor, written = _editor()
    _type(editor, "\x01")
    assert editor.text == ""
    assert written == []


def test_key_reader_drains_every_waiting_key():
    editor, _ = _editor()
    waiting = list("ab\r")
    reader = KeyReader(
        editor,
        kbhit=lambda: bool(waiting),
        getwch=lambda: waiting.pop(0),
    )
    assert reader.poll() == [Typed("ab")]
    assert reader.poll() == []


def test_confirmer_ignores_stale_answers_and_returns_the_new_one():
    answers: queue.Queue[bool] = queue.Queue()
    answers.put(True)  # left over from an earlier question
    posted = []

    def post(event):
        posted.append(event)
        if isinstance(event, ConfirmRequest):
            answers.put(False)

    confirmer = EventConfirmer(post, answers, timeout_s=1.0)
    assert confirmer.confirm("Delete it?") is False
    assert isinstance(posted[0], ConfirmRequest)
    assert posted[0].question == "Delete it?"


def test_confirmer_declines_on_timeout():
    answers: queue.Queue[bool] = queue.Queue()
    posted = []
    confirmer = EventConfirmer(posted.append, answers, timeout_s=0.05)
    assert confirmer.confirm("Delete it?") is False
    assert any(isinstance(event, ConfirmTimedOut) for event in posted)


def test_timeout_is_logged(caplog):
    caplog.set_level(logging.INFO, logger="companion.confirm")
    confirmer = EventConfirmer([].append, queue.Queue(), timeout_s=0.05)
    confirmer.confirm("Delete it?")
    assert "timed out" in caplog.text


def test_request_carries_the_configured_timeout():
    posted = []
    confirmer = EventConfirmer(posted.append, queue.Queue(), timeout_s=0.05)
    confirmer.confirm("Delete it?")
    assert posted[0].timeout_s == 0.05


def test_yes_answer_is_returned():
    answers: queue.Queue[bool] = queue.Queue()

    def post(event):
        if isinstance(event, ConfirmRequest):
            answers.put(True)

    assert EventConfirmer(post, answers, timeout_s=1.0).confirm("Delete it?") is True


def test_stop_ends_a_pending_confirmation_promptly_without_timing_out():
    stop = threading.Event()
    posted = []
    confirmer = EventConfirmer(posted.append, queue.Queue(), timeout_s=30.0, stop=stop)
    threading.Timer(0.05, stop.set).start()
    started = time.monotonic()
    assert confirmer.confirm("Delete it?") is False
    assert time.monotonic() - started < 2.0
    assert not any(isinstance(event, ConfirmTimedOut) for event in posted)


# ----- key routing -----


def _reader(keys: str) -> tuple[KeyReader, LineEditor, list[str]]:
    editor, written = _editor()
    waiting = list(keys)
    reader = KeyReader(
        editor,
        kbhit=lambda: bool(waiting),
        getwch=lambda: waiting.pop(0),
    )
    return reader, editor, written


def test_y_and_n_during_a_confirmation_are_answers():
    reader, editor, written = _reader("yNn")
    assert reader.poll(confirming=True) == [Answer(True), Answer(False), Answer(False)]
    assert written == []
    assert editor.text == ""


def test_other_keys_during_a_confirmation_are_not_echoed():
    reader, editor, written = _reader("hello")
    assert reader.poll(confirming=True) == []
    assert written == []
    assert editor.text == ""


def test_enter_during_a_confirmation_is_ignored():
    reader, _, written = _reader("\r")
    assert reader.poll(confirming=True) == []
    assert written == []


def test_ctrl_c_during_a_confirmation_cancels_it_instead_of_quitting():
    reader, _, _ = _reader("\x03")
    assert reader.poll(confirming=True) == [Cancel()]


def test_idle_keys_still_echo_and_submit():
    reader, _, written = _reader("ab\r")
    assert reader.poll() == [Typed("ab")]
    assert "".join(written).startswith("ab")


def test_routing_returns_to_normal_once_the_confirmation_ends():
    waiting = list("y")
    editor, written = _editor()
    reader = KeyReader(editor, kbhit=lambda: bool(waiting), getwch=lambda: waiting.pop(0))
    assert reader.poll(confirming=True) == [Answer(True)]
    waiting.extend("a\r")
    assert reader.poll(confirming=False) == [Typed("a")]
    assert "".join(written).startswith("a")


def test_prefix_bytes_are_skipped_in_both_modes():
    reader, _, _ = _reader("\xe0y")
    assert reader.poll(confirming=True) == []
    reader, editor, _ = _reader("\x00K")
    assert reader.poll() == []
    assert editor.text == ""
