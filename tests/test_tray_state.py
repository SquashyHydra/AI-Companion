"""Tests for the tray Pause/Resume item, the status line, and the voice loop's listening notifications."""

import queue
from pathlib import Path

from companion.tray import TrayIcon, status_text, toggle_text
from companion.voice.keys import LineEditor
from companion.voice.loop import VoiceLoop
from companion.voice.machine import ListenMachine


class FakeIcon:
    def __init__(self):
        self.title = ""
        self.updates = 0

    def update_menu(self):
        self.updates += 1


class FakePipeline:
    def start(self):
        pass

    def stop(self):
        pass


class FakeKeys:
    """One empty poll, then Quit. Keeps the loop alive long enough to handle a queued toggle."""

    def __init__(self):
        self._polls = 0

    def poll(self, confirming: bool = False):
        from companion.voice.machine import Quit

        self._polls += 1
        return [] if self._polls == 1 else [Quit()]


def test_labels_describe_the_state_and_the_action():
    assert status_text(True) == "Status: listening"
    assert status_text(False) == "Status: paused"
    assert toggle_text(True) == "Pause listening"
    assert toggle_text(False) == "Resume listening"


def test_set_listening_updates_tooltip_and_refreshes_menu():
    tray = TrayIcon(on_quit=lambda: None, logs_dir=Path("logs"))
    icon = FakeIcon()
    tray._icon = icon
    tray.set_listening(False)
    assert icon.title == "PC Companion: paused"
    assert icon.updates == 1
    tray.set_listening(True)
    assert icon.title == "PC Companion: listening"


def test_set_listening_before_start_is_harmless():
    tray = TrayIcon(on_quit=lambda: None, logs_dir=Path("logs"))
    tray.set_listening(False)
    assert tray._listening is False


def test_toggle_menu_action_calls_the_callback():
    calls: list[str] = []
    tray = TrayIcon(
        on_quit=lambda: None, logs_dir=Path("logs"), on_toggle=lambda: calls.append("toggle")
    )
    tray._toggle(None, None)
    assert calls == ["toggle"]


def test_toggle_without_callback_does_nothing():
    tray = TrayIcon(on_quit=lambda: None, logs_dir=Path("logs"))
    tray._toggle(None, None)  # must not raise


def _loop(events: queue.Queue) -> VoiceLoop:
    return VoiceLoop(
        machine=ListenMachine(barge_in_ms=400, enter_pauses=False),
        pipeline=FakePipeline(),
        transcribe=lambda audio: None,
        stream=lambda *args: None,
        run_command=lambda text: "",
        new_history=lambda: [],
        keys=FakeKeys(),
        editor=LineEditor(lambda text: None),
        events=events,
        answers=queue.Queue(),
        low_confidence_logprob=-1.0,
        write=lambda text: None,
        tick_s=0.01,
    )


def test_listener_reports_start_then_each_pause_and_resume():
    events: queue.Queue = queue.Queue()
    loop = _loop(events)
    seen: list[bool] = []
    loop.set_listening_listener(seen.append)
    loop.request_toggle()
    assert loop.run() == 0
    assert seen == [True, False]


def test_listener_is_not_called_when_state_does_not_change():
    loop = _loop(queue.Queue())
    seen: list[bool] = []
    loop.set_listening_listener(seen.append)
    assert loop.run() == 0
    assert seen == [True]


def test_listener_errors_do_not_stop_the_loop():
    loop = _loop(queue.Queue())

    def broken(value: bool) -> None:
        raise RuntimeError("tray gone")

    loop.set_listening_listener(broken)
    assert loop.run() == 0
