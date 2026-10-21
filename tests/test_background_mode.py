"""Tests for background startup: no-keyboard mode, Enter-free hints, and the mode/background flags."""

import pytest

from companion.audio.segmenter import SpeechOngoing
from companion.main import build_parser
from companion.voice.keys import NoKeys
from companion.voice.machine import ListenMachine, MicFailed, Print, StartMic, StopSpeech, Toggle


def _prints(actions) -> list[str]:
    return [action.text for action in actions if isinstance(action, Print)]


def test_enter_pauses_hints_are_unchanged_by_default():
    machine = ListenMachine(barge_in_ms=400)
    assert _prints(machine.start()) == [
        "Listening. Speak anytime. Press Enter to pause. Ctrl+C to quit."
    ]


def test_background_start_hint_points_to_the_tray_icon():
    machine = ListenMachine(barge_in_ms=400, enter_pauses=False)
    hint = _prints(machine.start())[0]
    assert "Enter" not in hint
    assert "tray icon" in hint


def test_background_pause_and_resume_use_toggle_without_keys():
    machine = ListenMachine(barge_in_ms=400, enter_pauses=False)
    paused = machine.handle(Toggle())
    assert machine.paused
    assert _prints(paused) == ["Paused. Use the tray icon to listen again."]
    resumed = machine.handle(Toggle())
    assert not machine.paused
    assert any(isinstance(action, StartMic) for action in resumed)


def test_background_mic_failure_message_has_no_enter_instruction():
    machine = ListenMachine(barge_in_ms=400, enter_pauses=False, mic_retry_s=5)
    text = _prints(machine.handle(MicFailed("no device")))[0]
    assert "Retrying every 5 seconds." in text
    assert "Enter" not in text


def test_keyboard_mic_failure_message_keeps_enter_instruction():
    machine = ListenMachine(barge_in_ms=400, mic_retry_s=5)
    text = _prints(machine.handle(MicFailed("no device")))[0]
    assert "Press Enter to try the microphone now." in text


def test_no_keys_never_yields_events():
    keys = NoKeys()
    assert keys.poll() == []
    assert keys.poll(confirming=True) == []


def test_speaker_mode_does_not_barge_in_on_voice_activity():
    machine = ListenMachine(
        barge_in_ms=400, speech_busy=lambda: True, vad_barge_in=False
    )
    actions = machine.handle(SpeechOngoing(voiced_ms=5000))
    assert actions == []
    assert machine.speaking


def test_headphone_mode_still_barges_in_while_speech_plays():
    machine = ListenMachine(barge_in_ms=400, speech_busy=lambda: True)
    actions = machine.handle(SpeechOngoing(voiced_ms=5000))
    assert any(isinstance(action, StopSpeech) for action in actions)


def test_parser_accepts_background_flags():
    args = build_parser().parse_args(["--tray", "--mode", "continuous", "--background"])
    assert args.tray and args.background
    assert args.mode == "continuous"


def test_parser_defaults_leave_mode_to_config():
    args = build_parser().parse_args([])
    assert args.mode is None
    assert not args.background


def test_parser_rejects_unknown_mode():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--mode", "shouting"])
