"""Tests for the voice state machine. Pure logic: events in, actions out."""

import numpy as np

from companion.audio.segmenter import Noise, SpeechOngoing, Utterance
from companion.llm.client import TurnOutcome
from companion.voice.machine import (
    Answer,
    ApplyOutcome,
    Cancel,
    CancelReply,
    ConfirmRequest,
    ConfirmTimedOut,
    Command,
    MicFailed,
    Notice,
    Print,
    Quit,
    ReplyDelta,
    ReplyDone,
    ReplyFailed,
    ResetHistory,
    ResolveConfirmation,
    RunCommand,
    Shutdown,
    StartMic,
    StartReply,
    State,
    Stream,
    StopMic,
    StopSpeech,
    Toggle,
    Transcribe,
    Transcript,
    Typed,
    ListenMachine,
    parse_yes_no,
)

BARGE_IN_MS = 400


def machine() -> ListenMachine:
    return ListenMachine(barge_in_ms=BARGE_IN_MS)


def _audio() -> np.ndarray:
    return np.zeros(512, dtype=np.float32)


def _outcome(reply: str = "ok", interrupted: bool = False) -> TurnOutcome:
    return TurnOutcome([{"role": "system", "content": "s"}], reply, interrupted=interrupted)


def _kinds(actions) -> list[type]:
    return [type(action) for action in actions]


def _start_reply(m: ListenMachine, text: str = "hello") -> int:
    actions = m.handle(Typed(text))
    start = next(action for action in actions if isinstance(action, StartReply))
    return start.turn


# ----- start and pause -----


def test_start_opens_the_microphone_and_says_so():
    actions = machine().start()
    assert _kinds(actions) == [StartMic, Print]
    assert "Listening" in actions[1].text


def test_toggle_pauses_then_resumes_listening():
    m = machine()
    actions = m.handle(Toggle())
    assert _kinds(actions) == [StopMic, Print]
    assert m.state is State.PAUSED

    actions = m.handle(Toggle())
    assert _kinds(actions) == [StartMic, Print]
    assert m.state is State.LISTENING


def test_pausing_during_a_reply_cancels_it():
    m = machine()
    _start_reply(m)
    actions = m.handle(Toggle())
    assert CancelReply in _kinds(actions)
    assert StopMic in _kinds(actions)


def test_enter_during_a_confirmation_neither_pauses_nor_answers():
    m = machine()
    m.handle(ConfirmRequest("Delete the file?"))
    actions = m.handle(Toggle())
    assert _kinds(actions) == [Print]
    assert m.paused is False
    assert m.awaiting_confirm is True
    assert ResolveConfirmation(False) not in actions


def test_answer_key_resolves_the_pending_confirmation():
    m = machine()
    m.handle(ConfirmRequest("Delete the file?"))
    actions = m.handle(Answer(True))
    assert ResolveConfirmation(True) in actions
    assert m.awaiting_confirm is False


def test_answer_key_without_a_confirmation_is_ignored():
    m = machine()
    assert m.handle(Answer(False)) == []


def test_ctrl_c_cancels_a_confirmation_without_quitting():
    m = machine()
    m.handle(ConfirmRequest("Delete the file?"))
    actions = m.handle(Cancel())
    assert ResolveConfirmation(False) in actions
    assert Shutdown not in _kinds(actions)
    assert m.awaiting_confirm is False


def test_ctrl_c_with_no_confirmation_quits():
    actions = machine().handle(Cancel())
    assert Shutdown in _kinds(actions)


def test_typed_command_is_refused_during_a_confirmation():
    m = machine()
    m.handle(ConfirmRequest("Delete the file?"))
    actions = m.handle(Command("/reset"))
    assert _kinds(actions) == [Print]
    assert m.awaiting_confirm is True


def test_confirm_prompt_states_the_timeout():
    actions = machine().handle(ConfirmRequest("Delete the file?", timeout_s=30))
    assert "30 seconds" in actions[0].text


def test_speech_is_ignored_while_paused():
    m = machine()
    m.handle(Toggle())
    assert m.handle(SpeechOngoing(voiced_ms=500)) == []
    assert m.handle(Utterance(audio=_audio(), seconds=0.5)) == []


def test_empty_enter_with_no_pause_does_nothing_while_listening():
    m = machine()
    assert m.handle(Typed("   ")) == []


# ----- typed text and replies -----


def test_typed_text_starts_a_reply():
    m = machine()
    actions = m.handle(Typed("what time is it"))
    assert actions == [StartReply(1, "what time is it")]
    assert m.state is State.THINKING


def test_reply_deltas_stream_and_done_applies_the_outcome():
    m = machine()
    turn = _start_reply(m)
    assert m.handle(ReplyDelta(turn, "Hi")) == [Stream("Hi")]
    actions = m.handle(ReplyDone(turn, _outcome("Hi")))
    assert _kinds(actions) == [ApplyOutcome]
    assert m.state is State.LISTENING


def test_stale_reply_events_are_ignored():
    m = machine()
    _start_reply(m)
    assert m.handle(ReplyDone(99, _outcome())) == []
    assert m.handle(ReplyDelta(99, "x")) == []


def test_reply_failure_is_printed_and_the_turn_ends():
    m = machine()
    turn = _start_reply(m)
    actions = m.handle(ReplyFailed(turn, "server down"))
    assert _kinds(actions) == [Print]
    assert "server down" in actions[0].text
    assert not m.reply_running


def test_text_during_a_reply_cancels_it_and_runs_the_new_text_after():
    m = machine()
    turn = _start_reply(m, "first")
    actions = m.handle(Typed("second"))
    assert actions == [CancelReply()]

    actions = m.handle(ReplyDone(turn, _outcome("partial", interrupted=True)))
    assert _kinds(actions) == [ApplyOutcome, StartReply]
    assert actions[1] == StartReply(2, "second")


def test_deltas_after_a_cancel_are_dropped():
    m = machine()
    turn = _start_reply(m)
    m.handle(Typed("interrupt"))
    assert m.handle(ReplyDelta(turn, "late text")) == []


# ----- barge-in -----


def test_short_speech_during_a_reply_does_not_barge_in():
    m = machine()
    _start_reply(m)
    assert m.handle(SpeechOngoing(voiced_ms=BARGE_IN_MS - 1)) == []


def test_sustained_speech_during_a_reply_barges_in_once():
    m = machine()
    _start_reply(m)
    assert m.handle(SpeechOngoing(voiced_ms=BARGE_IN_MS)) == [CancelReply()]
    assert m.handle(SpeechOngoing(voiced_ms=BARGE_IN_MS + 100)) == []


def test_speech_does_not_barge_in_while_a_confirmation_is_pending():
    m = machine()
    _start_reply(m)
    m.handle(ConfirmRequest("Open Notepad?"))
    assert m.handle(SpeechOngoing(voiced_ms=BARGE_IN_MS * 2)) == []


def test_speech_with_no_reply_running_does_not_cancel_anything():
    m = machine()
    assert m.handle(SpeechOngoing(voiced_ms=BARGE_IN_MS * 2)) == []


# ----- transcription pipeline -----


def test_utterance_is_transcribed_and_transcript_becomes_a_reply():
    m = machine()
    audio = _audio()
    actions = m.handle(Utterance(audio=audio, seconds=0.5))
    assert _kinds(actions) == [Transcribe]
    assert actions[0].audio is audio
    assert m.state is State.TRANSCRIBING

    actions = m.handle(Transcript(text="open notes"))
    assert _kinds(actions) == [Print, StartReply]
    assert actions[0].text == 'Heard: "open notes"'


def test_low_confidence_transcript_is_marked_but_still_used():
    m = machine()
    m.handle(Utterance(audio=_audio(), seconds=0.5))
    actions = m.handle(Transcript(text="mumble", low_confidence=True))
    assert actions[0].text == 'Heard (low confidence): "mumble"'
    assert StartReply in _kinds(actions)


def test_utterance_during_transcription_is_queued_then_transcribed():
    m = machine()
    first, second = _audio(), _audio() + 1
    m.handle(Utterance(audio=first, seconds=0.5))
    assert m.handle(Utterance(audio=second, seconds=0.5)) == []

    actions = m.handle(Transcript(text="first one"))
    queued = [action for action in actions if isinstance(action, Transcribe)]
    assert len(queued) == 1
    assert queued[0].audio is second


def test_empty_transcript_says_nothing_was_understood():
    m = machine()
    m.handle(Utterance(audio=_audio(), seconds=0.5))
    actions = m.handle(Transcript(text="  "))
    assert _kinds(actions) == [Print]
    assert "Could not make out" in actions[0].text


def test_transcription_error_is_printed():
    m = machine()
    m.handle(Utterance(audio=_audio(), seconds=0.5))
    actions = m.handle(Transcript(error="model crashed"))
    assert _kinds(actions) == [Print]
    assert "model crashed" in actions[0].text


def test_long_speech_is_announced_when_cut():
    m = machine()
    actions = m.handle(Utterance(audio=_audio(), seconds=30.0, cut=True))
    assert Print in _kinds(actions)
    assert Transcribe in _kinds(actions)


def test_noise_clears_the_speaking_flag():
    m = machine()
    m.handle(SpeechOngoing(voiced_ms=100))
    assert m.state is State.RECORDING
    m.handle(Noise(voiced_ms=50))
    assert m.state is State.LISTENING


# ----- commands and quitting -----


def test_exit_commands_quit():
    for text in ("/exit", "/quit"):
        assert _kinds(machine().handle(Command(text))) == [StopMic, Shutdown]


def test_quit_during_a_reply_cancels_it_first():
    m = machine()
    _start_reply(m)
    assert _kinds(m.handle(Quit())) == [CancelReply, StopMic, Shutdown]


def test_reset_applies_now_when_idle():
    assert machine().handle(Command("/reset")) == [ResetHistory()]


def test_reset_during_a_reply_is_deferred_until_the_reply_ends():
    m = machine()
    turn = _start_reply(m)
    assert m.handle(Command("/reset")) == [CancelReply()]
    actions = m.handle(ReplyDone(turn, _outcome("partial", interrupted=True)))
    assert _kinds(actions) == [ApplyOutcome, ResetHistory]


def test_other_slash_commands_are_run_by_the_loop():
    assert machine().handle(Command("/tools")) == [RunCommand("/tools")]


# ----- confirmations -----


def test_confirmation_yes_and_no_are_passed_to_the_tool():
    m = machine()
    m.handle(ConfirmRequest("Delete notes.md?"))
    assert m.awaiting_confirm
    assert m.handle(Typed("yes")) == [ResolveConfirmation(True)]
    assert not m.awaiting_confirm

    m.handle(ConfirmRequest("Delete notes.md?"))
    assert m.handle(Typed("n")) == [ResolveConfirmation(False)]


def test_typed_confirmation_must_be_a_plain_yes_or_no():
    m = machine()
    m.handle(ConfirmRequest("Delete notes.md?"))
    assert _kinds(m.handle(Typed("yes please"))) == [Print]
    assert m.awaiting_confirm


def test_unclear_confirmation_answer_asks_again():
    m = machine()
    m.handle(ConfirmRequest("Delete notes.md?"))
    actions = m.handle(Typed("maybe later"))
    assert _kinds(actions) == [Print]
    assert m.awaiting_confirm


def test_spoken_confirmation_is_answered_by_voice():
    m = machine()
    m.handle(ConfirmRequest("Delete notes.md?"))
    actions = m.handle(Transcript(text="No."))
    assert ResolveConfirmation(False) in actions


def test_confirmation_timeout_is_reported():
    m = machine()
    m.handle(ConfirmRequest("Delete notes.md?"))
    actions = m.handle(ConfirmTimedOut())
    assert _kinds(actions) == [Print]
    assert not m.awaiting_confirm


# ----- microphone failures -----


def test_microphone_failure_pauses_and_keeps_typing_working():
    m = machine()
    actions = m.handle(MicFailed("device unplugged"))
    assert _kinds(actions) == [StopMic, Print]
    assert m.state is State.PAUSED
    assert "device unplugged" in actions[1].text
    assert m.handle(Typed("hi")) == [StartReply(1, "hi")]


def test_toggle_retries_the_microphone_after_a_failure():
    m = machine()
    m.handle(MicFailed("device unplugged"))
    assert _kinds(m.handle(Toggle())) == [StartMic, Print]


# ----- yes / no parsing -----


def test_parse_yes_no():
    assert parse_yes_no("yes") is True
    assert parse_yes_no("Yes.") is True
    assert parse_yes_no("y") is True
    assert parse_yes_no("nope") is False
    assert parse_yes_no("No.") is False
    assert parse_yes_no("n") is False
    # Strict: anything beyond the exact words is not an answer.
    assert parse_yes_no("Yes, please.") is None
    assert parse_yes_no("no way") is None
    assert parse_yes_no("Don't do it") is None
    assert parse_yes_no("ok") is None
    assert parse_yes_no("sure") is None
    assert parse_yes_no("maybe") is None
    assert parse_yes_no("not sure") is None


# ----- Ctrl+C window -----


class FakeClock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_second_ctrl_c_within_two_seconds_quits_during_a_confirmation():
    clock = FakeClock()
    m = ListenMachine(barge_in_ms=BARGE_IN_MS, clock=clock)
    m.handle(ConfirmRequest("Delete?"))
    assert ResolveConfirmation(False) in m.handle(Cancel())

    m.handle(ConfirmRequest("Delete again?"))
    clock.now += 1.5
    assert Shutdown in _kinds(m.handle(Cancel()))


def test_ctrl_c_after_the_window_declines_again_instead_of_quitting():
    clock = FakeClock()
    m = ListenMachine(barge_in_ms=BARGE_IN_MS, clock=clock)
    m.handle(ConfirmRequest("Delete?"))
    m.handle(Cancel())

    m.handle(ConfirmRequest("Delete again?"))
    clock.now += 2.5
    actions = m.handle(Cancel())
    assert ResolveConfirmation(False) in actions
    assert Shutdown not in _kinds(actions)


# ----- spoken confirmations -----


def test_clear_spoken_yes_is_accepted():
    m = machine()
    m.handle(ConfirmRequest("Delete notes.md?"))
    assert ResolveConfirmation(True) in m.handle(Transcript(text="Yes."))


def test_ambiguous_spoken_answer_declines_rather_than_guessing():
    m = machine()
    m.handle(ConfirmRequest("Delete notes.md?"))
    actions = m.handle(Transcript(text="ok, sure"))
    assert ResolveConfirmation(False) in actions
    assert ResolveConfirmation(True) not in actions
    assert not m.awaiting_confirm


def test_low_confidence_spoken_yes_declines():
    m = machine()
    m.handle(ConfirmRequest("Delete notes.md?"))
    actions = m.handle(Transcript(text="yes", low_confidence=True))
    assert ResolveConfirmation(False) in actions
    assert ResolveConfirmation(True) not in actions


# ----- speech output -----


def test_barge_in_while_speech_plays_stops_speech():
    m = ListenMachine(barge_in_ms=BARGE_IN_MS, speech_busy=lambda: True)
    assert m.handle(SpeechOngoing(voiced_ms=BARGE_IN_MS)) == [StopSpeech()]


def test_barge_in_during_reply_and_speech_stops_both():
    m = ListenMachine(barge_in_ms=BARGE_IN_MS, speech_busy=lambda: True)
    _start_reply(m)
    assert m.handle(SpeechOngoing(voiced_ms=BARGE_IN_MS)) == [StopSpeech(), CancelReply()]


def test_pausing_while_speech_plays_stops_it():
    m = ListenMachine(barge_in_ms=BARGE_IN_MS, speech_busy=lambda: True)
    assert StopSpeech() in m.handle(Toggle())


def test_pausing_while_silent_does_not_send_stop_speech():
    assert StopSpeech() not in machine().handle(Toggle())


def test_notice_from_another_thread_is_printed():
    actions = machine().handle(Notice("Voice output failed"))
    assert _kinds(actions) == [Print]
    assert "Voice output failed" in actions[0].text
