"""Tests for the follow-up window, wake-word barge-in, and the headphones hold."""

from __future__ import annotations

import queue
import time

import numpy as np
import pytest

from companion.audio.pipeline import ListeningPipeline
from companion.audio.segmenter import Segmenter
from companion.audio.wakeword import WakeGate
from companion.config import ConfigError, _wake_word_config
from companion.voice.keys import LineEditor
from companion.voice.loop import VoiceLoop
from companion.voice.machine import CancelReply, ListenMachine, Quit, StopSpeech, WakeHeard

# 512 samples at 16 kHz is 0.032 s per frame.
FRAME = np.zeros(512, dtype=np.float32)


class FakeWake:
    """Reports a hit on the frames whose 1-based numbers are in hits."""

    def __init__(self, hits=()) -> None:
        self._hits = set(hits)
        self._count = 0

    def hit(self, _frame) -> bool:
        self._count += 1
        return self._count in self._hits

    def reset(self) -> None:
        pass


def _gate(
    *,
    hits=(),
    detect_value: float = 0.9,
    window_s: float = 0.2,
    hold=None,
    speech_level=None,
    on_wake=None,
):
    seen: list[int] = []

    def detect(_frame) -> float:
        seen.append(1)
        return detect_value

    gate = WakeGate(
        detect,
        FakeWake(hits),
        window_s=window_s,
        hold=hold,
        speech_level=speech_level,
        on_wake=on_wake,
    )
    return gate, seen


# ----- WakeGate: follow-up window, hold, speech extension, on_wake -----


def test_open_for_lets_speech_through_without_the_wake_word():
    gate, seen = _gate(detect_value=0.9)
    gate.open_for(0.1)
    # Frames end at 0.032, 0.064, 0.096 s (open) and 0.128 s (closed).
    assert [gate(FRAME) for _ in range(4)] == [0.9, 0.9, 0.9, 0.0]
    assert len(seen) == 3


def test_open_for_zero_or_less_does_nothing():
    gate, _seen = _gate()
    gate.open_for(0.0)
    gate.open_for(-1.0)
    assert [gate(FRAME) for _ in range(3)] == [0.0, 0.0, 0.0]
    assert not gate.open


def test_hold_keeps_the_gate_open_after_the_window_ends():
    holding = {"on": True}
    gate, seen = _gate(hold=lambda: holding["on"])
    assert [gate(FRAME) for _ in range(10)] == [0.9] * 10
    assert len(seen) == 10


def test_gate_closes_once_hold_is_released_and_the_window_has_passed():
    holding = {"on": True}
    gate, _seen = _gate(hold=lambda: holding["on"])
    for _ in range(3):
        gate(FRAME)
    holding["on"] = False
    assert gate(FRAME) == 0.0
    assert not gate.open


def test_speech_extends_an_open_window_so_a_sentence_is_not_cut_off():
    # Speech at 0.9 on every frame, speech_level 0.5: the window keeps moving forward.
    gate, _seen = _gate(hits={1}, detect_value=0.9, speech_level=0.5)
    outputs = [gate(FRAME) for _ in range(30)]
    assert outputs == [0.9] * 30


def test_quiet_frames_do_not_extend_the_window():
    gate, _seen = _gate(hits={1}, detect_value=0.1, speech_level=0.5)
    outputs = [gate(FRAME) for _ in range(10)]
    # Window is 0.2 s after the hit, so frames 1 to 7 are open, as before.
    assert outputs[:7] == [0.1] * 7
    assert outputs[7:] == [0.0] * 3


def test_speech_does_not_reopen_a_closed_window():
    gate, seen = _gate(hits=set(), detect_value=0.9, speech_level=0.5)
    assert [gate(FRAME) for _ in range(5)] == [0.0] * 5
    assert seen == []


def test_on_wake_fires_when_the_wake_word_is_heard_while_listening():
    woken: list[int] = []
    gate, _seen = _gate(hits={1}, on_wake=lambda: woken.append(1))
    gate(FRAME)
    assert woken == [1]


def test_muted_frames_still_feed_the_wake_word_and_fire_on_wake():
    woken: list[int] = []
    gate, seen = _gate(hits={2}, on_wake=lambda: woken.append(1))
    gate.muted(FRAME)
    assert woken == []
    gate.muted(FRAME)
    assert woken == [1]
    # The detector is never asked about muted frames, but the window is open afterwards.
    assert gate(FRAME) == 0.9
    assert seen == [1]


def test_open_for_then_hit_keeps_the_later_end_time():
    gate, _seen = _gate(hits={1}, window_s=0.2)
    gate.open_for(1.0)
    outputs = [gate(FRAME) for _ in range(20)]
    assert outputs == [0.9] * 20


# ----- ListeningPipeline: muted frames -----


class FakeSource:
    def __init__(self, frames) -> None:
        self._frames: queue.Queue = queue.Queue()
        for frame in frames:
            self._frames.put(frame)

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def read(self, timeout: float):
        try:
            return self._frames.get(timeout=min(timeout, 0.01))
        except queue.Empty:
            return None


def _segmenter() -> Segmenter:
    return Segmenter(
        vad_threshold=0.5,
        pre_roll_ms=0,
        min_speech_ms=100,
        end_silence_ms=200,
        max_utterance_s=10.0,
    )


def test_gated_frames_go_to_muted_frame_and_skip_detection_and_segmenter():
    source = FakeSource([FRAME, FRAME, FRAME])
    muted: list[np.ndarray] = []
    detected: list[int] = []
    events: queue.Queue = queue.Queue()

    def detect(_frame) -> float:
        detected.append(1)
        return 0.9

    pipeline = ListeningPipeline(
        source=source,
        detect=detect,
        segmenter=_segmenter(),
        on_event=events.put,
        on_error=lambda message: events.put(("error", message)),
        poll_s=0.01,
        gate=lambda: True,
        muted_frame=muted.append,
    )

    pipeline.start()
    try:
        deadline = time.monotonic() + 3.0
        while len(muted) < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        pipeline.stop()

    assert len(muted) == 3
    assert detected == []
    assert events.empty()


# ----- ListenMachine: wake-word barge-in -----


def test_wake_heard_cancels_a_running_reply():
    machine = ListenMachine(barge_in_ms=400, speech_busy=lambda: False)
    machine._reply_turn = 1
    assert machine.handle(WakeHeard()) == [CancelReply()]


def test_wake_heard_stops_speech_that_is_playing():
    machine = ListenMachine(barge_in_ms=400, speech_busy=lambda: True)
    assert machine.handle(WakeHeard()) == [StopSpeech()]


def test_wake_heard_cancels_the_reply_and_stops_speech_together():
    machine = ListenMachine(barge_in_ms=400, speech_busy=lambda: True)
    machine._reply_turn = 1
    assert machine.handle(WakeHeard()) == [CancelReply(), StopSpeech()]


def test_wake_heard_does_nothing_when_idle():
    machine = ListenMachine(barge_in_ms=400, speech_busy=lambda: False)
    assert machine.handle(WakeHeard()) == []


def test_wake_heard_does_not_cancel_twice():
    machine = ListenMachine(barge_in_ms=400, speech_busy=lambda: False)
    machine._reply_turn = 1
    machine._reply_cancelled = True
    assert machine.handle(WakeHeard()) == []


def test_wake_heard_is_ignored_while_paused():
    machine = ListenMachine(barge_in_ms=400, speech_busy=lambda: True)
    machine.paused = True
    assert machine.handle(WakeHeard()) == []


def test_wake_heard_works_on_speakers_where_voice_barge_in_is_off():
    machine = ListenMachine(barge_in_ms=400, speech_busy=lambda: True, vad_barge_in=False)
    assert machine.handle(WakeHeard()) == [StopSpeech()]


# ----- VoiceLoop: follow-up when a reply stops playing -----


class FakeSpeech:
    def __init__(self) -> None:
        self.busy = False

    def speaking(self, tail_s: float = 0.0) -> bool:
        return self.busy

    def close(self) -> None:
        pass


class FakeKeys:
    def poll(self, confirming: bool = False):
        return [Quit()]


class IdlePipeline:
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


def _loop(follow_up=None, speech=None) -> VoiceLoop:
    return VoiceLoop(
        machine=ListenMachine(barge_in_ms=400),
        pipeline=IdlePipeline(),
        transcribe=lambda audio: None,
        stream=lambda *args: None,
        run_command=lambda text: "",
        new_history=lambda: [],
        keys=FakeKeys(),
        editor=LineEditor(lambda text: None),
        events=queue.Queue(),
        answers=queue.Queue(),
        low_confidence_logprob=-1.0,
        write=lambda text: None,
        tick_s=0.01,
        speech=speech,
        follow_up=follow_up,
    )


def test_follow_up_opens_when_a_reply_finishes():
    calls: list[int] = []
    loop = _loop(follow_up=lambda: calls.append(1))
    loop._machine._reply_turn = 1
    loop._watch_playback()
    assert calls == []

    loop._machine._reply_turn = None
    loop._watch_playback()
    assert calls == [1]

    loop._watch_playback()
    assert calls == [1]


def test_follow_up_waits_for_speech_to_finish_after_the_reply_ends():
    calls: list[int] = []
    speech = FakeSpeech()
    loop = _loop(follow_up=lambda: calls.append(1), speech=speech)

    speech.busy = True
    loop._watch_playback()
    speech.busy = False
    loop._watch_playback()
    assert calls == [1]


def test_follow_up_does_not_open_while_paused():
    calls: list[int] = []
    loop = _loop(follow_up=lambda: calls.append(1))
    loop._machine.paused = True
    loop._machine._reply_turn = 1
    loop._watch_playback()
    loop._machine._reply_turn = None
    loop._watch_playback()
    assert calls == []


def test_no_follow_up_hook_is_fine():
    loop = _loop(follow_up=None)
    loop._machine._reply_turn = 1
    loop._watch_playback()
    loop._machine._reply_turn = None
    loop._watch_playback()


# ----- config -----


def test_follow_up_defaults_to_eight_seconds():
    assert _wake_word_config({}).follow_up_s == 8.0


def test_follow_up_zero_turns_it_off():
    assert _wake_word_config({"follow_up_s": 0}).follow_up_s == 0.0


@pytest.mark.parametrize("bad", [-1, 61])
def test_follow_up_out_of_range_is_refused(bad):
    with pytest.raises(ConfigError):
        _wake_word_config({"follow_up_s": bad})
