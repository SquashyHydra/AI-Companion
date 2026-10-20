"""Tests for speech wiring in the voice loop and the speakers-mode microphone gate."""

from __future__ import annotations

import queue
import threading

import numpy as np

from companion.audio.segmenter import FRAME_SAMPLES
from companion.audio.pipeline import ListeningPipeline
from companion.llm.client import INTERRUPTED_MARK
from companion.voice.keys import LineEditor
from companion.voice.loop import VoiceLoop
from companion.voice.machine import ListenMachine, Quit, StopSpeech


class FakePipeline:
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


class FakeKeys:
    """Returns one scripted batch of keys per poll. Quits once the script runs out."""

    def __init__(self, batches) -> None:
        self._batches = list(batches)

    def poll(self, confirming: bool = False):
        if self._batches:
            return self._batches.pop(0)
        return [Quit()]


class FakeSpeech:
    def __init__(self, spoken: str = "First sentence.", busy: bool = True) -> None:
        self.spoken = spoken
        self.busy = busy
        self.interrupts = 0
        self.closed = 0
        self.fed: list[str] = []
        self.finished = 0

    def begin_reply(self) -> None:
        pass

    def feed(self, text: str) -> None:
        self.fed.append(text)

    def finish(self) -> None:
        self.finished += 1

    def interrupt(self) -> tuple[str, bool]:
        self.interrupts += 1
        return self.spoken, self.busy

    def speaking(self, tail_s: float = 0.0) -> bool:
        return False

    def close(self) -> None:
        self.closed += 1


def make_loop(speech: FakeSpeech | None, keys: FakeKeys | None = None) -> VoiceLoop:
    return VoiceLoop(
        machine=ListenMachine(barge_in_ms=400),
        pipeline=FakePipeline(),
        transcribe=lambda audio: None,
        stream=lambda *args: None,
        run_command=lambda text: "",
        new_history=lambda: [],
        keys=keys or FakeKeys([[]]),
        editor=LineEditor(lambda text: None),
        events=queue.Queue(),
        answers=queue.Queue(),
        low_confidence_logprob=-1.0,
        write=lambda text: None,
        tick_s=0.01,
        speech=speech,
    )


def test_stopping_speech_keeps_only_the_spoken_sentences_in_history():
    speech = FakeSpeech(spoken="First sentence.", busy=True)
    loop = make_loop(speech)
    loop._history = [
        {"role": "user", "content": "tell me a story"},
        {"role": "assistant", "content": "First sentence. Second sentence. Third sentence."},
    ]
    loop._reply_in_history = True

    loop._perform(StopSpeech())

    assert loop._history[-1]["content"] == f"First sentence. {INTERRUPTED_MARK}"
    assert loop._history[0]["content"] == "tell me a story"


def test_stop_speech_runs_once_per_reply():
    speech = FakeSpeech()
    loop = make_loop(speech)
    loop._reply_in_history = True
    loop._history = [{"role": "assistant", "content": "Hi there. More."}]

    loop._perform(StopSpeech())
    loop._perform(StopSpeech())

    assert speech.interrupts == 1


def test_stopping_when_nothing_was_playing_leaves_history_alone():
    speech = FakeSpeech(spoken="", busy=False)
    loop = make_loop(speech)
    loop._reply_in_history = True
    loop._history = [{"role": "assistant", "content": "Full reply here."}]

    loop._perform(StopSpeech())

    assert loop._history[-1]["content"] == "Full reply here."


def test_run_closes_speech_on_shutdown():
    speech = FakeSpeech()
    loop = make_loop(speech)
    assert loop.run() == 0
    assert speech.closed == 1


def test_no_speech_object_means_text_mode_is_silent_and_still_works():
    loop = make_loop(None)
    loop._perform(StopSpeech())  # must not raise
    assert loop.run() == 0


# ----- speakers-mode microphone gate -----


class FakeSource:
    """Serves a fixed list of chunks. Sets `stop` once the list runs out."""

    def __init__(self, chunks: list[np.ndarray], stop: threading.Event) -> None:
        self._chunks = list(chunks)
        self._stop = stop

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def read(self, timeout: float):
        if not self._chunks:
            self._stop.set()
            return None
        return self._chunks.pop(0)


class FakeSegmenter:
    def __init__(self) -> None:
        self.feeds = 0
        self.resets = 0

    def feed(self, frame, probability):
        self.feeds += 1
        return []

    def reset(self) -> None:
        self.resets += 1


def test_the_gate_drops_mic_audio_while_speech_plays_and_resets_once():
    stop = threading.Event()
    chunks = [np.zeros(FRAME_SAMPLES, dtype=np.float32) for _ in range(4)]
    gate_values = iter([False, True, True, False])
    segmenter = FakeSegmenter()
    detector_resets: list[int] = []
    detects: list[int] = []

    pipeline = ListeningPipeline(
        source=FakeSource(chunks, stop),
        detect=lambda frame: detects.append(1) or 0.0,
        segmenter=segmenter,
        on_event=lambda event: None,
        on_error=lambda message: None,
        reset_detector=lambda: detector_resets.append(1),
        poll_s=0.01,
        gate=lambda: next(gate_values),
    )

    pipeline._run(stop)

    assert len(detects) == 2  # the first and last chunks were heard; the two gated ones were not
    assert segmenter.feeds == 2
    assert len(detector_resets) == 1  # reset once when the gate closes, not on every gated chunk
    assert segmenter.resets == 1


def test_without_a_gate_every_chunk_is_heard():
    stop = threading.Event()
    chunks = [np.zeros(FRAME_SAMPLES, dtype=np.float32) for _ in range(3)]
    segmenter = FakeSegmenter()
    pipeline = ListeningPipeline(
        source=FakeSource(chunks, stop),
        detect=lambda frame: 0.0,
        segmenter=segmenter,
        on_event=lambda event: None,
        on_error=lambda message: None,
        poll_s=0.01,
    )
    pipeline._run(stop)
    assert segmenter.feeds == 3
