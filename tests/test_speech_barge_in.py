"""Barge-in: interrupting speech cancels the engine's queued work by generation number."""

from __future__ import annotations

import threading
import time

import numpy as np

from companion.audio.errors import SpeechError
from companion.voice.speech import SpeechOut


class CancellableSynth:
    sample_rate = 10
    tags: frozenset[str] = frozenset()

    def __init__(self, *, gate: threading.Event | None = None, cancel_error: str | None = None):
        self.cancelled: list[int] = []
        self.calls: list[str] = []
        self._gate = gate
        self._cancel_error = cancel_error

    def synthesize(self, text: str, *, generation: int = 0) -> np.ndarray:
        self.calls.append(text)
        if self._gate is not None:
            self._gate.wait(2.0)
        return np.zeros(len(text), dtype=np.float32)

    def cancel(self, generation: int) -> None:
        self.cancelled.append(generation)
        if self._cancel_error is not None:
            raise SpeechError(self._cancel_error)


class QuietPlayer:
    def __init__(self) -> None:
        self.writes: list[object] = []

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def write(self, samples: np.ndarray, mark: object = None) -> None:
        self.writes.append(mark)

    def flush(self) -> list[object]:
        return []

    def busy(self, tail_s: float = 0.0) -> bool:
        return False

    def take_completed(self) -> list[object]:
        return []


def wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def test_each_interrupt_tells_the_engine_the_new_generation():
    synth = CancellableSynth()
    speech = SpeechOut(synth, QuietPlayer())
    speech.start()

    speech.interrupt()
    speech.interrupt()

    assert synth.cancelled == [1, 2]
    speech.close()


def test_interrupt_during_synthesis_cancels_from_the_next_generation():
    gate = threading.Event()
    synth = CancellableSynth(gate=gate)
    speech = SpeechOut(synth, QuietPlayer())
    speech.start()
    speech.begin_reply()
    speech.feed("Old reply here. ")
    assert wait_until(lambda: synth.calls == ["Old reply here."])

    speech.interrupt()
    gate.set()
    speech.close()

    # The engine is told to drop every request from generation 1 onward; the in-flight job was 0.
    assert synth.cancelled == [1, 2]


def test_a_failed_cancel_is_logged_and_does_not_break_the_interrupt():
    synth = CancellableSynth(cancel_error="server unreachable")
    speech = SpeechOut(synth, QuietPlayer())
    speech.start()

    spoken, was_busy = speech.interrupt()

    assert (spoken, was_busy) == ("", False)
    assert synth.cancelled == [1]
    speech.close()


def test_an_engine_without_cancel_is_interrupted_normally():
    class NoCancelSynth:
        sample_rate = 10

        def synthesize(self, text: str, *, generation: int = 0) -> np.ndarray:
            return np.zeros(1, dtype=np.float32)

    speech = SpeechOut(NoCancelSynth(), QuietPlayer())
    speech.start()
    assert speech.interrupt() == ("", False)
    speech.close()
