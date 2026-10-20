"""Tests for the continuous listening pipeline, using a fake frame source."""

import queue
import threading
import time

import numpy as np

from companion.audio.pipeline import ListeningPipeline, Rechunker
from companion.audio.segmenter import Segmenter, SpeechOngoing, Utterance


class FakeSource:
    """Hands out queued frames. Returns None when the queue is empty."""

    def __init__(self, frames, fail_start: Exception | None = None) -> None:
        self._frames = queue.Queue()
        for frame in frames:
            self._frames.put(frame)
        self._fail_start = fail_start
        self.started = False
        self.stopped = False

    def start(self) -> None:
        if self._fail_start is not None:
            raise self._fail_start
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def read(self, timeout: float):
        try:
            return self._frames.get(timeout=min(timeout, 0.01))
        except queue.Empty:
            return None


def _frame(value: float) -> np.ndarray:
    # The fake detector reads the probability from the first sample.
    frame = np.zeros(512, dtype=np.float32)
    frame[0] = value
    return frame


def _detect(frame: np.ndarray) -> float:
    return float(frame[0])


def _segmenter() -> Segmenter:
    return Segmenter(
        vad_threshold=0.5,
        pre_roll_ms=0,
        min_speech_ms=100,
        end_silence_ms=200,
        max_utterance_s=10.0,
    )


def _wait_for(events: queue.Queue, kind, timeout: float = 3.0):
    deadline = time.monotonic() + timeout
    seen = []
    while time.monotonic() < deadline:
        try:
            event = events.get(timeout=0.05)
        except queue.Empty:
            continue
        seen.append(event)
        if isinstance(event, kind):
            return event, seen
    raise AssertionError(f"no {kind.__name__} within {timeout}s; saw {seen}")


def test_rechunker_emits_fixed_size_frames_and_keeps_the_remainder():
    rechunker = Rechunker(size=4)
    first = rechunker.feed(np.arange(6, dtype=np.float32))
    assert len(first) == 1
    assert first[0].tolist() == [0.0, 1.0, 2.0, 3.0]

    second = rechunker.feed(np.arange(6, 8, dtype=np.float32))
    assert [frame.tolist() for frame in second] == [[4.0, 5.0, 6.0, 7.0]]


def test_rechunker_reset_drops_the_remainder():
    rechunker = Rechunker(size=4)
    rechunker.feed(np.arange(3, dtype=np.float32))
    rechunker.reset()
    assert rechunker.feed(np.arange(4, dtype=np.float32))[0].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_pipeline_turns_speech_frames_into_an_utterance():
    frames = [_frame(0.9)] * 6 + [_frame(0.0)] * 10
    source = FakeSource(frames)
    events: queue.Queue = queue.Queue()
    pipeline = ListeningPipeline(
        source=source,
        detect=_detect,
        segmenter=_segmenter(),
        on_event=events.put,
        on_error=lambda message: events.put(("error", message)),
        poll_s=0.01,
    )

    pipeline.start()
    try:
        utterance, seen = _wait_for(events, Utterance)
    finally:
        pipeline.stop()

    assert utterance.seconds > 0.1
    assert any(isinstance(event, SpeechOngoing) for event in seen)
    assert source.started
    assert source.stopped
    assert not pipeline.running


def test_pipeline_reports_a_source_that_fails_to_start():
    from companion.audio.capture import SpeechError

    pipeline = ListeningPipeline(
        source=FakeSource([], fail_start=SpeechError("device busy")),
        detect=_detect,
        segmenter=_segmenter(),
        on_event=lambda event: None,
        on_error=lambda message: None,
    )
    try:
        pipeline.start()
    except SpeechError as exc:
        assert "device busy" in str(exc)
    else:
        raise AssertionError("start() should raise when the source fails")
    assert not pipeline.running


def test_pipeline_reports_an_exception_in_the_listen_thread():
    errors: list[str] = []
    done = threading.Event()

    def bad_detect(frame):
        raise RuntimeError("detector exploded")

    def on_error(message: str) -> None:
        errors.append(message)
        done.set()

    pipeline = ListeningPipeline(
        source=FakeSource([_frame(0.9)]),
        detect=bad_detect,
        segmenter=_segmenter(),
        on_event=lambda event: None,
        on_error=on_error,
        poll_s=0.01,
    )
    pipeline.start()
    try:
        assert done.wait(timeout=3.0)
    finally:
        pipeline.stop()
    assert "detector exploded" in errors[0]
