"""Continuous listening: microphone frames -> speech probability -> segmenter -> events.

The pipeline runs on its own thread and reports results through callbacks. It never blocks
on the UI or the LLM.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Protocol

import numpy as np

from companion.audio.segmenter import FRAME_SAMPLES, Noise, Segmenter, SegmenterEvent, Utterance
from companion.audio.vad import ProbabilityFn

logger = logging.getLogger("companion.audio")


class FrameSource(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def read(self, timeout: float) -> np.ndarray | None: ...


class Rechunker:
    """Turns chunks of any size into fixed-size frames, keeping the remainder for next time."""

    def __init__(self, size: int = FRAME_SAMPLES) -> None:
        self._size = size
        self._buffer = np.zeros(0, dtype=np.float32)

    def feed(self, chunk: np.ndarray) -> list[np.ndarray]:
        self._buffer = np.concatenate((self._buffer, np.asarray(chunk, dtype=np.float32).ravel()))
        frames: list[np.ndarray] = []
        while self._buffer.size >= self._size:
            frames.append(self._buffer[: self._size].copy())
            self._buffer = self._buffer[self._size :]
        return frames

    def reset(self) -> None:
        self._buffer = np.zeros(0, dtype=np.float32)


class ListeningPipeline:
    """Owns the listening thread. start() and stop() are called from the UI thread."""

    def __init__(
        self,
        source: FrameSource,
        detect: ProbabilityFn,
        segmenter: Segmenter,
        on_event: Callable[[SegmenterEvent], None],
        on_error: Callable[[str], None],
        reset_detector: Callable[[], None] | None = None,
        poll_s: float = 0.1,
        gate: Callable[[], bool] | None = None,
        muted_frame: Callable[[np.ndarray], None] | None = None,
    ) -> None:
        self._source = source
        self._detect = detect
        self._segmenter = segmenter
        self._on_event = on_event
        self._on_error = on_error
        self._reset_detector = reset_detector or (lambda: None)
        self._poll_s = poll_s
        self._gate = gate
        self._muted_frame = muted_frame
        self._rechunker = Rechunker()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def running(self) -> bool:
        return self._thread is not None

    def start(self) -> None:
        """Open the microphone and start the thread. Raises SpeechError if the mic cannot open."""
        if self.running:
            return
        self._segmenter.reset()
        self._rechunker.reset()
        self._reset_detector()
        self._stop = threading.Event()
        self._source.start()
        self._thread = threading.Thread(target=self._run, args=(self._stop,), name="listen", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the thread, close the microphone, and drop any partial utterance."""
        thread, self._thread = self._thread, None
        if thread is None:
            return
        self._stop.set()
        thread.join(timeout=2.0)
        if thread.is_alive():
            logger.warning("listening thread did not stop within 2 seconds")
        self._source.stop()
        self._segmenter.reset()
        self._reset_detector()

    def _run(self, stop: threading.Event) -> None:
        gated = False
        try:
            while not stop.is_set():
                chunk = self._source.read(self._poll_s)
                if chunk is None:
                    continue
                if self._gate is not None and self._gate():
                    # Keep draining the mic, but ignore it. Drop any partial utterance once.
                    if not gated:
                        gated = True
                        self._segmenter.reset()
                        self._rechunker.reset()
                        self._reset_detector()
                    if self._muted_frame is not None:
                        for frame in self._rechunker.feed(chunk):
                            self._muted_frame(frame)
                    continue
                gated = False
                for frame in self._rechunker.feed(chunk):
                    if stop.is_set():
                        return
                    events = self._segmenter.feed(frame, self._detect(frame))
                    for event in events:
                        self._on_event(event)
                    if any(isinstance(event, (Utterance, Noise)) for event in events):
                        self._reset_detector()
        except Exception as exc:
            logger.exception("listening thread failed")
            if not stop.is_set():
                self._on_error(str(exc) or type(exc).__name__)
