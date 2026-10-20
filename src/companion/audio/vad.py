"""Silero voice activity detection. Trims a recording down to its speech span."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import Any

import numpy as np

from companion.audio.capture import SAMPLE_RATE

SpeechSpan = tuple[int, int]
"""(start, end) sample indices of one detected speech segment."""

DetectFn = Callable[[np.ndarray], list[SpeechSpan]]


class SpeechTrimmer:
    """Finds speech in a 16 kHz mono recording and returns only that part.

    The detector is injectable so tests can run without the Silero model.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        sample_rate: int = SAMPLE_RATE,
        detect: DetectFn | None = None,
    ) -> None:
        self._threshold = threshold
        self._sample_rate = sample_rate
        self._detect = detect
        self._model: object | None = None

    def load(self) -> None:
        """Load the Silero model now so the first utterance is not delayed."""
        if self._detect is None:
            self._silero_model()

    def speech_spans(self, audio: np.ndarray) -> list[SpeechSpan]:
        if audio.size == 0:
            return []
        if self._detect is not None:
            return self._detect(audio)
        return self._silero_spans(audio)

    def trim(self, audio: np.ndarray) -> np.ndarray:
        """Return audio from the first speech sample to the last. Empty if there is no speech."""
        spans = self.speech_spans(audio)
        if not spans:
            return np.zeros(0, dtype=np.float32)
        start = min(span_start for span_start, _ in spans)
        end = max(span_end for _, span_end in spans)
        return audio[start:end]

    def _silero_model(self) -> object:
        if self._model is None:
            from silero_vad import load_silero_vad

            self._model = load_silero_vad()
        return self._model

    def _silero_spans(self, audio: np.ndarray) -> list[SpeechSpan]:
        import torch
        from silero_vad import get_speech_timestamps

        stamps = get_speech_timestamps(
            torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)),
            self._silero_model(),
            threshold=self._threshold,
            sampling_rate=self._sample_rate,
            return_seconds=False,
        )
        return [(int(stamp["start"]), int(stamp["end"])) for stamp in stamps]


ProbabilityFn = Callable[[np.ndarray], float]
"""One 512-sample frame in, speech probability (0 to 1) out."""


@lru_cache(maxsize=1)
def load_silero_model() -> Any:
    """Load the Silero model once and share it between the trimmer and the frame detector."""
    from silero_vad import load_silero_vad

    return load_silero_vad()


class SileroFrameDetector:
    """Gives a speech probability for each frame. Keeps Silero's recurrent state between frames.

    Call reset() at utterance boundaries so one utterance does not affect the next.
    The probability function is injectable so tests can run without the model.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, probability: ProbabilityFn | None = None) -> None:
        self._sample_rate = sample_rate
        self._probability = probability

    def load(self) -> None:
        """Load the model now so the first frame is not delayed."""
        if self._probability is None:
            load_silero_model()

    def __call__(self, frame: np.ndarray) -> float:
        if self._probability is not None:
            return float(self._probability(frame))
        import torch

        tensor = torch.from_numpy(np.ascontiguousarray(frame, dtype=np.float32))
        with torch.no_grad():
            return float(load_silero_model()(tensor, self._sample_rate).item())

    def reset(self) -> None:
        if self._probability is not None:
            return
        reset_states = getattr(load_silero_model(), "reset_states", None)
        if callable(reset_states):
            reset_states()
