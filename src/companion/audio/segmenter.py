"""Turns per-frame voice probabilities into utterances. Pure logic, no audio I/O.

Hysteresis: speech starts when a frame's probability reaches the threshold, and counts as
voiced until it drops below threshold minus HYSTERESIS. This stops brief dips inside a word
from splitting an utterance.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from companion.audio.capture import SAMPLE_RATE

FRAME_SAMPLES = 512
"""Silero VAD expects 512-sample frames at 16 kHz (32 ms)."""

HYSTERESIS = 0.15


@dataclass(frozen=True, eq=False)
class SpeechOngoing:
    """Speech is in progress. voiced_ms is the total voiced time in the current candidate."""

    voiced_ms: int


@dataclass(frozen=True, eq=False)
class Utterance:
    """A finished utterance ready for transcription. cut is True when split for length."""

    audio: np.ndarray
    seconds: float
    cut: bool = False


@dataclass(frozen=True)
class Noise:
    """A sound that was too short to be speech (below min_speech_ms). Dropped."""

    voiced_ms: int


SegmenterEvent = SpeechOngoing | Utterance | Noise


class Segmenter:
    def __init__(
        self,
        *,
        vad_threshold: float,
        pre_roll_ms: int,
        min_speech_ms: int,
        end_silence_ms: int,
        max_utterance_s: float,
        sample_rate: int = SAMPLE_RATE,
        frame_samples: int = FRAME_SAMPLES,
    ) -> None:
        if not 0.0 < vad_threshold < 1.0:
            raise ValueError("vad_threshold must be between 0 and 1")
        self._on = vad_threshold
        self._off = max(0.01, vad_threshold - HYSTERESIS)
        self._sample_rate = sample_rate
        self._frame_ms = frame_samples * 1000 / sample_rate
        self._pre_roll: deque[np.ndarray] = deque(maxlen=max(0, round(pre_roll_ms / self._frame_ms)))
        self._min_speech_ms = min_speech_ms
        self._end_silence_ms = end_silence_ms
        self._max_frames = max(1, round(max_utterance_s * 1000 / self._frame_ms))
        self._active = False
        self._frames: list[np.ndarray] = []
        self._voiced_frames = 0
        self._silence_frames = 0

    @property
    def active(self) -> bool:
        return self._active

    def feed(self, frame: np.ndarray, probability: float) -> list[SegmenterEvent]:
        """Process one frame and return any events it produced."""
        if not self._active:
            if probability < self._on:
                self._pre_roll.append(frame)
                return []
            self._active = True
            self._frames = list(self._pre_roll)
            self._pre_roll.clear()
            self._voiced_frames = 0
            self._silence_frames = 0

        self._frames.append(frame)
        events: list[SegmenterEvent] = []
        if probability >= self._off:
            self._voiced_frames += 1
            self._silence_frames = 0
            events.append(SpeechOngoing(self._ms(self._voiced_frames)))
        else:
            self._silence_frames += 1

        if self._silence_frames * self._frame_ms >= self._end_silence_ms:
            events.append(self._finish(cut=False))
        elif len(self._frames) >= self._max_frames:
            events.append(self._finish(cut=True))
        return events

    def reset(self) -> None:
        """Drop any partial utterance and the pre-roll buffer."""
        self._pre_roll.clear()
        self._frames = []
        self._voiced_frames = 0
        self._silence_frames = 0
        self._active = False

    def _finish(self, cut: bool) -> SegmenterEvent:
        frames = self._frames
        voiced_ms = self._ms(self._voiced_frames)
        self.reset()
        if voiced_ms < self._min_speech_ms:
            return Noise(voiced_ms)
        audio = np.concatenate(frames).astype(np.float32, copy=False)
        return Utterance(audio=audio, seconds=audio.size / self._sample_rate, cut=cut)

    def _ms(self, frames: int) -> int:
        return int(frames * self._frame_ms)
