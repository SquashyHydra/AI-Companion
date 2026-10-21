"""Optional wake word using openWakeWord.

Speech is only passed on to the segmenter for a short window after the wake word is heard.
The wake word model sees every frame. The speech detector (VAD) is only consulted inside the window,
so speech outside it is ignored.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from typing import Any

import numpy as np

from companion.audio.capture import SAMPLE_RATE
from companion.audio.errors import SpeechError

FrameDetect = Callable[[np.ndarray], float]


def _load_openwakeword_model(name: str) -> Any:
    """Load a pretrained openWakeWord model. Downloads it once from GitHub if it is missing."""
    try:
        from openwakeword import FEATURE_MODELS, MODELS
        from openwakeword.model import Model
        from openwakeword.utils import download_models
    except ImportError as exc:
        raise SpeechError("the wake word needs openwakeword: pip install openwakeword") from exc

    if name not in MODELS:
        raise SpeechError(f"unknown wake word '{name}'. Choose one of: {', '.join(sorted(MODELS))}")

    needed = [MODELS[name]["model_path"], *(spec["model_path"] for spec in FEATURE_MODELS.values())]
    if not all(os.path.exists(path.replace(".tflite", ".onnx")) for path in needed):
        try:
            download_models([name])
        except Exception as exc:
            raise SpeechError(
                f"could not download the '{name}' wake word model (internet is needed once): {exc}"
            ) from exc
    try:
        return Model(wakeword_models=[name], inference_framework="onnx")
    except Exception as exc:
        raise SpeechError(f"could not load the '{name}' wake word model: {exc}") from exc


class WakeWordDetector:
    """Scores 16 kHz float frames with openWakeWord. hit() is True when any score reaches the threshold."""

    def __init__(self, model_name: str, threshold: float, *, model: Any | None = None) -> None:
        self._model_name = model_name
        self._threshold = threshold
        self._model = model

    def load(self) -> None:
        if self._model is None:
            self._model = _load_openwakeword_model(self._model_name)

    def scores(self, frame: np.ndarray) -> dict[str, float]:
        """Score one frame for each loaded wake word model, from 0 to 1."""
        if self._model is None:
            raise SpeechError("the wake word model is not loaded")
        pcm = np.clip(np.asarray(frame, dtype=np.float32).ravel() * 32767.0, -32768, 32767)
        raw = self._model.predict(pcm.astype(np.int16))
        return {str(name): float(score) for name, score in raw.items()}

    def hit(self, frame: np.ndarray) -> bool:
        return any(score >= self._threshold for score in self.scores(frame).values())

    def reset(self) -> None:
        if self._model is not None:
            self._model.reset()


class WakeGate:
    """Same signature as the speech detector. Returns 0.0 unless the wake word was heard in the last window_s seconds.

    Time is counted in audio samples, so the window does not depend on how fast frames arrive.
    hold() keeps the gate open while it returns True. speech_level extends an open window while
    speech is heard, so a sentence is not cut off at the end of the window.
    """

    def __init__(
        self,
        detect: FrameDetect,
        wake: WakeWordDetector,
        *,
        window_s: float,
        reset_detect: Callable[[], None] | None = None,
        hold: Callable[[], bool] | None = None,
        speech_level: float | None = None,
        on_wake: Callable[[], None] | None = None,
    ) -> None:
        if window_s <= 0:
            raise ValueError("window_s must be positive")
        self._detect = detect
        self._wake = wake
        self._window_s = window_s
        self._reset_detect = reset_detect
        self._hold = hold
        self._speech_level = speech_level
        self._on_wake = on_wake
        self._lock = threading.Lock()
        self._audio_s = 0.0
        self._open_until = 0.0

    @property
    def open(self) -> bool:
        with self._lock:
            return self._is_open()

    def _is_open(self) -> bool:
        if self._hold is not None and self._hold():
            return True
        return self._audio_s < self._open_until

    def _advance(self, frame: np.ndarray) -> bool:
        """Move the audio clock on by one frame. Returns True when the wake word was heard in it."""
        self._audio_s += len(frame) / SAMPLE_RATE
        if self._wake.hit(frame):
            # Keep the later end time, so a longer open_for window is not shortened by a hit.
            self._open_until = max(self._open_until, self._audio_s + self._window_s)
            if self._on_wake is not None:
                self._on_wake()
            return True
        return False

    def __call__(self, frame: np.ndarray) -> float:
        with self._lock:
            self._advance(frame)
            if not self._is_open():
                return 0.0
            probability = self._detect(frame)
            if (
                self._speech_level is not None
                and probability >= self._speech_level
                and self._audio_s < self._open_until
            ):
                self._open_until = self._audio_s + self._window_s
            return probability

    def muted(self, frame: np.ndarray) -> None:
        """Feed a frame that is not sent to the segmenter, so the wake word is still heard while talking."""
        with self._lock:
            self._advance(frame)

    def open_for(self, seconds: float) -> None:
        """Open the window for this many seconds from now. Safe to call from any thread."""
        if seconds <= 0:
            return
        with self._lock:
            self._open_until = max(self._open_until, self._audio_s + seconds)

    def reset(self) -> None:
        """Reset model state. The window stays open, so follow-up commands need no wake word."""
        self._wake.reset()
        if self._reset_detect is not None:
            self._reset_detect()
