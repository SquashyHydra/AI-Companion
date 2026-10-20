"""Debug records for continuous listening: a JSONL transcript log and optional utterance WAV files."""

from __future__ import annotations

import json
import logging
import threading
import wave
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from companion.audio.capture import SAMPLE_RATE
from companion.audio.stt import Transcription

logger = logging.getLogger("companion.audio")


class TranscriptLog:
    """Appends one JSON object per utterance. Write failures are logged, never raised."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def append(self, *, transcription: Transcription, seconds: float, low_confidence: bool) -> None:
        record: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "text": transcription.text,
            "seconds": round(seconds, 2),
            "avg_logprob": _round(transcription.avg_logprob),
            "no_speech_prob": _round(transcription.no_speech_prob),
            "low_confidence": low_confidence,
        }
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            except OSError:
                logger.warning("could not write the transcript log", exc_info=True)


def save_utterance_wav(audio: np.ndarray, directory: Path, sample_rate: int = SAMPLE_RATE) -> Path:
    """Save one utterance as 16-bit mono PCM WAV and return the file path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{datetime.now():%Y%m%d-%H%M%S-%f}.wav"
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return path


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 3)
