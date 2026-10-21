"""faster-whisper transcription. Uses the GPU when available and falls back to CPU."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from companion.audio.errors import SpeechError
from companion.config import AudioConfig
from companion.safety.allowlist import load_allowlist

logger = logging.getLogger("companion.audio")

ModelFactory = Callable[[str, str, str], Any]
"""(model_size, device, compute_type) -> model with a faster-whisper style transcribe()."""

DEFAULT_LOW_CONFIDENCE_LOGPROB = -1.0


@dataclass(frozen=True)
class Transcription:
    """Text plus the confidence signals Whisper gives. Signals are None when the model omits them."""

    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None

    def low_confidence(self, threshold: float = DEFAULT_LOW_CONFIDENCE_LOGPROB) -> bool:
        return self.avg_logprob is not None and self.avg_logprob < threshold


def _whisper_factory(model_size: str, device: str, compute_type: str) -> Any:
    from faster_whisper import WhisperModel

    return WhisperModel(model_size, device=device, compute_type=compute_type)


class Transcriber:
    """Turns 16 kHz mono float32 audio into text."""

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str | None = None,
        model_factory: ModelFactory | None = None,
        beam_size: int = 5,
        initial_prompt: str | None = None,
        hotwords: str | None = None,
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._factory = model_factory or _whisper_factory
        self._beam_size = beam_size
        self._initial_prompt = initial_prompt or None
        self._hotwords = hotwords or None
        self._model: Any = None

    @property
    def device(self) -> str:
        """The device actually in use. This can be cpu after a CUDA fallback."""
        return self._device

    def load(self) -> None:
        """Load the model now so startup reports problems early."""
        if self._model is None:
            self._model = self._build()

    def transcribe(self, audio: np.ndarray) -> str:
        return self.transcribe_detailed(audio).text

    def transcribe_detailed(self, audio: np.ndarray) -> Transcription:
        if audio.size == 0:
            return Transcription(text="")
        self.load()
        try:
            return self._run(self._model, audio)
        except Exception as exc:
            if self._device == "cpu":
                raise SpeechError(f"transcription failed: {exc}") from exc
            logger.warning("GPU transcription failed (%s); reloading the model on CPU", exc)
            self._switch_to_cpu()
            self._model = self._build()
            try:
                return self._run(self._model, audio)
            except Exception as retry_exc:
                raise SpeechError(f"transcription failed on CPU: {retry_exc}") from retry_exc

    def _build(self) -> Any:
        try:
            return self._factory(self._model_size, self._device, self._compute_type)
        except Exception as exc:
            if self._device == "cpu":
                raise SpeechError(f"could not load Whisper model '{self._model_size}': {exc}") from exc
            logger.warning("Whisper on %s failed to load (%s); falling back to CPU", self._device, exc)
            self._switch_to_cpu()
            return self._build()

    def _switch_to_cpu(self) -> None:
        self._device = "cpu"
        self._compute_type = "int8"

    def _run(self, model: Any, audio: np.ndarray) -> Transcription:
        # The segmenter or VAD trimmer has already cut the audio to speech, so Whisper's
        # own VAD filter stays off. Each utterance is decoded on its own, so earlier
        # text is not fed back in as context.
        segments, _info = model.transcribe(
            audio,
            language=self._language,
            beam_size=self._beam_size,
            vad_filter=False,
            condition_on_previous_text=False,
            initial_prompt=self._initial_prompt,
            hotwords=self._hotwords,
        )
        parts = list(segments)
        text = " ".join(str(getattr(part, "text", "")).strip() for part in parts).strip()
        return Transcription(
            text=text,
            avg_logprob=_lowest(getattr(part, "avg_logprob", None) for part in parts),
            no_speech_prob=_highest(getattr(part, "no_speech_prob", None) for part in parts),
        )


def _lowest(values: Iterable[float | None]) -> float | None:
    found = [value for value in values if value is not None]
    return min(found) if found else None


def _highest(values: Iterable[float | None]) -> float | None:
    found = [value for value in values if value is not None]
    return max(found) if found else None


def speech_vocabulary(audio_config: AudioConfig, apps_file: Path | None) -> list[str]:
    """Words to bias recognition toward: the configured vocabulary, allowlisted app names, and
    artist and album folder names from the file index (once the index has been built)."""
    words = list(audio_config.vocabulary)
    if audio_config.include_app_names and apps_file is not None:
        words += list(load_allowlist(apps_file).keys())
    if audio_config.include_file_names:
        from companion.tools.builtin.files import file_index

        words += file_index().speech_terms()
    unique: dict[str, str] = {}
    for word in words:
        cleaned = " ".join(str(word).split())
        if cleaned:
            unique.setdefault(cleaned.lower(), cleaned)
    return list(unique.values())


def make_transcriber(audio_config: AudioConfig, apps_file: Path | None = None) -> Transcriber:
    """Build a Transcriber from config. Used by both push-to-talk and continuous listening."""
    vocabulary = speech_vocabulary(audio_config, apps_file)
    return Transcriber(
        model_size=audio_config.whisper_model,
        device=audio_config.whisper_device,
        compute_type=audio_config.whisper_compute_type,
        language=audio_config.language,
        beam_size=audio_config.beam_size,
        initial_prompt=build_prompt(vocabulary),
        hotwords=" ".join(vocabulary) if audio_config.use_hotwords and vocabulary else None,
    )


def build_prompt(words: Iterable[str]) -> str | None:
    """Build a Whisper initial prompt that biases spelling toward known names.

    Whisper treats this as text it saw before the audio. It helps with names and
    jargon, but it does not force a match.
    """
    unique: list[str] = []
    for word in words:
        cleaned = " ".join(str(word).split())
        if cleaned and cleaned.lower() not in (item.lower() for item in unique):
            unique.append(cleaned)
    if not unique:
        return None
    return "Vocabulary: " + ", ".join(unique) + "."
