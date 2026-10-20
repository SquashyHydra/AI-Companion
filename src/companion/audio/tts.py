"""Text to speech: the voice engines and the choice between them.

Chatterbox-Turbo and Chatterbox-Nano run in a separate process (see chatterbox_server.py) and can
render voice tags. Piper runs on CPU in this process and is the fallback when Chatterbox cannot
start.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from companion.audio.errors import SpeechError, SupersededError
from companion.audio.resample import resample
from companion.audio.tts_chatterbox import ChatterboxSynthesizer
from companion.config import PROJECT_ROOT, AudioConfig, TtsConfig

MIN_SPEED = 0.5
MAX_SPEED = 2.0
VOICE_HINT = "python -m piper.download_voices en_US-lessac-medium --download-dir data/voices"
CHATTERBOX_LABELS = {"chatterbox_turbo": "Chatterbox-Turbo", "chatterbox_nano": "Chatterbox-Nano"}


class Synthesizer(Protocol):
    """What SpeechOut needs from a voice. Tests substitute a fake.

    tags is the set of voice tags the engine renders, such as {"laugh"}. Empty for Piper.
    generation numbers the request so a barge-in can drop it (Chatterbox only; Piper ignores it).
    cancel(generation) (optional) drops stale requests on the engine side.
    close() (optional) releases the engine when speech stops.
    """

    sample_rate: int
    tags: frozenset[str]

    def synthesize(self, text: str, *, generation: int = 0) -> np.ndarray: ...


class PiperSynthesizer:
    """Wraps a Piper voice model. Speed is applied through Piper's length scale (1 / speed).

    Piper has no voice tags, so tags is empty and tags are removed from its input.
    """

    tags: frozenset[str] = frozenset()

    def __init__(self, model_path: Path, speed: float = 1.0) -> None:
        if not MIN_SPEED <= speed <= MAX_SPEED:
            raise ValueError(f"speed must be between {MIN_SPEED} and {MAX_SPEED}")
        self._model_path = Path(model_path)
        self._speed = speed
        self._voice: Any = None
        self.sample_rate = 0

    @property
    def loaded(self) -> bool:
        return self._voice is not None

    def load(self) -> None:
        if not self._model_path.is_file():
            raise SpeechError(f"voice file not found: {self._model_path}. Download it with: {VOICE_HINT}")
        try:
            from piper import PiperVoice

            self._voice = PiperVoice.load(str(self._model_path))
        except Exception as exc:
            raise SpeechError(f"could not load the Piper voice {self._model_path}: {exc}") from exc
        self.sample_rate = int(self._voice.config.sample_rate)

    def synthesize(self, text: str, *, generation: int = 0) -> np.ndarray:
        """Return the whole utterance as mono float32 samples at sample_rate."""
        if self._voice is None:
            raise SpeechError("the voice is not loaded")
        from piper import SynthesisConfig

        config = SynthesisConfig(length_scale=1.0 / self._speed)
        parts = [
            np.asarray(chunk.audio_float_array, dtype=np.float32)
            for chunk in self._voice.synthesize(text, syn_config=config)
        ]
        if not parts:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(parts)


def load_voice(
    tts: TtsConfig,
    audio: AudioConfig,
    *,
    notify: Callable[[str], None] = print,
    notice: Callable[[str, str | None], None] | None = None,
    make_chatterbox: Callable[[TtsConfig], ChatterboxSynthesizer] | None = None,
    make_piper: Callable[[AudioConfig], PiperSynthesizer] | None = None,
) -> Synthesizer:
    """Load the configured voice. If Chatterbox cannot start, fall back to Piper with a notice.

    With Chatterbox, the returned voice also recovers from failures: one restart, then Piper for
    the rest of the session. notice(text, spoken) reports those events; spoken is None for a
    plain notice.

    Raises SpeechError only if Piper also fails, which leaves the app without speech output.
    """
    make_piper_fn = make_piper or _piper_from_config
    if tts.engine in CHATTERBOX_LABELS:
        label = CHATTERBOX_LABELS[tts.engine]
        chatterbox = (make_chatterbox or _chatterbox_from_config)(tts)
        notify(f"Loading the {label} voice (the first run downloads the model)...")
        try:
            chatterbox.load()
        except SpeechError as exc:
            chatterbox.close()
            notify(f"Chatterbox voice unavailable: {exc}. Falling back to Piper.")
        else:
            notify(f"Voice ready ({label} on {tts.device}).")
            if audio.speech_speed != 1.0:
                notify(f"audio.speech_speed applies to Piper only; {label} ignores it.")
            report = notice or (lambda text, spoken=None: notify(text))
            return ResilientSynthesizer(
                chatterbox,
                lambda: _loaded_piper(make_piper_fn, audio),
                notice=report,
            )

    piper = make_piper_fn(audio)
    notify("Loading the Piper voice...")
    piper.load()
    notify(f"Voice ready (Piper {audio.voice_path.name}).")
    return piper


def _loaded_piper(make: Callable[[AudioConfig], PiperSynthesizer], audio: AudioConfig) -> PiperSynthesizer:
    piper = make(audio)
    piper.load()
    return piper


class ResilientSynthesizer:
    """Chatterbox with recovery. A failed request restarts the server once and retries. A second
    failure switches to Piper for the rest of the session.

    The output rate stays at the Chatterbox rate, because the speaker is opened for it. Piper audio
    is resampled to that rate. Barge-in (superseded requests) is passed through unchanged.
    """

    def __init__(
        self,
        primary: Synthesizer,
        fallback_factory: Callable[[], Synthesizer],
        *,
        notice: Callable[[str, str | None], None],
    ) -> None:
        self._primary = primary
        self._fallback_factory = fallback_factory
        self._notice = notice
        self._fallback: Synthesizer | None = None
        self._restarted = False
        self.sample_rate = primary.sample_rate

    @property
    def tags(self) -> frozenset[str]:
        return self._fallback.tags if self._fallback is not None else self._primary.tags

    @property
    def on_fallback(self) -> bool:
        return self._fallback is not None

    def synthesize(self, text: str, *, generation: int = 0) -> np.ndarray:
        if self._fallback is not None:
            return self._fallback_audio(text, generation)
        try:
            return self._primary.synthesize(text, generation=generation)
        except SupersededError:
            raise
        except SpeechError as exc:
            if not self._restarted:
                self._restarted = True
                self._notice(f"The Chatterbox voice stopped responding ({exc}). Restarting it once.", None)
                try:
                    self._primary.close()
                    self._primary.load()
                    return self._primary.synthesize(text, generation=generation)
                except SupersededError:
                    raise
                except SpeechError as retry_exc:
                    exc = retry_exc
            self._switch_to_fallback(exc)
            return self._fallback_audio(text, generation)

    def _switch_to_fallback(self, exc: Exception) -> None:
        fallback = self._fallback_factory()  # raises SpeechError if Piper cannot load
        self._fallback = fallback
        self._notice(
            "Chatterbox is unavailable, so I am switching to the Piper voice for this session.",
            "Chatterbox is unavailable, so I am switching to the Piper voice for this session.",
        )

    def _fallback_audio(self, text: str, generation: int) -> np.ndarray:
        assert self._fallback is not None
        audio = self._fallback.synthesize(text, generation=generation)
        if self._fallback.sample_rate == self.sample_rate:
            return audio
        return resample(audio, self._fallback.sample_rate, self.sample_rate)

    def cancel(self, generation: int) -> None:
        target = self._fallback if self._fallback is not None else self._primary
        cancel = getattr(target, "cancel", None)
        if callable(cancel):
            cancel(generation)

    def close(self) -> None:
        self._primary.close()
        if self._fallback is not None:
            close = getattr(self._fallback, "close", None)
            if callable(close):
                close()


def _chatterbox_from_config(tts: TtsConfig) -> ChatterboxSynthesizer:
    cb = tts.chatterbox
    return ChatterboxSynthesizer(
        python=cb.python,
        project_root=PROJECT_ROOT,
        engine=tts.engine,
        host=cb.host,
        port=cb.port,
        device=tts.device,
        offline=tts.offline,
        voice_reference=cb.voice_reference,
        autostart=cb.autostart,
        startup_timeout_s=cb.startup_timeout_s,
        request_timeout_s=cb.request_timeout_s,
    )


def _piper_from_config(audio: AudioConfig) -> PiperSynthesizer:
    return PiperSynthesizer(audio.voice_path, speed=audio.speech_speed)
