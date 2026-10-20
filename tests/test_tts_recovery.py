"""Chatterbox recovery tests: one restart after a failed request, then Piper for the session.
The engines are faked, so no server or model is needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from companion.audio.errors import SpeechError, SupersededError
from companion.audio.tts import ResilientSynthesizer


class FlakyPrimary:
    """Fails the first `failures` requests with SpeechError, then returns a short clip."""

    sample_rate = 24000
    tags = frozenset({"laugh"})

    def __init__(self, failures: int = 0, error: Exception | None = None) -> None:
        self.failures = failures
        self.error = error
        self.calls = 0
        self.loads = 0
        self.closes = 0

    def synthesize(self, text: str, *, generation: int = 0) -> np.ndarray:
        self.calls += 1
        if self.error is not None:
            raise self.error
        if self.failures > 0:
            self.failures -= 1
            raise SpeechError("server did not answer")
        return np.ones(240, dtype=np.float32)

    def load(self) -> None:
        self.loads += 1

    def close(self) -> None:
        self.closes += 1


class FakePiper:
    """Piper-like fallback at 22050 Hz: 2205 samples is exactly 0.1 second."""

    sample_rate = 22050
    tags = frozenset()

    def __init__(self) -> None:
        self.calls = 0

    def synthesize(self, text: str, *, generation: int = 0) -> np.ndarray:
        self.calls += 1
        return np.ones(2205, dtype=np.float32)


def _voice(primary: FlakyPrimary, fallback=None, notes=None, factory_error=None):
    built: list[FakePiper] = []
    fallback = fallback or FakePiper()

    def factory():
        if factory_error is not None:
            raise factory_error
        built.append(fallback)
        return fallback

    log = notes if notes is not None else []
    voice = ResilientSynthesizer(
        primary, factory, notice=lambda text, spoken=None: log.append((text, spoken))
    )
    return voice, built


def test_one_failure_restarts_chatterbox_and_keeps_it():
    primary = FlakyPrimary(failures=1)
    notes: list[tuple[str, str | None]] = []
    voice, built = _voice(primary, notes=notes)

    audio = voice.synthesize("hello")

    assert len(audio) == 240
    assert primary.closes == 1
    assert primary.loads == 1
    assert built == []
    assert not voice.on_fallback
    assert len(notes) == 1
    assert "Restarting it once" in notes[0][0]


def test_second_failure_switches_to_piper_with_resampled_audio():
    primary = FlakyPrimary(failures=2)
    fallback = FakePiper()
    notes: list[tuple[str, str | None]] = []
    voice, _ = _voice(primary, fallback=fallback, notes=notes)

    audio = voice.synthesize("hello")

    assert voice.on_fallback
    assert audio.dtype == np.float32
    assert len(audio) == 2400  # 0.1 s at the 24000 Hz output rate
    assert fallback.calls == 1
    assert primary.loads == 1
    assert any("switching to the Piper voice" in text for text, _ in notes)

    voice.synthesize("again")

    assert primary.calls == 2  # the fallback handles later requests
    assert fallback.calls == 2
    assert voice.sample_rate == 24000


def test_fallback_switch_is_spoken_once():
    primary = FlakyPrimary(failures=2)
    notes: list[tuple[str, str | None]] = []
    voice, _ = _voice(primary, notes=notes)

    voice.synthesize("one")
    voice.synthesize("two")

    spoken = [spoken for _, spoken in notes if spoken]
    assert len(spoken) == 1


def test_barge_in_is_not_treated_as_a_failure():
    primary = FlakyPrimary(error=SupersededError("cancelled"))
    voice, built = _voice(primary)

    with pytest.raises(SupersededError):
        voice.synthesize("hello")

    assert primary.loads == 0
    assert primary.closes == 0
    assert built == []
    assert not voice.on_fallback


def test_no_piper_voice_surfaces_an_error():
    primary = FlakyPrimary(failures=2)
    voice, _ = _voice(primary, factory_error=SpeechError("voice file not found"))

    with pytest.raises(SpeechError, match="voice file not found"):
        voice.synthesize("hello")
