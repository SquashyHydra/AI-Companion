"""Tests for the diagnose_tts engine comparison and the tag-preserving sentence path."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from companion import diagnose_tts
from companion.audio.errors import SpeechError
from companion.config import ChatterboxConfig, TtsConfig

SAMPLE_RATE = 24000


class FakeSynth:
    """Stands in for a synthesizer. Records every sentence it was asked to speak."""

    def __init__(self, engine: str, device: str, *, running: bool = False, fail_load: bool = False):
        self.engine = engine
        self.device = device
        self._running = running
        self._fail_load = fail_load
        self.sample_rate = SAMPLE_RATE
        self.tags: frozenset[str] = frozenset({"laugh"}) if engine.startswith("chatterbox") else frozenset()
        self.gpu_peak_mb: float | None = 1000.0 if device == "cuda" else None
        self.spoken: list[str] = []
        self.closed = False

    def running(self) -> bool:
        return self._running

    def load(self) -> None:
        if self._fail_load:
            raise SpeechError("model missing")

    def synthesize(self, text: str, *, generation: int = 0) -> np.ndarray:
        self.spoken.append(text)
        return np.zeros(SAMPLE_RATE // 2, dtype=np.float32)

    def refresh_gpu_peak(self) -> None:
        if self.device == "cuda":
            self.gpu_peak_mb = 1500.0

    def close(self) -> None:
        self.closed = True


def _tts() -> TtsConfig:
    cb = ChatterboxConfig(
        python=Path("python"),
        host="127.0.0.1",
        port=8765,
        voice_reference=None,
        autostart=True,
        startup_timeout_s=300,
        request_timeout_s=60,
    )
    return TtsConfig(engine="chatterbox_turbo", device="cuda", chatterbox=cb)


class Factories:
    """Builds fakes and remembers them so tests can inspect what was loaded."""

    def __init__(self, *, running: bool = False, fail_nano_cuda: bool = False) -> None:
        self.running = running
        self.fail_nano_cuda = fail_nano_cuda
        self.chatterbox: list[FakeSynth] = []
        self.piper: list[FakeSynth] = []

    def chatterbox_factory(self, cfg: TtsConfig) -> FakeSynth:
        fail = self.fail_nano_cuda and cfg.engine == "chatterbox_nano" and cfg.device == "cuda"
        synth = FakeSynth(cfg.engine, cfg.device, running=self.running, fail_load=fail)
        self.chatterbox.append(synth)
        return synth

    def piper_factory(self, path: Path, speed: float) -> FakeSynth:
        synth = FakeSynth("piper", "cpu")
        self.piper.append(synth)
        return synth


def _run_compare(tmp_path: Path, factories: Factories) -> tuple[int, list[str]]:
    lines: list[str] = []
    code = diagnose_tts.compare(
        _tts(),
        voice_path=Path("voice.onnx"),
        speed=1.0,
        make_chatterbox=factories.chatterbox_factory,
        make_piper=factories.piper_factory,
        out=lines.append,
        output_dir=tmp_path,
    )
    return code, lines


def test_compare_runs_every_row_and_writes_one_wav_per_sentence(tmp_path: Path) -> None:
    factories = Factories()
    code, lines = _run_compare(tmp_path, factories)

    assert code == 0
    assert len(factories.chatterbox) == 3  # turbo cuda, nano cuda, nano cpu
    assert len(factories.piper) == 1
    assert len(list(tmp_path.glob("*.wav"))) == 4 * len(diagnose_tts.COMPARE_TEXTS)
    table = "\n".join(lines)
    assert "chatterbox_nano" in table and "piper" in table and "RTF" in table


def test_compare_strips_tags_for_piper_and_keeps_them_for_chatterbox(tmp_path: Path) -> None:
    factories = Factories()
    _run_compare(tmp_path, factories)

    laugh_for_chatterbox = [s for c in factories.chatterbox for s in c.spoken if "funny" in s]
    assert laugh_for_chatterbox and all("[laugh]" in s for s in laugh_for_chatterbox)
    laugh_for_piper = [s for s in factories.piper[0].spoken if "funny" in s]
    assert laugh_for_piper and all("[laugh]" not in s for s in laugh_for_piper)


def test_compare_skips_a_row_when_a_server_is_already_running(tmp_path: Path) -> None:
    factories = Factories(running=True)
    code, lines = _run_compare(tmp_path, factories)

    assert code == 1
    assert diagnose_tts.SERVER_RUNNING in lines
    assert all(not synth.spoken for synth in factories.chatterbox)


def test_compare_reports_a_load_failure_and_keeps_going(tmp_path: Path) -> None:
    factories = Factories(fail_nano_cuda=True)
    code, lines = _run_compare(tmp_path, factories)

    assert code == 1
    assert "  failed to load: model missing" in lines
    assert len(factories.piper) == 1 and factories.piper[0].spoken  # later rows still ran


def test_compare_reads_gpu_peak_after_synthesis(tmp_path: Path) -> None:
    factories = Factories()
    _run_compare(tmp_path, factories)

    turbo = factories.chatterbox[0]
    assert turbo.gpu_peak_mb == 1500.0
    assert all(synth.closed for synth in factories.chatterbox + factories.piper)


def test_diagnose_keeps_supported_tags_in_the_spoken_sentence() -> None:
    synth = FakeSynth("chatterbox_turbo", "cuda")
    code = diagnose_tts._diagnose(
        synth,
        label="chatterbox_turbo",
        text=diagnose_tts.CHATTERBOX_TEST_TEXT,
        play=False,
        make_player=None,
        clock=lambda: 0.0,
        out=lambda _line: None,
        chosen=0,
    )

    assert code == 0
    assert "[laugh]" in synth.spoken[0]


def test_write_wav_produces_16_bit_mono_audio(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "tone.wav"
    samples: Any = np.array([0.0, 0.5, -0.5, 2.0], dtype=np.float32)

    diagnose_tts._write_wav(path, samples, SAMPLE_RATE)

    with wave.open(str(path), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == SAMPLE_RATE
        frames = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    assert frames.tolist() == [0, 16383, -16383, 32767]


@pytest.mark.parametrize("engine", ["chatterbox_turbo", "chatterbox_nano"])
def test_compare_rows_use_the_requested_chatterbox_engine(engine: str, tmp_path: Path) -> None:
    factories = Factories()
    _run_compare(tmp_path, factories)

    engines = {synth.engine for synth in factories.chatterbox}
    assert engine in engines
