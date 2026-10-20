"""Tests for the audio diagnostic. The device, stream, and resolver are all fakes."""

import wave
from pathlib import Path

import numpy as np

from companion.audio.capture import SAMPLE_RATE, InputDeviceInfo
from companion.audio.errors import SpeechError
from companion.diagnose_audio import measure_levels, run, warnings_for

DEVICE = InputDeviceInfo(
    index=7,
    name="Test Mic",
    host_api="Windows WASAPI",
    default_rate=48000,
    max_channels=2,
)


class FakeStream:
    def __init__(self, level: float, overflow_every: int = 0):
        self.level = level
        self.overflow_every = overflow_every
        self.reads = 0
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        pass

    def close(self):
        self.closed = True

    def read(self, frames: int):
        self.reads += 1
        data = np.full((frames, 2), self.level, dtype=np.float32)
        overflowed = bool(self.overflow_every) and self.reads % self.overflow_every == 0
        return data, overflowed


def _factory(stream: FakeStream):
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        return stream

    factory.calls = calls
    return factory


def _run(tmp_path: Path, *, level: float = 0.5, devices=None, resolve=None, stream=None):
    lines: list[str] = []
    stream = stream or FakeStream(level)
    factory = _factory(stream)
    code = run(
        None,
        seconds=5.0,
        directory=tmp_path,
        devices=devices or (lambda: [DEVICE]),
        resolve=resolve or (lambda value: DEVICE.index),
        stream_factory=factory,
        out=lines.append,
    )
    return code, "\n".join(lines), factory, stream


def test_lists_wasapi_devices_and_marks_the_configured_one(tmp_path):
    code, output, _, _ = _run(tmp_path)
    assert code == 0
    assert "Test Mic" in output
    assert "Windows WASAPI" in output
    assert "*   7" in output or "* 7" in output


def test_records_five_seconds_at_the_native_rate(tmp_path):
    _, _, factory, stream = _run(tmp_path)
    assert factory.calls[0]["samplerate"] == 48000
    assert factory.calls[0]["device"] == 7
    assert factory.calls[0]["channels"] == 2
    assert stream.started and stream.closed


def test_saves_a_16k_mono_wav_under_the_given_directory(tmp_path):
    code, output, _, _ = _run(tmp_path)
    assert code == 0
    wavs = list(tmp_path.glob("*.wav"))
    assert len(wavs) == 1
    with wave.open(str(wavs[0]), "rb") as wav:
        assert wav.getframerate() == SAMPLE_RATE
        assert wav.getnchannels() == 1
        assert wav.getnframes() == 5 * SAMPLE_RATE
    assert str(wavs[0]) in output


def test_reports_peak_and_average_levels(tmp_path):
    _, output, _, _ = _run(tmp_path, level=0.5)
    assert "Peak: 0.500" in output
    assert "Average level (RMS)" in output


def test_constant_half_level_has_no_warnings(tmp_path):
    _, output, _, _ = _run(tmp_path, level=0.5)
    assert "Warning" not in output


def test_silence_produces_a_warning(tmp_path):
    _, output, _, _ = _run(tmp_path, level=0.0)
    assert "Warning: near silence" in output


def test_unusable_configured_device_returns_1(tmp_path):
    def resolve(value):
        raise SpeechError("no WASAPI input matches 'nope'")

    code, output, factory, _ = _run(tmp_path, resolve=resolve)
    assert code == 1
    assert "cannot be used" in output
    assert factory.calls == []


def test_no_wasapi_devices_returns_1(tmp_path):
    code, output, _, _ = _run(tmp_path, devices=lambda: [])
    assert code == 1
    assert "No WASAPI microphones" in output


def test_stream_error_returns_1_without_saving(tmp_path):
    class BrokenFactory:
        def __call__(self, **kwargs):
            raise OSError("-9997 Invalid sample rate")

    lines: list[str] = []
    code = run(
        None,
        directory=tmp_path,
        devices=lambda: [DEVICE],
        resolve=lambda value: DEVICE.index,
        stream_factory=BrokenFactory(),
        out=lines.append,
    )
    assert code == 1
    assert "Recording failed" in "\n".join(lines)
    assert list(tmp_path.glob("*.wav")) == []


def test_overflows_are_reported(tmp_path):
    _, output, _, _ = _run(tmp_path, stream=FakeStream(0.5, overflow_every=2))
    assert "overflowed" in output


def test_measure_levels_handles_empty_audio():
    levels = measure_levels(np.zeros(0, dtype=np.float32))
    assert levels.peak == 0.0
    assert levels.peak_dbfs is None


def test_warnings_flag_clipping():
    levels = measure_levels(np.ones(100, dtype=np.float32))
    assert any("clipping" in note for note in warnings_for(levels, overflows=0))
