"""Tests for the pure-numpy mono resampler used by the audio diagnostic."""

import numpy as np
import pytest

from companion.audio.resample import resample, to_mono


def _tone(freq: float, rate: int, seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


@pytest.mark.parametrize("src_rate", [48000, 44100])
def test_one_second_becomes_16000_samples(src_rate):
    out = resample(_tone(1000, src_rate), src_rate, 16000)
    assert out.shape == (16000,)
    assert out.dtype == np.float32


def test_in_band_tone_keeps_its_amplitude():
    out = resample(_tone(1000, 48000), 48000, 16000)
    middle = out[4000:12000]
    assert abs(float(np.max(np.abs(middle))) - 1.0) < 0.01


def test_in_band_tone_stays_aligned_with_the_reference_sine():
    out = resample(_tone(1000, 48000), 48000, 16000)
    reference = _tone(1000, 16000)
    assert float(np.max(np.abs(out[200:-200] - reference[200:-200]))) < 0.02


def test_tone_above_the_new_nyquist_is_removed():
    # 12 kHz is above 8 kHz (the 16 kHz Nyquist), so it must not fold back into the output.
    out = resample(_tone(12000, 48000), 48000, 16000)
    assert float(np.max(np.abs(out[4000:12000]))) < 0.05


def test_dc_level_is_preserved():
    out = resample(np.full(48000, 0.3, dtype=np.float32), 48000, 16000)
    assert abs(float(np.mean(out[100:-100])) - 0.3) < 0.01


def test_same_rate_is_a_copy():
    audio = _tone(440, 16000, 0.1)
    out = resample(audio, 16000, 16000)
    np.testing.assert_allclose(out, audio)


def test_empty_input_gives_empty_output():
    assert resample(np.zeros(0, dtype=np.float32), 48000, 16000).size == 0


@pytest.mark.parametrize("src, dst", [(0, 16000), (48000, 0), (-1, 16000)])
def test_invalid_rates_raise(src, dst):
    with pytest.raises(ValueError):
        resample(np.zeros(10, dtype=np.float32), src, dst)


def test_to_mono_averages_channels():
    frames = np.array([[1.0, 0.0], [0.5, 0.5], [-1.0, 1.0]], dtype=np.float32)
    np.testing.assert_allclose(to_mono(frames), [0.5, 0.5, 0.0])


def test_to_mono_passes_mono_through():
    mono = np.array([0.1, 0.2], dtype=np.float32)
    np.testing.assert_allclose(to_mono(mono), mono)
