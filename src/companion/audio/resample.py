"""Mono resampling with a windowed-sinc low-pass filter. Pure numpy, no scipy needed.

Used by the audio diagnostic to turn a device's native rate (for example 44.1 or 48 kHz)
into the 16 kHz mono audio that Silero VAD and Whisper expect. When downsampling, the filter
cuts off above the target Nyquist frequency so high-frequency content does not fold back
into the speech band (aliasing).
"""

from __future__ import annotations

import numpy as np

from companion.audio.capture import SAMPLE_RATE

DEFAULT_HALF_WIDTH = 8
"""Filter half-length in zero crossings of the cutoff. 8 gives about 49 taps at 48 kHz."""


def to_mono(frames: np.ndarray) -> np.ndarray:
    """Average the channels of a (samples, channels) array into a 1-D float32 array."""
    data = np.asarray(frames, dtype=np.float32)
    if data.ndim == 1:
        return data
    if data.ndim != 2:
        raise ValueError("expected a 1-D or 2-D audio array")
    return data.mean(axis=1, dtype=np.float32)


def resample(
    audio: np.ndarray,
    src_rate: int,
    dst_rate: int = SAMPLE_RATE,
    half_width: int = DEFAULT_HALF_WIDTH,
) -> np.ndarray:
    """Resample 1-D mono audio from src_rate to dst_rate. Returns float32.

    The output has floor(len(audio) * dst_rate / src_rate) samples. Output sample n is taken
    at input position n * src_rate / dst_rate, so the start of the audio stays aligned.
    """
    if src_rate <= 0 or dst_rate <= 0:
        raise ValueError("sample rates must be greater than 0")
    x = np.asarray(audio, dtype=np.float64).reshape(-1)
    if src_rate == dst_rate:
        return x.astype(np.float32)
    if x.size == 0:
        return np.zeros(0, dtype=np.float32)

    ratio = src_rate / dst_rate
    n_out = int(np.floor(x.size / ratio))
    positions = np.arange(n_out, dtype=np.float64) * ratio

    # Downsampling stretches the kernel and lowers the cutoff to the target Nyquist frequency.
    scale = max(1.0, ratio)
    cutoff = 0.5 / scale
    reach = half_width * scale
    k = int(np.ceil(reach))
    offsets = np.arange(-k, k + 1)

    indices = np.floor(positions).astype(np.int64)[:, None] + offsets[None, :]
    distance = positions[:, None] - indices
    window = np.where(np.abs(distance) < reach, 0.5 * (1.0 + np.cos(np.pi * distance / reach)), 0.0)
    kernel = 2.0 * cutoff * np.sinc(2.0 * cutoff * distance) * window

    # Taps outside the signal are zero-padded. Normalize over the taps that land inside the
    # signal, so a constant input keeps its level at the edges (side-lobe taps can be negative).
    in_range = (indices >= 0) & (indices < x.size)
    kernel = np.where(in_range, kernel, 0.0)
    kernel /= kernel.sum(axis=1, keepdims=True)

    padded = np.pad(x, (k, k))
    out = np.sum(padded[indices + k] * kernel, axis=1)
    return out.astype(np.float32)
