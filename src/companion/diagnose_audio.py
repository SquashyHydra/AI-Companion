"""Audio diagnostic: lists WASAPI microphones, records 5 seconds, and reports levels.

Run from the project root:

    python -m companion.diagnose_audio

The recording is resampled to 16 kHz mono (what the listener receives) and saved under
logs/diagnostics/. Play that WAV back to hear exactly what the companion will hear.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from companion.audio.capture import (
    SAMPLE_RATE,
    InputDeviceInfo,
    StreamFactory,
    _default_stream_factory,
    describe_input_devices,
    describe_stream_error,
    resolve_input_device,
)
from companion.audio.errors import SpeechError
from companion.audio.records import save_utterance_wav
from companion.audio.resample import resample, to_mono
from companion.config import PROJECT_ROOT, ConfigError, load_config

RECORD_SECONDS = 5.0
BLOCK_FRAMES = 1024
DIAGNOSTIC_DIR = PROJECT_ROOT / "logs" / "diagnostics"
SILENT_PEAK = 0.001
"""Peaks below this (about -60 dBFS) are reported as near silence."""
CLIP_PEAK = 0.99
"""Peaks at or above this are reported as clipping."""


@dataclass(frozen=True)
class Levels:
    peak: float
    peak_dbfs: float | None
    average_dbfs: float | None


def measure_levels(audio: np.ndarray) -> Levels:
    """Peak (linear and dBFS) and average level (RMS, dBFS) of float audio in [-1, 1]."""
    if audio.size == 0:
        return Levels(0.0, None, None)
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    return Levels(peak, _to_dbfs(peak), _to_dbfs(rms))


def warnings_for(levels: Levels, overflows: int) -> list[str]:
    notes: list[str] = []
    if levels.peak < SILENT_PEAK:
        notes.append(
            "near silence. The device may be muted, the wrong device, or receiving no audio. "
            "Check the Windows input level and the mute switch, then try another device number."
        )
    elif levels.peak >= CLIP_PEAK:
        notes.append("clipping. The peak hit full scale. Lower the microphone gain to avoid distortion.")
    if overflows:
        notes.append(f"the audio buffer overflowed {overflows} time(s); some audio may be missing.")
    return notes


def _to_dbfs(value: float) -> float | None:
    return 20.0 * math.log10(value) if value > 0 else None


def _format_dbfs(value: float | None) -> str:
    return "silent" if value is None else f"{value:.1f} dBFS"


def _device_lines(devices: list[InputDeviceInfo], chosen: int | None) -> list[str]:
    lines = ["WASAPI microphones (* = configured device):"]
    for device in devices:
        mark = "*" if device.index == chosen else " "
        lines.append(
            f"  {mark} {device.index:>3}  {device.name:<45} {device.default_rate:>6} Hz  "
            f"{device.max_channels} ch  {device.host_api}"
        )
    return lines


def record(
    stream_factory: StreamFactory,
    device: int,
    sample_rate: int,
    channels: int,
    seconds: float,
) -> tuple[np.ndarray, int]:
    """Record blocking for `seconds`. Returns (frames shaped (samples, channels), overflow count)."""
    target = int(round(seconds * sample_rate))
    stream = stream_factory(
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
        device=device,
        blocksize=BLOCK_FRAMES,
    )
    chunks: list[np.ndarray] = []
    collected = 0
    overflows = 0
    try:
        stream.start()
        while collected < target:
            data, overflowed = stream.read(min(BLOCK_FRAMES, target - collected))
            overflows += int(bool(overflowed))
            block = np.array(data, dtype=np.float32, copy=True)
            chunks.append(block)
            collected += block.shape[0]
    finally:
        _close(stream)
    if not chunks:
        return np.zeros((0, channels), dtype=np.float32), overflows
    return np.concatenate(chunks)[:target], overflows


def _close(stream: object) -> None:
    for method in ("stop", "close"):
        try:
            getattr(stream, method)()
        except Exception:
            pass


def run(
    input_device: int | str | None,
    *,
    seconds: float = RECORD_SECONDS,
    directory: Path = DIAGNOSTIC_DIR,
    devices: Callable[[], list[InputDeviceInfo]] = describe_input_devices,
    resolve: Callable[[int | str | None], int] = resolve_input_device,
    stream_factory: StreamFactory = _default_stream_factory,
    out: Callable[[str], None] = print,
) -> int:
    """Run the diagnostic. Returns a process exit code (0 on success)."""
    available = devices()
    if not available:
        out("No WASAPI microphones found. Check that the device is connected and enabled in Windows.")
        return 1

    try:
        chosen = resolve(input_device)
    except SpeechError as exc:
        out("\n".join(_device_lines(available, None)))
        out(f"The configured microphone ({input_device!r}) cannot be used: {exc}")
        return 1
    device = next((item for item in available if item.index == chosen), None)
    if device is None:
        out("\n".join(_device_lines(available, None)))
        out(f"The configured microphone ({input_device!r}) is not a WASAPI input.")
        return 1

    out("\n".join(_device_lines(available, chosen)))
    channels = max(1, min(2, device.max_channels))
    out(
        f"Recording {seconds:g} s from [{device.index}] {device.name} at {device.default_rate} Hz "
        f"({channels} ch). Speak now."
    )
    try:
        frames, overflows = record(
            stream_factory, device.index, device.default_rate, channels, seconds
        )
    except Exception as exc:
        out(f"Recording failed: {describe_stream_error(exc)}")
        out(
            "Likely causes: another app holds the device exclusively, the device was unplugged, "
            "or Windows disabled it. Pick another device number in config.yaml and run again."
        )
        return 1

    audio = resample(to_mono(frames), device.default_rate, SAMPLE_RATE)
    levels = measure_levels(audio)
    path = save_utterance_wav(audio, directory)

    out(
        f"Captured {audio.size / SAMPLE_RATE:.2f} s at {device.default_rate} Hz, "
        f"resampled to {SAMPLE_RATE} Hz mono."
    )
    out(f"Peak: {levels.peak:.3f} ({_format_dbfs(levels.peak_dbfs)})")
    out(f"Average level (RMS): {_format_dbfs(levels.average_dbfs)}")
    out(f"Saved: {path}")
    out("Play the WAV back to hear what the companion will receive.")
    for note in warnings_for(levels, overflows):
        out(f"Warning: {note}")
    return 0


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    try:
        config = load_config()
    except ConfigError as exc:
        print(f"Config error: {exc}")
        return 1
    return run(config.audio.input_device)


if __name__ == "__main__":
    sys.exit(main())
