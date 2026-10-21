"""Wake word diagnostic: records a few seconds and reports how the wake word model scored them.

Run from the project root:

    python -m companion.diagnose_wakeword --seconds 10

Say the wake word a few times while it records. The report shows the highest score, how many
times the score crossed the threshold, and when. Use the numbers to pick input.wake_word.threshold
in config.yaml. Run python -m companion.diagnose_audio first if the scores stay near zero.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np

from companion.audio.capture import (
    SAMPLE_RATE,
    StreamFactory,
    _default_stream_factory,
    describe_input_devices,
    describe_stream_error,
    resolve_input_device,
)
from companion.audio.errors import SpeechError
from companion.audio.pipeline import Rechunker
from companion.audio.resample import resample, to_mono
from companion.audio.segmenter import FRAME_SAMPLES
from companion.audio.wakeword import WakeWordDetector
from companion.config import ConfigError, load_config
from companion.diagnose_audio import record

RECORD_SECONDS = 10.0
SUGGEST_FLOOR = 0.2
"""A best score below this is too weak to suggest a lower threshold for."""
THRESHOLD_MARGIN = 0.05
"""Suggested threshold sits this far below the best score seen."""

Scorer = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class Report:
    threshold: float
    frames: int
    frame_s: float
    max_score: float
    triggers: list[float] = field(default_factory=list)
    """Seconds into the recording where the score first crossed the threshold."""

    @property
    def duration_s(self) -> float:
        return self.frames * self.frame_s


def scan(frames: Iterable[np.ndarray], score: Scorer, threshold: float) -> Report:
    """Score each frame and count rising edges across the threshold. Pure logic, no audio I/O."""
    frame_s = FRAME_SAMPLES / SAMPLE_RATE
    count = 0
    max_score = 0.0
    triggers: list[float] = []
    above = False
    for frame in frames:
        value = float(score(frame))
        max_score = max(max_score, value)
        now_above = value >= threshold
        if now_above and not above:
            triggers.append(count * frame_s)
        above = now_above
        count += 1
    return Report(
        threshold=threshold,
        frames=count,
        frame_s=frame_s,
        max_score=max_score,
        triggers=triggers,
    )


def advice(report: Report) -> list[str]:
    """Plain-language next steps based on the report."""
    if report.frames == 0:
        return ["No audio was scored. Check the microphone with python -m companion.diagnose_audio."]
    if report.triggers:
        return [f"The wake word triggered {len(report.triggers)} time(s). The threshold is working."]
    if report.max_score >= SUGGEST_FLOOR:
        suggested = round(report.max_score - THRESHOLD_MARGIN, 2)
        return [
            f"The best score was {report.max_score:.2f}, just under the threshold {report.threshold:.2f}. "
            f"Try input.wake_word.threshold: {suggested:.2f} in config.yaml."
        ]
    return [
        "No strong score. Check the microphone level with python -m companion.diagnose_audio, "
        "and say the wake word closer to the microphone."
    ]


def format_report(report: Report) -> list[str]:
    lines = [
        f"Scored {report.frames} frames over {report.duration_s:.1f} s. Threshold {report.threshold:.2f}.",
        f"Highest score: {report.max_score:.3f}",
    ]
    if report.triggers:
        times = ", ".join(f"{t:.1f} s" for t in report.triggers)
        lines.append(f"Triggers: {len(report.triggers)} (at {times})")
    else:
        lines.append("Triggers: 0")
    lines.extend(advice(report))
    return lines


def capture(
    input_device: int | str | None,
    seconds: float,
    *,
    stream_factory: StreamFactory = _default_stream_factory,
) -> np.ndarray:
    """Record from the configured microphone. Returns 16 kHz mono float audio."""
    chosen = resolve_input_device(input_device)
    device = next((item for item in describe_input_devices() if item.index == chosen), None)
    if device is None:
        raise SpeechError(f"the configured microphone ({input_device!r}) is not a WASAPI input")
    channels = max(1, min(2, device.max_channels))
    frames, _ = record(stream_factory, device.index, device.default_rate, channels, seconds)
    return resample(to_mono(frames), device.default_rate, SAMPLE_RATE)


def run(
    input_device: int | str | None,
    *,
    seconds: float,
    threshold: float,
    model_name: str,
    detector: WakeWordDetector | None = None,
    capture_audio: Callable[[int | str | None, float], np.ndarray] = capture,
    out: Callable[[str], None] = print,
) -> int:
    """Run the diagnostic. Returns a process exit code (0 on success)."""
    wake = detector or WakeWordDetector(model_name, threshold)
    try:
        wake.load()
    except SpeechError as exc:
        out(str(exc))
        return 1

    out(f"Recording {seconds:g} s. Say the wake word a few times now.")
    try:
        audio = capture_audio(input_device, seconds)
    except SpeechError as exc:
        out(str(exc))
        return 1
    except Exception as exc:
        out(f"Recording failed: {describe_stream_error(exc)}")
        return 1

    wake.reset()
    frames = Rechunker(size=FRAME_SAMPLES).feed(audio)

    def score(frame: np.ndarray) -> float:
        return max(wake.scores(frame).values(), default=0.0)

    report = scan(frames, score, threshold)
    for line in format_report(report):
        out(line)
    return 0


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="Score a short recording with the wake word model.")
    parser.add_argument("--seconds", type=float, default=RECORD_SECONDS, help="recording length (default 10)")
    parser.add_argument("--threshold", type=float, help="override the configured threshold")
    parser.add_argument("--model", help="override the configured wake word model")
    args = parser.parse_args(argv)
    if args.seconds <= 0:
        parser.error("--seconds must be greater than 0")

    try:
        config = load_config()
    except ConfigError as exc:
        print(f"Config error: {exc}")
        return 1
    wake_cfg = config.input.wake_word
    threshold = args.threshold if args.threshold is not None else wake_cfg.threshold
    model = args.model or wake_cfg.model
    return run(
        config.audio.input_device,
        seconds=args.seconds,
        threshold=threshold,
        model_name=model,
    )


if __name__ == "__main__":
    sys.exit(main())
