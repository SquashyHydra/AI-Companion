"""Speech output diagnostic: lists WASAPI speakers, loads the Piper voice, and plays a test sentence.

Run from the project root:

    python -m companion.diagnose_tts                       # the configured engine, played aloud
    python -m companion.diagnose_tts --engine chatterbox_nano --device cpu
    python -m companion.diagnose_tts --compare             # every engine, timed, WAVs written

It reports the voice load time, the synthesis time (real-time factor), and the time to first audio
for the test sentence. Use it to pick output_device in config.yaml. --compare loads each engine in
turn (Turbo and Nano on cuda, Nano on cpu, Piper on cpu) and times a short, a long, and a [laugh]
sentence. It writes the audio to logs/diagnostics and plays nothing.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from companion.audio.errors import SpeechError
from companion.audio.playback import (
    OutputDeviceInfo,
    SpeakerPlayer,
    describe_output_devices,
    resolve_output_device,
)
from companion.audio.tts import PiperSynthesizer, _chatterbox_from_config
from companion.config import (
    CHATTERBOX_ENGINES,
    PROJECT_ROOT,
    TTS_DEVICES,
    TTS_ENGINES,
    ConfigError,
    TtsConfig,
    load_config,
)
from companion.voice.speech import sanitize_for_speech

TEST_TEXT = (
    "Hello, this is your companion. It is Tuesday, March 4, 2025, and the temperature is 21 °C. "
    "If you can hear this clearly, speech output is working."
)
CHATTERBOX_TEST_TEXT = (
    "Hello, this is your companion. That was so funny [laugh], I can't believe it. "
    "If you can hear this clearly, speech output is working."
)
COMPARE_TEXTS: dict[str, str] = {
    "short": "Hello, it is good to hear from you.",
    "long": (
        "The weather tomorrow looks mostly sunny with a light breeze, so it should be a good day "
        "to walk outside, and the forecast says it stays dry until the evening."
    ),
    "laugh": "That was so funny [laugh], I can't believe you did that.",
}
COMPARE_ROWS: tuple[tuple[str, str], ...] = (
    ("chatterbox_turbo", "cuda"),
    ("chatterbox_nano", "cuda"),
    ("chatterbox_nano", "cpu"),
    ("piper", "cpu"),
)
COMPARE_DIR = PROJECT_ROOT / "logs" / "diagnostics"
SERVER_RUNNING = (
    "A Chatterbox server is already running at the configured address. Stop it first, or the "
    "test would measure that server instead of a fresh load."
)
PLAY_TIMEOUT_EXTRA_S = 5.0
POLL_S = 0.02
DONE_MARK = "done"


def _device_lines(devices: list[OutputDeviceInfo], chosen: int | None) -> list[str]:
    lines = ["WASAPI speakers (* = configured device):"]
    for device in devices:
        mark = "*" if device.index == chosen else " "
        lines.append(
            f"  {mark} {device.index:>3}  {device.name:<45} {device.default_rate:>6} Hz  "
            f"{device.max_channels} ch"
        )
    return lines


def run(
    output_device: int | str | None,
    *,
    voice_path: Path = PROJECT_ROOT / "data" / "voices" / "en_US-lessac-medium.onnx",
    speed: float = 1.0,
    text: str | None = None,
    engine: str = "piper",
    make_chatterbox: Callable[[], Any] | None = None,
    play: bool = True,
    devices: Callable[[], list[OutputDeviceInfo]] = describe_output_devices,
    resolve: Callable[[int | str | None], int] = resolve_output_device,
    make_synth: Callable[..., Any] = PiperSynthesizer,
    make_player: Callable[..., Any] = SpeakerPlayer,
    clock: Callable[[], float] = time.perf_counter,
    out: Callable[[str], None] = print,
) -> int:
    """Run the diagnostic. Returns a process exit code (0 on success)."""
    available = devices()
    if not available:
        out("No WASAPI speakers found. Check that the output device is connected and enabled in Windows.")
        return 1

    try:
        chosen = resolve(output_device)
    except SpeechError as exc:
        out("\n".join(_device_lines(available, None)))
        out(f"The configured speaker ({output_device!r}) cannot be used: {exc}")
        return 1
    device = next((item for item in available if item.index == chosen), None)
    if device is None:
        out("\n".join(_device_lines(available, None)))
        out(f"The configured speaker ({output_device!r}) is not a WASAPI output.")
        return 1
    out("\n".join(_device_lines(available, chosen)))

    if engine in CHATTERBOX_ENGINES:
        if make_chatterbox is None:
            raise ValueError(f"make_chatterbox is required for the {engine} engine")
        synth = make_chatterbox()
        label = engine
        text = text or CHATTERBOX_TEST_TEXT
    else:
        synth = make_synth(voice_path, speed=speed)
        label = voice_path.name
        text = text or TEST_TEXT
    try:
        return _diagnose(
            synth,
            label=label,
            text=text,
            play=play,
            make_player=make_player,
            clock=clock,
            out=out,
            chosen=chosen,
        )
    finally:
        _close(synth)


def _close(synth: Any) -> None:
    close = getattr(synth, "close", None)
    if callable(close):
        close()


def _diagnose(
    synth: Any,
    *,
    label: str,
    text: str,
    play: bool,
    make_player: Callable[..., Any],
    clock: Callable[[], float],
    out: Callable[[str], None],
    chosen: int,
) -> int:
    """Load the voice, synthesize the text, and optionally play it. Returns an exit code."""
    if _server_running(synth):
        out(SERVER_RUNNING)
        return 1
    out(f"Loading {label} ...")
    started = clock()
    try:
        synth.load()
    except SpeechError as exc:
        out(f"Voice failed to load: {exc}")
        return 1
    out(f"Voice loaded in {clock() - started:.2f} s (sample rate {synth.sample_rate} Hz).")

    sentence = sanitize_for_speech(text, tags=getattr(synth, "tags", frozenset()))
    out(f"Spoken text: {sentence}")
    started = clock()
    try:
        samples = synth.synthesize(sentence)
    except SpeechError as exc:
        out(f"Synthesis failed: {exc}")
        return 1
    synth_s = clock() - started
    audio_s = samples.size / synth.sample_rate
    if audio_s <= 0:
        out("Synthesis produced no audio.")
        return 1
    out(
        f"Synthesized {audio_s:.2f} s of audio in {synth_s:.2f} s "
        f"(real-time factor {synth_s / audio_s:.2f}; below 1.0 keeps up with speech)."
    )
    out(
        f"Synthesis time for the whole test text: {synth_s:.2f} s. The companion starts playing "
        f"after the first sentence, so its time to first audio is shorter than this."
    )

    if not play:
        return 0

    try:
        player = make_player(synth.sample_rate, device=chosen)
        player.start()
    except SpeechError as exc:
        out(f"Could not open the speaker: {exc}")
        return 1
    try:
        player.write(samples, mark=DONE_MARK)
        deadline = clock() + audio_s + PLAY_TIMEOUT_EXTRA_S
        while clock() < deadline:
            if DONE_MARK in player.take_completed():
                out("Playback finished.")
                return 0
            time.sleep(POLL_S)
        out("Playback did not finish in time. Check the speaker output and Windows volume.")
        return 1
    finally:
        player.stop()


def _server_running(synth: Any) -> bool:
    running = getattr(synth, "running", None)
    return bool(callable(running) and running())


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write mono float samples to a 16-bit PCM WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def compare(
    tts: TtsConfig,
    *,
    voice_path: Path,
    speed: float,
    make_chatterbox: Callable[[TtsConfig], Any] = _chatterbox_from_config,
    make_piper: Callable[[Path, float], Any] = PiperSynthesizer,
    clock: Callable[[], float] = time.perf_counter,
    out: Callable[[str], None] = print,
    output_dir: Path = COMPARE_DIR,
) -> int:
    """Load each engine in turn and time the test sentences. Returns 0 when every row ran."""
    rows: list[dict[str, Any]] = []
    failures = 0
    out("Chatterbox-Turbo is compared on cuda only. Turbo on cpu is not run; it is the GPU voice.")
    for engine, device in COMPARE_ROWS:
        name = f"{engine} on {device}"
        if engine == "piper":
            synth = make_piper(voice_path, speed)
        else:
            synth = make_chatterbox(dataclasses.replace(tts, engine=engine, device=device))
        out(f"\n{name}: loading ...")
        if _server_running(synth):
            out(SERVER_RUNNING)
            failures += 1
            continue
        started = clock()
        try:
            synth.load()
        except SpeechError as exc:
            out(f"  failed to load: {exc}")
            failures += 1
            _close(synth)
            continue
        load_s = clock() - started
        out(f"  loaded in {load_s:.2f} s (sample rate {synth.sample_rate} Hz)")

        row_start = len(rows)
        for text_name, text in COMPARE_TEXTS.items():
            sentence = sanitize_for_speech(text, tags=getattr(synth, "tags", frozenset()))
            started = clock()
            try:
                samples = synth.synthesize(sentence)
            except SpeechError as exc:
                out(f"  {text_name}: synthesis failed: {exc}")
                failures += 1
                continue
            synth_s = clock() - started
            audio_s = samples.size / synth.sample_rate
            _write_wav(output_dir / f"{engine}_{device}_{text_name}.wav", samples, synth.sample_rate)
            rows.append(
                {
                    "engine": engine,
                    "device": device,
                    "text": text_name,
                    "load_s": load_s,
                    "synth_s": synth_s,
                    "audio_s": audio_s,
                    "rtf": synth_s / audio_s if audio_s > 0 else float("nan"),
                    "gpu_mb": None,
                }
            )
        _refresh_gpu_peak(synth)
        gpu_mb = getattr(synth, "gpu_peak_mb", None)
        for row in rows[row_start:]:
            row["gpu_mb"] = gpu_mb
        _close(synth)

    _print_table(rows, output_dir, out)
    return 0 if failures == 0 else 1


def _refresh_gpu_peak(synth: Any) -> None:
    refresh = getattr(synth, "refresh_gpu_peak", None)
    if callable(refresh):
        refresh()


def _print_table(rows: list[dict[str, Any]], output_dir: Path, out: Callable[[str], None]) -> None:
    out("")
    out(
        f"{'engine':<17}{'device':<8}{'text':<7}{'load s':>8}{'synth s':>9}"
        f"{'audio s':>9}{'RTF':>7}{'GPU MB':>9}"
    )
    for row in rows:
        gpu = "-" if row["gpu_mb"] is None else f"{row['gpu_mb']:.0f}"
        out(
            f"{row['engine']:<17}{row['device']:<8}{row['text']:<7}{row['load_s']:>8.2f}"
            f"{row['synth_s']:>9.2f}{row['audio_s']:>9.2f}{row['rtf']:>7.2f}{gpu:>9}"
        )
    out(f"Audio written to {output_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Speech output diagnostic.")
    parser.add_argument(
        "--engine",
        choices=TTS_ENGINES,
        help="TTS engine to test. Default: tts.engine from config.yaml.",
    )
    parser.add_argument(
        "--device",
        choices=TTS_DEVICES,
        help="Device for the Chatterbox engines. Default: tts.device from config.yaml.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Time every engine on short, long and [laugh] sentences and write WAV files.",
    )
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    try:
        config = load_config()
    except ConfigError as exc:
        print(f"Config error: {exc}")
        return 1
    audio = config.audio
    if args.compare:
        return compare(config.tts, voice_path=audio.voice_path, speed=audio.speech_speed)
    tts = dataclasses.replace(
        config.tts,
        engine=args.engine or config.tts.engine,
        device=args.device or config.tts.device,
    )
    return run(
        audio.output_device,
        engine=tts.engine,
        voice_path=audio.voice_path,
        speed=audio.speech_speed,
        make_chatterbox=lambda: _chatterbox_from_config(tts),
    )


if __name__ == "__main__":
    sys.exit(main())
