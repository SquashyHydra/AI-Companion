"""Tests for the Phase 4 config keys and the speech diagnostic. No audio hardware is used."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest
import yaml

from companion import diagnose_tts
from companion.audio.errors import SpeechError
from companion.audio.playback import OutputDeviceInfo
from companion.config import PROJECT_ROOT, ConfigError, load_config

PROJECT_YAML = PROJECT_ROOT / "config.yaml"


def write_audio_override(tmp_path: Path, **audio: object) -> Path:
    """Copy the project config and replace keys in its audio block."""
    data = yaml.safe_load(PROJECT_YAML.read_text(encoding="utf-8"))
    data.setdefault("audio", {}).update(audio)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_project_config_has_the_phase_4_defaults():
    config = load_config(PROJECT_YAML, environ={}).audio
    assert config.speak_replies is True
    assert config.output_mode == "headphones"
    assert config.output_device is None
    assert config.speech_speed == 1.0
    assert config.speakers_tail_ms == 400
    assert config.voice_path.name == "en_US-lessac-medium.onnx"


def test_output_mode_speakers_and_a_numeric_device_are_loaded(tmp_path: Path):
    path = write_audio_override(tmp_path, output_mode="speakers", output_device=3)
    config = load_config(path, environ={}).audio
    assert config.output_mode == "speakers"
    assert config.output_device == 3


def test_a_name_fragment_device_is_kept_as_text(tmp_path: Path):
    path = write_audio_override(tmp_path, output_device="USB Headset")
    assert load_config(path, environ={}).audio.output_device == "USB Headset"


def test_an_unknown_output_mode_is_rejected(tmp_path: Path):
    path = write_audio_override(tmp_path, output_mode="bluetooth")
    with pytest.raises(ConfigError):
        load_config(path, environ={})


def test_speech_speed_outside_the_range_is_rejected(tmp_path: Path):
    path = write_audio_override(tmp_path, speech_speed=3.0)
    with pytest.raises(ConfigError):
        load_config(path, environ={})


def test_a_negative_tail_is_rejected(tmp_path: Path):
    path = write_audio_override(tmp_path, speakers_tail_ms=-1)
    with pytest.raises(ConfigError):
        load_config(path, environ={})


# ----- diagnose_tts.run -----


class FakeSynth:
    sample_rate = 10

    def __init__(self, voice_path=None, speed=1.0, fail_load: bool = False) -> None:
        self.fail_load = fail_load
        self.loaded = False
        self.texts: list[str] = []

    def load(self) -> None:
        if self.fail_load:
            raise SpeechError("voice file is missing")
        self.loaded = True

    def synthesize(self, text: str, *, generation: int = 0) -> np.ndarray:
        self.texts.append(text)
        return np.zeros(20, dtype=np.float32)  # 2.0 s at 10 Hz


class FakePlayer:
    def __init__(self, sample_rate, device=None) -> None:
        self.device = device
        self.writes: list[object] = []
        self.stopped = False

    def start(self) -> None:
        pass

    def write(self, samples, mark=None) -> None:
        self.writes.append(mark)

    def take_completed(self) -> list[object]:
        return [diagnose_tts.DONE_MARK] if self.writes else []

    def stop(self) -> None:
        self.stopped = True


SPEAKER = OutputDeviceInfo(index=4, name="Headphones (USB Headset)", default_rate=48000, max_channels=2)


def run_diagnostic(**overrides):
    lines: list[str] = []
    kwargs = dict(
        output_device=None,
        voice_path=Path("voice.onnx"),
        speed=1.0,
        play=True,
        devices=lambda: [SPEAKER],
        resolve=lambda value: 4,
        make_synth=FakeSynth,
        make_player=FakePlayer,
        clock=itertools.count(0.0, 0.5).__next__,
        out=lines.append,
    )
    kwargs.update(overrides)
    code = diagnose_tts.run(**kwargs)
    return code, "\n".join(lines)


def test_a_full_run_plays_the_test_sentence_and_exits_zero():
    players: list[FakePlayer] = []

    def make_player(sample_rate, device=None):
        player = FakePlayer(sample_rate, device)
        players.append(player)
        return player

    code, output = run_diagnostic(make_player=make_player)
    assert code == 0
    assert "Playback finished." in output
    assert "real-time factor" in output
    assert players[0].device == 4
    assert players[0].stopped


def test_play_false_synthesizes_without_opening_a_speaker():
    opened: list[bool] = []
    code, output = run_diagnostic(play=False, make_player=lambda *a, **k: opened.append(True))
    assert code == 0
    assert opened == []


def test_no_speakers_found_exits_one():
    code, output = run_diagnostic(devices=lambda: [])
    assert code == 1
    assert "No WASAPI speakers found" in output


def test_a_configured_device_that_cannot_be_resolved_exits_one():
    def resolve(value):
        raise SpeechError("no WASAPI speaker matches 'bluetooth'")

    code, output = run_diagnostic(output_device="bluetooth", resolve=resolve)
    assert code == 1
    assert "cannot be used" in output


def test_a_resolved_index_missing_from_the_device_list_exits_one():
    code, output = run_diagnostic(resolve=lambda value: 99)
    assert code == 1
    assert "is not a WASAPI output" in output


def test_a_voice_that_fails_to_load_exits_one():
    code, output = run_diagnostic(make_synth=lambda voice_path, speed=1.0: FakeSynth(fail_load=True))
    assert code == 1
    assert "Voice failed to load" in output


def test_a_speaker_that_cannot_open_exits_one():
    def make_player(sample_rate, device=None):
        raise SpeechError("the device rejected the sample rate")

    code, output = run_diagnostic(make_player=make_player)
    assert code == 1
    assert "Could not open the speaker" in output
