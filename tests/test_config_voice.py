"""Tests for the voice settings in config.yaml and the config loader."""

from pathlib import Path

import pytest

from companion.config import ConfigError, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_shipped_config_has_continuous_mode_and_listening_defaults():
    config = load_config(PROJECT_ROOT / "config.yaml", {})
    assert config.input.mode == "continuous"
    assert config.audio.end_silence_ms == 900
    assert config.audio.barge_in_ms == 900
    assert config.audio.beam_size == 5
    assert config.audio.save_utterances is False


def test_unknown_input_mode_is_rejected(tmp_path):
    text = (PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8")
    assert "mode: continuous" in text
    broken = tmp_path / "config.yaml"
    broken.write_text(text.replace("mode: continuous", "mode: shouting", 1), encoding="utf-8")

    with pytest.raises(ConfigError):
        load_config(broken, {})
