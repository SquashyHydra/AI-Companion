"""Tests for configuration loading."""

from pathlib import Path

import pytest

from companion.config import PROJECT_ROOT, ConfigError, load_config

YAML = """
llm:
  base_url: http://localhost:1234/v1/
  model: qwen3-8b
  api_key: lm-studio
  temperature: 0.7
  timeout_s: 60
  max_tool_rounds: 5
tools:
  notes_dir: data/notes
  apps_file: data/apps.json
"""


def _write(tmp_path: Path, text: str = YAML) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_values_and_normalises_paths(tmp_path):
    config = load_config(_write(tmp_path), environ={})

    assert config.llm.base_url == "http://localhost:1234/v1"
    assert config.llm.model == "qwen3-8b"
    assert config.llm.max_tool_rounds == 5
    assert config.tools.notes_dir == PROJECT_ROOT / "data" / "notes"
    assert config.tools.apps_file == PROJECT_ROOT / "data" / "apps.json"


def test_environment_overrides_take_precedence(tmp_path):
    environ = {
        "COMPANION_LLM_BASE_URL": "http://example.test:9/v1",
        "COMPANION_LLM_MODEL": "other-model",
        "COMPANION_LLM_API_KEY": "secret",
        "COMPANION_NOTES_DIR": str(tmp_path / "notes"),
    }
    config = load_config(_write(tmp_path), environ=environ)

    assert config.llm.base_url == "http://example.test:9/v1"
    assert config.llm.model == "other-model"
    assert config.llm.api_key == "secret"
    assert config.tools.notes_dir == tmp_path / "notes"


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.yaml", environ={})


def test_out_of_range_value_raises(tmp_path):
    bad = YAML.replace("temperature: 0.7", "temperature: 5")
    with pytest.raises(ConfigError, match="temperature"):
        load_config(_write(tmp_path, bad), environ={})


def test_bad_base_url_raises(tmp_path):
    bad = YAML.replace("http://localhost:1234/v1/", "localhost:1234")
    with pytest.raises(ConfigError, match="base_url"):
        load_config(_write(tmp_path, bad), environ={})


def test_missing_required_setting_raises(tmp_path):
    bad = YAML.replace("  model: qwen3-8b\n", "")
    with pytest.raises(ConfigError, match="llm.model"):
        load_config(_write(tmp_path, bad), environ={})


def test_audio_section_defaults_when_missing(tmp_path):
    config = load_config(_write(tmp_path), environ={})
    assert config.audio.enabled is True
    assert config.audio.input_device is None
    assert config.audio.whisper_model == "small"
    assert config.audio.language == "en"


def test_audio_section_overrides(tmp_path):
    text = YAML + (
        "audio:\n"
        "  enabled: false\n"
        "  input_device: Microphone\n"
        "  whisper_device: cpu\n"
        "  whisper_compute_type: int8\n"
        "  language: null\n"
        "  vad_threshold: 0.6\n"
        "  min_seconds: 0.5\n"
    )
    config = load_config(_write(tmp_path, text), environ={})
    assert config.audio.enabled is False
    assert config.audio.input_device == "Microphone"
    assert config.audio.whisper_device == "cpu"
    assert config.audio.whisper_compute_type == "int8"
    assert config.audio.language is None
    assert config.audio.vad_threshold == 0.6
    assert config.audio.min_seconds == 0.5


def test_audio_input_device_accepts_a_number(tmp_path):
    config = load_config(_write(tmp_path, YAML + "audio:\n  input_device: 2\n"), environ={})
    assert config.audio.input_device == 2


@pytest.mark.parametrize(
    "bad",
    [
        "audio:\n  enabled: maybe\n",
        "audio:\n  whisper_device: tpu\n",
        "audio:\n  vad_threshold: 2\n",
        "audio:\n  input_device: -1\n",
    ],
)
def test_invalid_audio_settings_raise(tmp_path, bad):
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, YAML + bad), environ={})


def test_confirm_timeout_defaults_to_30_seconds(tmp_path):
    config = load_config(_write(tmp_path), environ={})
    assert config.input.confirm_timeout_s == 30.0


def test_confirm_timeout_can_be_set(tmp_path):
    config = load_config(_write(tmp_path, YAML + "input:\n  confirm_timeout_s: 45\n"), environ={})
    assert config.input.confirm_timeout_s == 45.0


@pytest.mark.parametrize("bad", ["4", "301", "0"])
def test_confirm_timeout_out_of_range_raises(tmp_path, bad):
    with pytest.raises(ConfigError, match="confirm_timeout_s"):
        load_config(_write(tmp_path, YAML + f"input:\n  confirm_timeout_s: {bad}\n"), environ={})


def test_repository_config_is_valid():
    load_config(environ={})
