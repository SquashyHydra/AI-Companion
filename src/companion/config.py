"""Configuration loading with environment variable overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

ENV_OVERRIDES: dict[str, tuple[str, str]] = {
    "COMPANION_LLM_BASE_URL": ("llm", "base_url"),
    "COMPANION_LLM_MODEL": ("llm", "model"),
    "COMPANION_LLM_API_KEY": ("llm", "api_key"),
    "COMPANION_NOTES_DIR": ("tools", "notes_dir"),
}


DEFAULT_CONTEXT_TOKENS = 8192


class ConfigError(ValueError):
    """Raised when the configuration file is missing or invalid."""


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    model: str
    api_key: str
    temperature: float
    timeout_s: float
    max_tool_rounds: int
    context_tokens: int = 0
    """Context window in tokens. 0 asks LM Studio for the loaded size, else DEFAULT_CONTEXT_TOKENS."""
    reply_reserve_tokens: int = 1024
    """Tokens kept free for the reply and tool results when deciding what history to send."""
    compact_at: float = 0.7
    """Share of the context window that triggers a background summary of older turns."""
    keep_recent_turns: int = 4
    """Recent user turns that are always sent word for word and never summarized."""
    max_tool_result_chars: int = 2000
    max_summary_chars: int = 1500
    connect_retries: int = 2
    """Retries when the LLM server is unreachable before any reply text has arrived."""
    retry_delay_s: float = 1.0


@dataclass(frozen=True)
class MemoryConfig:
    """Long-term facts the user asks the companion to remember."""

    enabled: bool = True
    db_path: Path = PROJECT_ROOT / "data" / "memory" / "memories.sqlite"
    max_in_prompt: int = 5
    max_chars: int = 300
    auto_save: bool = True
    """Save durable facts from each finished turn in the background (no tool call needed)."""
    announce_auto: bool = False
    """Also speak the "[noted: ...]" line for automatic saves. The line is always printed."""
    embeddings: bool = True
    """Use a small CPU embedding model so recall can match synonyms. Falls back to keywords."""
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_cache: Path = PROJECT_ROOT / "data" / "models"
    semantic_min_score: float = 0.55
    """Cosine similarity a memory needs to count as related when embeddings are on."""


@dataclass(frozen=True)
class ConversationConfig:
    persona_file: Path = PROJECT_ROOT / "data" / "persona.md"
    log_enabled: bool = True
    log_dir: Path = PROJECT_ROOT / "logs" / "conversations"


@dataclass(frozen=True)
class ToolsConfig:
    notes_dir: Path
    apps_file: Path


INPUT_MODES = ("continuous", "push_to_talk", "text")


@dataclass(frozen=True)
class InputConfig:
    """How the user talks to the companion: always-listening, push-to-talk, or typed text."""

    mode: str = "continuous"
    confirm_timeout_s: float = 30.0
    """Seconds to wait for a yes or no. No answer in time counts as no."""
    confirm_low_confidence_tools: bool = True
    """Ask for a yes before any tool runs in a turn whose speech was heard with low confidence."""


@dataclass(frozen=True)
class AudioConfig:
    enabled: bool
    input_device: int | str | None
    whisper_model: str
    whisper_device: str
    whisper_compute_type: str
    language: str | None
    vad_threshold: float
    min_seconds: float
    pre_roll_ms: int = 400
    min_speech_ms: int = 250
    end_silence_ms: int = 700
    max_utterance_s: float = 30.0
    barge_in_ms: int = 400
    beam_size: int = 5
    vocabulary: tuple[str, ...] = ()
    include_app_names: bool = True
    use_hotwords: bool = False
    low_confidence_logprob: float = -1.0
    save_utterances: bool = False
    utterance_dir: Path = PROJECT_ROOT / "logs" / "utterances"
    speak_replies: bool = True
    output_mode: str = "headphones"
    output_device: int | str | None = None
    voice_path: Path = PROJECT_ROOT / "data" / "voices" / "en_US-lessac-medium.onnx"
    speech_speed: float = 1.0
    speakers_tail_ms: int = 400


TTS_ENGINES = ("chatterbox_turbo", "chatterbox_nano", "piper")
CHATTERBOX_ENGINES = ("chatterbox_turbo", "chatterbox_nano")
TTS_DEVICES = ("cuda", "cpu")


@dataclass(frozen=True)
class ChatterboxConfig:
    python: Path
    host: str
    port: int
    voice_reference: Path | None
    autostart: bool
    startup_timeout_s: float
    request_timeout_s: float


@dataclass(frozen=True)
class TtsConfig:
    """Which voice speaks replies. Piper settings live in AudioConfig (voice_path, speech_speed)."""

    engine: str = "chatterbox_turbo"
    device: str = "cuda"
    """Device for the Chatterbox engines: cuda or cpu. Piper always runs on CPU."""
    offline: bool = True
    """When true, the Chatterbox server sets HF_HUB_OFFLINE=1 and uses only the local cache."""
    first_chunk_words: int = 7
    """Word limit for the first piece of a reply, so speech starts sooner. 0 turns it off."""
    chatterbox: ChatterboxConfig = ChatterboxConfig(
        python=PROJECT_ROOT / ".venv-tts" / "Scripts" / "python.exe",
        host="127.0.0.1",
        port=8765,
        voice_reference=None,
        autostart=True,
        startup_timeout_s=300.0,
        request_timeout_s=60.0,
    )


@dataclass(frozen=True)
class DisplayConfig:
    show_voice_tags: bool = True
    """Show voice tags such as [laugh] in the terminal. They are never spoken either way."""


@dataclass(frozen=True)
class AppConfig:
    llm: LLMConfig
    tools: ToolsConfig
    audio: AudioConfig
    input: InputConfig = InputConfig()
    tts: TtsConfig = TtsConfig()
    display: DisplayConfig = DisplayConfig()
    memory: MemoryConfig = MemoryConfig()
    conversation: ConversationConfig = ConversationConfig()


def load_config(path: Path | None = None, environ: Mapping[str, str] | None = None) -> AppConfig:
    """Load and validate the YAML config, applying environment overrides."""
    env = os.environ if environ is None else environ
    config_path = Path(path or env.get("COMPANION_CONFIG") or DEFAULT_CONFIG_PATH)
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} must contain a mapping at the top level")

    sections = {
        name: _section(raw, name)
        for name in ("llm", "tools", "audio", "input", "tts", "display", "memory", "conversation")
    }
    for env_name, (section, key) in ENV_OVERRIDES.items():
        if env.get(env_name):
            sections[section][key] = env[env_name]

    llm_raw = {**LLM_DEFAULTS, **sections["llm"]}
    tools_raw = sections["tools"]

    base_url = _str("llm", llm_raw, "base_url").rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        raise ConfigError("llm.base_url must start with http:// or https://")

    notes_dir = _project_path(_str("tools", tools_raw, "notes_dir"))
    apps_file = _project_path(_str("tools", tools_raw, "apps_file"))

    return AppConfig(
        llm=LLMConfig(
            base_url=base_url,
            model=_str("llm", llm_raw, "model"),
            api_key=_str("llm", llm_raw, "api_key"),
            temperature=_float("llm", llm_raw, "temperature", 0.0, 2.0),
            timeout_s=_float("llm", llm_raw, "timeout_s", 1.0, 600.0),
            max_tool_rounds=_int("llm", llm_raw, "max_tool_rounds", 1, 20),
            context_tokens=_int("llm", llm_raw, "context_tokens", 0, 1_000_000),
            reply_reserve_tokens=_int("llm", llm_raw, "reply_reserve_tokens", 128, 100_000),
            compact_at=_float("llm", llm_raw, "compact_at", 0.3, 0.95),
            keep_recent_turns=_int("llm", llm_raw, "keep_recent_turns", 1, 50),
            max_tool_result_chars=_int("llm", llm_raw, "max_tool_result_chars", 200, 100_000),
            max_summary_chars=_int("llm", llm_raw, "max_summary_chars", 200, 20_000),
            connect_retries=_int("llm", llm_raw, "connect_retries", 0, 10),
            retry_delay_s=_float("llm", llm_raw, "retry_delay_s", 0.0, 60.0),
        ),
        tools=ToolsConfig(notes_dir=notes_dir, apps_file=apps_file),
        audio=_audio_config(sections["audio"]),
        input=_input_config(sections["input"]),
        tts=_tts_config(sections["tts"]),
        display=_display_config(sections["display"]),
        memory=_memory_config(sections["memory"]),
        conversation=_conversation_config(sections["conversation"]),
    )


LLM_DEFAULTS: dict[str, Any] = {
    "context_tokens": 0,
    "reply_reserve_tokens": 1024,
    "compact_at": 0.7,
    "keep_recent_turns": 4,
    "max_tool_result_chars": 2000,
    "max_summary_chars": 1500,
    "connect_retries": 2,
    "retry_delay_s": 1.0,
}


def _memory_config(raw: dict[str, Any]) -> MemoryConfig:
    values: dict[str, Any] = {
        "enabled": True,
        "db_path": "data/memory/memories.sqlite",
        "max_in_prompt": 5,
        "max_chars": 300,
        "auto_save": True,
        "announce_auto": False,
        "embeddings": True,
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_cache": "data/models",
        "semantic_min_score": 0.55,
        **raw,
    }
    for flag in ("enabled", "auto_save", "announce_auto", "embeddings"):
        if not isinstance(values[flag], bool):
            raise ConfigError(f"memory.{flag} must be true or false")
    return MemoryConfig(
        enabled=values["enabled"],
        db_path=_project_path(_str("memory", values, "db_path")),
        max_in_prompt=_int("memory", values, "max_in_prompt", 0, 20),
        max_chars=_int("memory", values, "max_chars", 40, 2000),
        auto_save=values["auto_save"],
        announce_auto=values["announce_auto"],
        embeddings=values["embeddings"],
        embedding_model=_str("memory", values, "embedding_model"),
        embedding_cache=_project_path(_str("memory", values, "embedding_cache")),
        semantic_min_score=_float("memory", values, "semantic_min_score", 0.0, 1.0),
    )


def _conversation_config(raw: dict[str, Any]) -> ConversationConfig:
    values = {
        "persona_file": "data/persona.md",
        "log_enabled": True,
        "log_dir": "logs/conversations",
        **raw,
    }
    if not isinstance(values["log_enabled"], bool):
        raise ConfigError("conversation.log_enabled must be true or false")
    return ConversationConfig(
        persona_file=_project_path(_str("conversation", values, "persona_file")),
        log_enabled=values["log_enabled"],
        log_dir=_project_path(_str("conversation", values, "log_dir")),
    )


AUDIO_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "input_device": None,
    "whisper_model": "small",
    "whisper_device": "cuda",
    "whisper_compute_type": "float16",
    "language": "en",
    "vad_threshold": 0.5,
    "min_seconds": 0.3,
    "pre_roll_ms": 400,
    "min_speech_ms": 250,
    "end_silence_ms": 700,
    "max_utterance_s": 30,
    "barge_in_ms": 400,
    "beam_size": 5,
    "vocabulary": [],
    "include_app_names": True,
    "use_hotwords": False,
    "low_confidence_logprob": -1.0,
    "save_utterances": False,
    "utterance_dir": "logs/utterances",
    "speak_replies": True,
    "output_mode": "headphones",
    "output_device": None,
    "voice_path": "data/voices/en_US-lessac-medium.onnx",
    "speech_speed": 1.0,
    "speakers_tail_ms": 400,
}

OUTPUT_MODES = ("headphones", "speakers")


TTS_DEFAULTS: dict[str, Any] = {
    "engine": "chatterbox_turbo",
    "device": "cuda",
    "offline": True,
    "first_chunk_words": 7,
    "chatterbox": {
        "python": ".venv-tts/Scripts/python.exe",
        "host": "127.0.0.1",
        "port": 8765,
        "voice_reference": "",
        "autostart": True,
        "startup_timeout_s": 300,
        "request_timeout_s": 60,
    },
}


def _tts_config(raw: dict[str, Any]) -> TtsConfig:
    values = {**TTS_DEFAULTS, **raw}
    chatterbox_raw = values["chatterbox"]
    if not isinstance(chatterbox_raw, dict):
        raise ConfigError("tts.chatterbox must be a mapping")
    if "device" in chatterbox_raw:
        raise ConfigError("tts.chatterbox.device has moved to tts.device (top level of the tts section)")
    cb = {**TTS_DEFAULTS["chatterbox"], **chatterbox_raw}

    host = _str("tts.chatterbox", cb, "host")
    if host not in ("127.0.0.1", "localhost"):
        raise ConfigError("tts.chatterbox.host must be 127.0.0.1 or localhost (the server is local only)")
    if not isinstance(cb["autostart"], bool):
        raise ConfigError("tts.chatterbox.autostart must be true or false")
    if not isinstance(values["offline"], bool):
        raise ConfigError("tts.offline must be true or false")
    reference = str(cb["voice_reference"] or "").strip()

    return TtsConfig(
        engine=_choice("tts", values, "engine", TTS_ENGINES),
        device=_choice("tts", values, "device", TTS_DEVICES),
        offline=values["offline"],
        first_chunk_words=_int("tts", values, "first_chunk_words", 0, 20),
        chatterbox=ChatterboxConfig(
            python=_project_path(_str("tts.chatterbox", cb, "python")),
            host=host,
            port=_int("tts.chatterbox", cb, "port", 1024, 65535),
            voice_reference=_project_path(reference) if reference else None,
            autostart=cb["autostart"],
            startup_timeout_s=_float("tts.chatterbox", cb, "startup_timeout_s", 10.0, 1800.0),
            request_timeout_s=_float("tts.chatterbox", cb, "request_timeout_s", 1.0, 300.0),
        ),
    )


def _display_config(raw: dict[str, Any]) -> DisplayConfig:
    show = raw.get("show_voice_tags", True)
    if not isinstance(show, bool):
        raise ConfigError("display.show_voice_tags must be true or false")
    return DisplayConfig(show_voice_tags=show)


def _input_config(raw: dict[str, Any]) -> InputConfig:
    values = {"mode": "continuous", "confirm_timeout_s": 30, "confirm_low_confidence_tools": True, **raw}
    if not isinstance(values["confirm_low_confidence_tools"], bool):
        raise ConfigError("input.confirm_low_confidence_tools must be true or false")
    return InputConfig(
        mode=_choice("input", values, "mode", INPUT_MODES),
        confirm_timeout_s=_float("input", values, "confirm_timeout_s", 5.0, 300.0),
        confirm_low_confidence_tools=values["confirm_low_confidence_tools"],
    )


def _audio_config(raw: dict[str, Any]) -> AudioConfig:
    values = {**AUDIO_DEFAULTS, **raw}

    enabled = values["enabled"]
    if not isinstance(enabled, bool):
        raise ConfigError("audio.enabled must be true or false")

    input_device = _device_setting("audio.input_device", values["input_device"])

    language = values["language"]
    language_code = None if language in (None, "") else str(language).strip().lower()

    vocabulary = values["vocabulary"]
    if not isinstance(vocabulary, list) or not all(isinstance(term, str) for term in vocabulary):
        raise ConfigError("audio.vocabulary must be a list of words or phrases")
    for key in ("include_app_names", "use_hotwords", "save_utterances", "speak_replies"):
        if not isinstance(values[key], bool):
            raise ConfigError(f"audio.{key} must be true or false")

    return AudioConfig(
        enabled=enabled,
        input_device=input_device,
        whisper_model=_str("audio", values, "whisper_model"),
        whisper_device=_choice("audio", values, "whisper_device", ("cuda", "cpu")),
        whisper_compute_type=_str("audio", values, "whisper_compute_type"),
        language=language_code,
        vad_threshold=_float("audio", values, "vad_threshold", 0.0, 1.0),
        min_seconds=_float("audio", values, "min_seconds", 0.0, 10.0),
        pre_roll_ms=_int("audio", values, "pre_roll_ms", 0, 2000),
        min_speech_ms=_int("audio", values, "min_speech_ms", 0, 5000),
        end_silence_ms=_int("audio", values, "end_silence_ms", 100, 5000),
        max_utterance_s=_float("audio", values, "max_utterance_s", 1.0, 120.0),
        barge_in_ms=_int("audio", values, "barge_in_ms", 100, 5000),
        beam_size=_int("audio", values, "beam_size", 1, 10),
        vocabulary=tuple(term.strip() for term in vocabulary if term.strip()),
        include_app_names=values["include_app_names"],
        use_hotwords=values["use_hotwords"],
        low_confidence_logprob=_float("audio", values, "low_confidence_logprob", -10.0, 0.0),
        save_utterances=values["save_utterances"],
        utterance_dir=_project_path(_str("audio", values, "utterance_dir")),
        speak_replies=values["speak_replies"],
        output_mode=_choice("audio", values, "output_mode", OUTPUT_MODES),
        output_device=_device_setting("audio.output_device", values["output_device"]),
        voice_path=_project_path(_str("audio", values, "voice_path")),
        speech_speed=_float("audio", values, "speech_speed", 0.5, 2.0),
        speakers_tail_ms=_int("audio", values, "speakers_tail_ms", 0, 2000),
    )


def _device_setting(name: str, device: Any) -> int | str | None:
    """Empty means the default device. A number is a PortAudio index. Text is a name fragment."""
    if device in (None, ""):
        return None
    if isinstance(device, int) and not isinstance(device, bool) and device >= 0:
        return device
    if isinstance(device, str) and device.strip():
        return device.strip()
    raise ConfigError(f"{name} must be empty, a device number, or part of a device name")


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Return the application config, loaded once per process."""
    return load_config()


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name) or {}
    if not isinstance(value, dict):
        raise ConfigError(f"'{name}' must be a mapping")
    return dict(value)


def _value(section: str, values: dict[str, Any], key: str) -> Any:
    if values.get(key) in (None, ""):
        raise ConfigError(f"missing required setting: {section}.{key}")
    return values[key]


def _str(section: str, values: dict[str, Any], key: str) -> str:
    return str(_value(section, values, key)).strip()


def _choice(section: str, values: dict[str, Any], key: str, options: tuple[str, ...]) -> str:
    value = _str(section, values, key).lower()
    if value not in options:
        raise ConfigError(f"{section}.{key} must be one of: {', '.join(options)}")
    return value


def _float(section: str, values: dict[str, Any], key: str, low: float, high: float) -> float:
    raw = _value(section, values, key)
    try:
        number = float(str(raw))
    except ValueError as exc:
        raise ConfigError(f"{section}.{key} must be a number") from exc
    if not low <= number <= high:
        raise ConfigError(f"{section}.{key} must be between {low:g} and {high:g}")
    return number


def _int(section: str, values: dict[str, Any], key: str, low: int, high: int) -> int:
    raw = _value(section, values, key)
    try:
        number = int(str(raw).strip())
    except ValueError as exc:
        raise ConfigError(f"{section}.{key} must be a whole number") from exc
    if not low <= number <= high:
        raise ConfigError(f"{section}.{key} must be between {low} and {high}")
    return number
