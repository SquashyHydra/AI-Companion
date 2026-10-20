"""Entry point for the companion. Picks the input mode from config.yaml (input.mode)."""

from __future__ import annotations

import sys

from companion.audio.errors import SpeechError
from companion.audio.push_to_talk import PushToTalk, build_push_to_talk
from companion.chat import new_history, run_command, run_text_loop
from companion.config import AppConfig, ConfigError, get_config
from companion.llm.client import LLMClient
from companion.logging_setup import configure_logging
from companion.safety.confirm import TerminalConfirmer
from companion.tools import builtin  # noqa: F401  (registers built-in tools)
from companion.tools.plugins import load_plugins
from companion.tools.registry import default_registry
from companion.voice.loop import VoiceLoop, build_voice_loop


def _load_push_to_talk(config: AppConfig) -> PushToTalk | None:
    """Load push-to-talk. Returns None (text-only) if it is disabled or unavailable."""
    if not config.audio.enabled:
        print("Speech input is off (audio.enabled is false in config.yaml). Typing only.")
        return None
    try:
        return build_push_to_talk(config.audio, apps_file=config.tools.apps_file)
    except SpeechError as exc:
        print(f"Speech input unavailable: {exc}. Typing only.")
        return None


def _build_continuous(config: AppConfig, llm: LLMClient) -> VoiceLoop | None:
    """Build the always-listening loop. Returns None (text-only) if speech cannot start."""
    try:
        return build_voice_loop(
            config,
            llm,
            default_registry,
            run_command=lambda text: run_command(text, default_registry),
            new_history=new_history,
        )
    except SpeechError as exc:
        print(f"Speech input unavailable: {exc}. Typing only.")
        default_registry.set_confirmer(TerminalConfirmer())
        return None


def main() -> int:
    try:
        config = get_config()
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 1

    configure_logging()
    report = load_plugins()
    for plugin_name in report.loaded:
        print(f"Loaded plugin: {plugin_name}")
    for plugin_name, reason in report.failed.items():
        print(f"Plugin skipped: {plugin_name} ({reason})")
    default_registry.set_confirmer(TerminalConfirmer())

    llm = LLMClient(config.llm, default_registry)
    print(f"PC Companion ready. Model: {config.llm.model} at {config.llm.base_url}")

    mode = config.input.mode
    if mode == "continuous":
        if not config.audio.enabled:
            print("Speech input is off (audio.enabled is false in config.yaml). Typing only.")
            return run_text_loop(llm, default_registry, None)
        voice = _build_continuous(config, llm)
        if voice is not None:
            return voice.run()
        return run_text_loop(llm, default_registry, None)

    speech = _load_push_to_talk(config) if mode == "push_to_talk" else None
    return run_text_loop(llm, default_registry, speech)
