"""Entry point for the companion. Picks the input mode from config.yaml (input.mode)."""

from __future__ import annotations

import json
import sys

from companion.audio.errors import SpeechError
from companion.audio.push_to_talk import PushToTalk, build_push_to_talk
from companion.chat import run_command, run_text_loop
from companion.config import AppConfig, ConfigError, get_config
from companion.conversation import Conversation
from companion.conversation_log import ConversationLog
from companion.llm.client import LLMClient
from companion.llm.context import HistoryCompactor, estimate_tokens
from companion.llm.limits import detect_context_length
from companion.llm.prompts import build_system_prompt, load_persona
from companion.logging_setup import configure_logging
from companion.memory.embedder import build_embedder
from companion.memory.store import MemoryStore
from companion.safety.confirm import TerminalConfirmer
from companion.tools import builtin  # noqa: F401  (registers built-in tools)
from companion.tools.builtin.memory_tools import set_forget_stager, set_store
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


def _tool_schema_tokens() -> int:
    return estimate_tokens(json.dumps(default_registry.schemas()))


def _build_conversation(config: AppConfig, llm: LLMClient) -> Conversation:
    """Wire persona, memory, history compaction, and the conversation log around the LLM."""
    persona_path = config.conversation.persona_file
    compactor = HistoryCompactor(
        llm.summarize,
        window_tokens=llm.context_window,
        overhead_tokens=_tool_schema_tokens,
        compact_at=config.llm.compact_at,
        keep_recent_turns=config.llm.keep_recent_turns,
        max_summary_chars=config.llm.max_summary_chars,
    )
    log = ConversationLog(
        config.conversation.log_dir,
        enabled=config.conversation.log_enabled,
    )
    store: MemoryStore | None = None
    if config.memory.enabled:
        embedder = build_embedder(
            enabled=config.memory.embeddings,
            model_name=config.memory.embedding_model,
            cache_dir=config.memory.embedding_cache,
        )
        store = MemoryStore(
            config.memory.db_path, embedder, min_score=config.memory.semantic_min_score
        )
    set_store(store)
    conversation = Conversation(
        llm=llm,
        system_prompt=lambda: build_system_prompt(load_persona(persona_path)),
        memory=store,
        memory_limit=config.memory.max_in_prompt,
        memory_chars=config.memory.max_chars,
        compactor=compactor,
        log=log,
        model=config.llm.model,
        auto_save=config.memory.auto_save,
        announce_auto=config.memory.announce_auto,
    )
    set_forget_stager(conversation.stage_forget)
    return conversation


def _build_continuous(
    config: AppConfig, llm: LLMClient, conversation: Conversation
) -> VoiceLoop | None:
    """Build the always-listening loop. Returns None (text-only) if speech cannot start."""
    try:
        return build_voice_loop(
            config,
            llm,
            default_registry,
            run_command=lambda text: run_command(text, default_registry),
            new_history=conversation.new_history,
            conversation=conversation,
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

    llm = LLMClient(
        config.llm,
        default_registry,
        context_probe=lambda: detect_context_length(
            config.llm.base_url, config.llm.model, config.llm.api_key
        ),
    )
    conversation = _build_conversation(config, llm)
    print(f"PC Companion ready. Model: {config.llm.model} at {config.llm.base_url}")

    mode = config.input.mode
    if mode == "continuous":
        if not config.audio.enabled:
            print("Speech input is off (audio.enabled is false in config.yaml). Typing only.")
            return run_text_loop(llm, default_registry, None, conversation=conversation)
        voice = _build_continuous(config, llm, conversation)
        if voice is not None:
            return voice.run()
        return run_text_loop(llm, default_registry, None, conversation=conversation)

    speech = _load_push_to_talk(config) if mode == "push_to_talk" else None
    return run_text_loop(llm, default_registry, speech, conversation=conversation)
