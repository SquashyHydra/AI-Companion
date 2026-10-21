"""Entry point for the companion. Picks the input mode from config.yaml (input.mode)."""

from __future__ import annotations

import argparse
import json
import sys

from companion.audio.errors import SpeechError
from companion.audio.push_to_talk import PushToTalk, build_push_to_talk
from companion.chat import run_command, run_text_loop
from companion.config import PROJECT_ROOT, AppConfig, ConfigError, get_config
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
from companion.search.service import build_web_search, set_active_search
from companion.files.roots import root_problems
from companion.tools import builtin  # noqa: F401  (registers built-in tools)
from companion.tools.builtin.files import file_index
from companion.tools.builtin.memory_tools import set_forget_stager, set_store
from companion.tools.builtin.timers import configure_store as configure_timer_store
from companion.tools.plugins import load_plugins
from companion.tools.registry import default_registry
from companion.tray import TrayIcon
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
    search = build_web_search(config.search)
    set_active_search(search)
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
        search=search,
    )
    set_forget_stager(conversation.stage_forget)
    return conversation


def _build_continuous(
    config: AppConfig, llm: LLMClient, conversation: Conversation, *, background: bool = False
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
            keyboard=not background,
        )
    except SpeechError as exc:
        print(f"Speech input unavailable: {exc}." + ("" if background else " Typing only."))
        if not background:
            default_registry.set_confirmer(TerminalConfirmer())
        return None


def _run_voice(voice: VoiceLoop, tray_on: bool) -> int:
    """Run the voice loop. The tray icon (if on) runs beside it and its Quit ends the loop."""
    tray: TrayIcon | None = None
    if tray_on:
        tray = TrayIcon(
            on_quit=voice.request_quit,
            logs_dir=PROJECT_ROOT / "logs",
            on_toggle=voice.request_toggle,
        )
        if not tray.start():
            print("Tray icon unavailable. Install pystray with: pip install -r requirements.txt")
            tray = None
    if tray is not None:
        voice.set_listening_listener(tray.set_listening)
    try:
        return voice.run()
    finally:
        if tray is not None:
            tray.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PC Companion")
    parser.add_argument(
        "--tray",
        action="store_true",
        help="show a tray icon (voice mode only). Used by the sign-in launcher.",
    )
    parser.add_argument(
        "--mode",
        choices=("continuous", "push_to_talk", "text"),
        help="input mode. Overrides input.mode in config.yaml for this run.",
    )
    parser.add_argument(
        "--background",
        action="store_true",
        help=(
            "no keyboard or console prompts; the tray icon is the only control. "
            "Needs continuous mode. Used by the sign-in launcher."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = get_config()
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 1

    configure_logging()
    for warning in config.warnings:
        print(f"Config warning: {warning}")
    restored = configure_timer_store(
        config.timers.store_file, announce_missed=config.timers.announce_missed
    )
    if restored:
        print(f"Restored {restored} timer(s) and reminder(s).")
    for problem in root_problems(config.tools.file_roots):
        print(problem)
    file_index().start(config.files.index_refresh_minutes)
    if config.audio.enabled and config.audio.include_file_names:
        # Speech hints use the index, so wait for the first build (bounded) before voice setup.
        if not file_index().wait_ready(timeout_s=60.0):
            print("File index is still building; music names are left out of speech hints this session.")
    report = load_plugins()
    for plugin_name in report.loaded:
        print(f"Loaded plugin: {plugin_name}")
    for plugin_name, reason in report.failed.items():
        print(f"Plugin skipped: {plugin_name} ({reason})")
    if not args.background:
        # Background mode has no console to ask on, so risky tools need the voice confirmation prompt.
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

    mode = args.mode or config.input.mode
    if args.background and mode != "continuous":
        print("Background mode needs continuous voice input. Exiting.")
        return 1
    if mode == "continuous":
        if not config.audio.enabled:
            if args.background:
                print("Background mode needs speech input (audio.enabled is false in config.yaml). Exiting.")
                return 1
            print("Speech input is off (audio.enabled is false in config.yaml). Typing only.")
            return run_text_loop(llm, default_registry, None, conversation=conversation)
        voice = _build_continuous(config, llm, conversation, background=args.background)
        if voice is not None:
            return _run_voice(voice, tray_on=args.tray or config.display.tray_icon)
        if args.background:
            print("Voice could not start in background mode. Exiting.")
            return 1
        return run_text_loop(llm, default_registry, None, conversation=conversation)

    if args.tray:
        print("Tray icon needs voice mode (input.mode: continuous). Starting in text mode without it.")
    speech = _load_push_to_talk(config) if mode == "push_to_talk" else None
    return run_text_loop(llm, default_registry, speech, conversation=conversation)
