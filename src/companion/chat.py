"""Typed and push-to-talk conversation loop, shared commands, and the streaming reply printer."""

from __future__ import annotations

from collections.abc import Callable

import threading

from companion.audio.capture import list_input_devices
from companion.audio.push_to_talk import PushToTalk
from companion.conversation import Conversation
from companion.llm.client import LLMClient, LLMError, Messages
from companion.llm.prompts import SYSTEM_PROMPT
from companion.notices import notices
from companion.tools.registry import ToolRegistry

COMMANDS = "/tools  /mics  /new  /status  /exit"


def new_history(system_prompt: str = SYSTEM_PROMPT) -> Messages:
    return [{"role": "system", "content": system_prompt}]


def help_text(voice: bool) -> str:
    lines = []
    if voice:
        lines.append("Type a message, or press Enter on an empty line to talk.")
    lines.append(f"Commands: {COMMANDS}")
    return "\n".join(lines)


def microphones_text() -> str:
    try:
        devices = list_input_devices()
    except Exception as exc:
        return f"Could not list microphones: {exc}"
    if not devices:
        return "No WASAPI microphones found."
    lines = [f"  {index}: {name}" for index, name in devices]
    lines.append("Set audio.input_device in config.yaml to a number or part of a name.")
    return "\n".join(lines)


def tools_text(registry: ToolRegistry) -> str:
    lines = []
    for registered in registry.tools:
        rule = registered.requires_confirmation
        if callable(rule):
            note = " (may ask first)"
        else:
            note = " (asks first)" if rule else ""
        lines.append(f"  {registered.name}{note}: {registered.description}")
    return "\n".join(lines) or "No tools registered."


def run_command(text: str, registry: ToolRegistry) -> str:
    """Handle a slash command. Returns the text to show the user."""
    command = text.strip().lower()
    if command == "/mics":
        return microphones_text()
    if command == "/tools":
        return tools_text(registry)
    return f"Unknown command: {text.strip()}. Commands: {COMMANDS}"


def answer_turn(
    llm: LLMClient,
    messages: Messages,
    text: str,
    conversation: Conversation | None = None,
) -> Messages:
    """Stream one reply to the screen. Returns the updated history.

    Ctrl+C stops the reply and keeps the previous history. LLMError propagates to the caller.
    """
    print("\nCompanion> ", end="", flush=True)
    streamed = False

    def on_delta(chunk: str) -> None:
        nonlocal streamed
        streamed = True
        print(chunk, end="", flush=True)

    try:
        if conversation is not None:
            outcome = conversation.stream(messages, text, threading.Event(), on_delta)
        else:
            outcome = llm.stream_turn(messages, text, on_delta=on_delta)
    except KeyboardInterrupt:
        print("\n(interrupted)")
        return messages
    if not streamed:
        print("(no reply)", end="")
    print()
    if conversation is not None:
        conversation.after_turn(outcome.messages)
    return outcome.messages


def run_text_loop(
    llm: LLMClient,
    registry: ToolRegistry,
    speech: PushToTalk | None,
    input_fn: Callable[[str], str] = input,
    conversation: Conversation | None = None,
) -> int:
    """Typed chat. With speech set, an empty Enter records one push-to-talk utterance."""
    messages = conversation.new_history() if conversation is not None else new_history()
    print(help_text(voice=speech is not None))
    while True:
        if conversation is not None:
            messages = conversation.apply(messages)
        for notice in notices.drain():
            print(f"\n* {notice}")
        try:
            user_text = input_fn("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            if speech is None:
                continue
            user_text = speech.listen()
            if not user_text:
                continue
            print(f'Heard: "{user_text}"')
        if user_text.lower() in ("/exit", "/quit"):
            break
        if user_text.lower() in ("/reset", "/new"):
            messages = conversation.new_history() if conversation is not None else new_history()
            print("Conversation cleared.")
            continue
        if user_text.lower() == "/status":
            if conversation is not None:
                print(conversation.status(messages))
            else:
                print("No status available.")
            continue
        if user_text.startswith("/"):
            print(run_command(user_text, registry))
            continue
        try:
            messages = answer_turn(llm, messages, user_text, conversation)
        except LLMError as exc:
            print(f"Error: {exc}")
    return 0
