"""OpenAI-compatible LLM client with streaming replies, cancellation, and a tool-call loop."""

from __future__ import annotations

import json
import logging
import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import openai
from openai import NOT_GIVEN, OpenAI

from companion.config import DEFAULT_CONTEXT_TOKENS, LLMConfig
from companion.llm.context import (
    SUMMARY_INSTRUCTIONS,
    estimate_tokens,
    fit_to_budget,
    truncate_text,
)
from companion.llm.streaming import ThinkFilter
from companion.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Also matches an unclosed block at the end, which happens when a reply is cut off mid-thought.
_THINK_RE = re.compile(r"<think>.*?(?:</think>|\Z)\s*", re.DOTALL)
INTERRUPTED_MARK = "[interrupted]"

Messages = list[dict[str, Any]]
DeltaFn = Callable[[str], None]


class LLMError(RuntimeError):
    """A user-facing error from the LLM server."""


class _Unreachable(LLMError):
    """The server could not be reached. Retried while no reply text has arrived yet."""


class _ContextOverflow(LLMError):
    """The request was too large for the model's context window."""


CONTEXT_ERROR_RE = re.compile(r"context|too many tokens|too long|n_ctx|n_keep", re.IGNORECASE)
SUMMARY_MAX_TOKENS = 1024
LOW_CONFIDENCE_NOTE = "The speech recognizer was unsure what the user said."


@dataclass(frozen=True)
class TurnOutcome:
    """The result of one user turn. messages is the full history including this turn."""

    messages: Messages
    reply: str
    interrupted: bool = False


class LLMClient:
    """Streams chat turns from an OpenAI-compatible server and runs tool calls until a reply."""

    def __init__(
        self,
        config: LLMConfig,
        registry: ToolRegistry,
        api: OpenAI | None = None,
        *,
        context_probe: Callable[[], int | None] | None = None,
    ) -> None:
        self._config = config
        self._registry = registry
        self._api = api or OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout_s,
        )
        self._probe = context_probe
        self._detected: int | None = None

    def context_window(self) -> int:
        """Tokens the model accepts per request: config value, then the loaded model, then the default.

        A successful lookup is kept. A failed lookup is retried on the next call, so starting
        LM Studio later still works.
        """
        if self._config.context_tokens > 0:
            return self._config.context_tokens
        if self._detected is None and self._probe is not None:
            try:
                self._detected = self._probe()
            except Exception as exc:
                logger.debug("Context length lookup failed: %s", exc)
            if self._detected:
                logger.info("Using a context window of %d tokens from LM Studio", self._detected)
        return self._detected or DEFAULT_CONTEXT_TOKENS

    def summarize(self, transcript: str) -> str:
        """Ask the model for a short summary of older turns. Called from a background thread."""
        limit = int(self.context_window() * 2.5)
        text = transcript[-limit:] if len(transcript) > limit else transcript
        try:
            response = self._api.chat.completions.create(
                model=self._config.model,
                messages=[
                    {"role": "system", "content": SUMMARY_INSTRUCTIONS},
                    {"role": "user", "content": f"Conversation:\n{text}\n/no_think"},
                ],
                temperature=0.2,
                max_tokens=SUMMARY_MAX_TOKENS,
                stream=False,
            )
        except openai.APIError as exc:
            raise _translate(exc, self._config) from exc
        content = response.choices[0].message.content if response.choices else ""
        summary = _clean(content)
        if not summary:
            raise LLMError("the model returned no summary")
        return summary

    def complete(self, messages: list[dict[str, str]], *, max_tokens: int = 300) -> str:
        """One non-streaming reply for background jobs such as memory extraction. May return ""."""
        try:
            response = self._api.chat.completions.create(
                model=self._config.model,
                messages=messages,
                temperature=0.2,
                max_tokens=max_tokens,
                stream=False,
            )
        except openai.APIError as exc:
            raise _translate(exc, self._config) from exc
        content = response.choices[0].message.content if response.choices else ""
        return _clean(content)

    def stream_turn(
        self,
        messages: Messages,
        user_text: str,
        cancel: threading.Event | None = None,
        on_delta: DeltaFn | None = None,
        *,
        extra_system: str | None = None,
        confirm_note: str | None = None,
        tool_filler: Mapping[str, str] | None = None,
    ) -> TurnOutcome:
        """Run one user turn. The input list is not modified.

        on_delta receives visible reply text as it streams. If cancel is set, the turn stops
        early. The partial reply is kept in history and marked as interrupted. Errors raise
        LLMError and leave the caller's history unchanged.

        extra_system is sent for this turn only (for example, memories) and is not stored in the
        returned history. confirm_note is shown in front of tool confirmation questions.
        tool_filler maps a tool name to a short line spoken before that tool runs, but only when
        the model has not spoken yet this turn (for example, "Let me look that up.").
        """
        cancel = cancel or threading.Event()
        history: Messages = [*messages, {"role": "user", "content": user_text}]
        schemas = self._registry.schemas()
        tool_tokens = estimate_tokens(json.dumps(schemas, ensure_ascii=False)) if schemas else 0
        emitted = [False]

        def watched(text: str) -> None:
            emitted[0] = True
            if on_delta is not None:
                on_delta(text)

        for _ in range(self._config.max_tool_rounds):
            if cancel.is_set():
                return _interrupted(history, "")
            raw, calls, stopped = self._call_model(
                history, extra_system, schemas, tool_tokens, cancel, watched, emitted
            )
            if stopped:
                return _interrupted(history, raw)
            if not calls:
                reply = _clean(raw)
                history.append({"role": "assistant", "content": reply})
                return TurnOutcome(history, reply)

            history.append(
                {
                    "role": "assistant",
                    "content": _clean(raw) or None,
                    "tool_calls": [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {"name": call["name"], "arguments": call["arguments"]},
                        }
                        for call in calls
                    ],
                }
            )
            for call in calls:
                filler = tool_filler.get(call["name"]) if tool_filler else None
                if filler and not emitted[0]:
                    watched(filler)
                result = self._registry.call(
                    call["name"], call["arguments"], confirm_note=confirm_note
                ).text
                history.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": truncate_text(result, self._config.max_tool_result_chars),
                    }
                )
            if cancel.is_set():
                return _interrupted(history, "")

        raise LLMError(f"no final answer after {self._config.max_tool_rounds} tool rounds")

    def _call_model(
        self,
        history: Messages,
        extra_system: str | None,
        schemas: list[dict[str, Any]],
        tool_tokens: int,
        cancel: threading.Event,
        on_delta: DeltaFn,
        emitted: list[bool],
    ) -> tuple[str, list[dict[str, str]], bool]:
        """Send one request that fits the context budget. Retries a dropped server and overflows."""
        budget = self.context_window() - self._config.reply_reserve_tokens - tool_tokens
        request = fit_to_budget(_with_extra(history, extra_system), budget)
        attempts = 0
        overflow_retried = False
        while True:
            try:
                return self._stream_once(request, schemas, cancel, on_delta)
            except _Unreachable as exc:
                if emitted[0] or attempts >= self._config.connect_retries:
                    raise LLMError(str(exc)) from exc
                attempts += 1
                logger.warning("LLM server unreachable; retry %d", attempts)
                if cancel.wait(self._config.retry_delay_s * attempts):
                    return "", [], True
            except _ContextOverflow as exc:
                if overflow_retried or emitted[0]:
                    raise LLMError(
                        "this conversation is too long for the model's context window. "
                        "Say new conversation to start over."
                    ) from exc
                overflow_retried = True
                budget //= 2
                request = fit_to_budget(request, budget)
                logger.warning("Request was too large; retrying with a shorter history")

    def _stream_once(
        self,
        history: Messages,
        schemas: list[dict[str, Any]],
        cancel: threading.Event,
        on_delta: DeltaFn | None,
    ) -> tuple[str, list[dict[str, str]], bool]:
        """Stream one model call. Returns (raw text, tool calls, stopped by cancel)."""
        think = ThinkFilter()
        raw_parts: list[str] = []
        calls: dict[int, dict[str, str]] = {}
        stream: Any = None
        try:
            try:
                stream = self._api.chat.completions.create(
                    model=self._config.model,
                    messages=history,
                    tools=schemas or NOT_GIVEN,
                    temperature=self._config.temperature,
                    stream=True,
                )
                for chunk in stream:
                    if cancel.is_set():
                        return "".join(raw_parts), _ordered(calls), True
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta.content:
                        raw_parts.append(delta.content)
                        visible = think.feed(delta.content)
                        if visible and on_delta is not None:
                            on_delta(visible)
                    for part in delta.tool_calls or []:
                        _merge_tool_call(calls, part)
                tail = think.flush()
                if tail and on_delta is not None:
                    on_delta(tail)
            except openai.APIError as exc:
                raise _translate(exc, self._config) from exc
        finally:
            _close(stream)
        return "".join(raw_parts), _ordered(calls), False


def _translate(exc: openai.APIError, config: LLMConfig) -> LLMError:
    """Turn an SDK error into an LLMError whose message can be spoken or printed."""
    if isinstance(exc, openai.APITimeoutError):
        return LLMError(f"the LLM server timed out after {config.timeout_s:g}s")
    if isinstance(exc, openai.APIConnectionError):
        return _Unreachable(f"cannot reach the LLM server at {config.base_url}. Is LM Studio running?")
    if isinstance(exc, openai.APIStatusError):
        if exc.status_code == 400 and CONTEXT_ERROR_RE.search(str(exc.message)):
            return _ContextOverflow(f"the LLM server rejected the request as too long: {exc.message}")
        return LLMError(f"the LLM server returned HTTP {exc.status_code}: {exc.message}")
    return LLMError(f"LLM request failed: {exc}")


def _with_extra(history: Messages, extra: str | None) -> Messages:
    """Insert per-turn system text just before the current user message, for the request only."""
    if not extra:
        return list(history)
    index = max(i for i, message in enumerate(history) if message.get("role") == "user")
    return [*history[:index], {"role": "system", "content": extra}, *history[index:]]


def _merge_tool_call(calls: dict[int, dict[str, str]], part: Any) -> None:
    """Tool calls arrive in fragments. Join them by index."""
    index = getattr(part, "index", None)
    slot = calls.setdefault(0 if index is None else index, {"id": "", "name": "", "arguments": ""})
    if part.id:
        slot["id"] = part.id
    function = part.function
    if function is None:
        return
    if function.name and not slot["name"]:
        slot["name"] = function.name
    if function.arguments:
        slot["arguments"] += function.arguments


def _ordered(calls: dict[int, dict[str, str]]) -> list[dict[str, str]]:
    result = []
    for index in sorted(calls):
        slot = calls[index]
        result.append(
            {
                "id": slot["id"] or f"call_{index}",
                "name": slot["name"],
                "arguments": slot["arguments"] or "{}",
            }
        )
    return result


def _interrupted(history: Messages, partial: str) -> TurnOutcome:
    reply = _clean(partial)
    history.append({"role": "assistant", "content": f"{reply} {INTERRUPTED_MARK}".strip()})
    return TurnOutcome(history, reply, interrupted=True)


def _close(stream: Any) -> None:
    close = getattr(stream, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _clean(content: str | None) -> str:
    """Strip Qwen3-style reasoning blocks and surrounding whitespace."""
    return _THINK_RE.sub("", content or "").strip()
