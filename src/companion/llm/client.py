"""OpenAI-compatible LLM client with streaming replies, cancellation, and a tool-call loop."""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import openai
from openai import NOT_GIVEN, OpenAI

from companion.config import LLMConfig
from companion.llm.streaming import ThinkFilter
from companion.tools.registry import ToolRegistry

# Also matches an unclosed block at the end, which happens when a reply is cut off mid-thought.
_THINK_RE = re.compile(r"<think>.*?(?:</think>|\Z)\s*", re.DOTALL)
INTERRUPTED_MARK = "[interrupted]"

Messages = list[dict[str, Any]]
DeltaFn = Callable[[str], None]


class LLMError(RuntimeError):
    """A user-facing error from the LLM server."""


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
    ) -> None:
        self._config = config
        self._registry = registry
        self._api = api or OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout_s,
        )

    def stream_turn(
        self,
        messages: Messages,
        user_text: str,
        cancel: threading.Event | None = None,
        on_delta: DeltaFn | None = None,
    ) -> TurnOutcome:
        """Run one user turn. The input list is not modified.

        on_delta receives visible reply text as it streams. If cancel is set, the turn stops
        early. The partial reply is kept in history and marked as interrupted. Errors raise
        LLMError and leave the caller's history unchanged.
        """
        cancel = cancel or threading.Event()
        history: Messages = [*messages, {"role": "user", "content": user_text}]
        schemas = self._registry.schemas()

        for _ in range(self._config.max_tool_rounds):
            if cancel.is_set():
                return _interrupted(history, "")
            raw, calls, stopped = self._stream_once(history, schemas, cancel, on_delta)
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
                result = self._registry.execute(call["name"], call["arguments"])
                history.append({"role": "tool", "tool_call_id": call["id"], "content": result})
            if cancel.is_set():
                return _interrupted(history, "")

        raise LLMError(f"no final answer after {self._config.max_tool_rounds} tool rounds")

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
            except openai.APITimeoutError as exc:
                raise LLMError(f"the LLM server timed out after {self._config.timeout_s:g}s") from exc
            except openai.APIConnectionError as exc:
                raise LLMError(
                    f"cannot reach the LLM server at {self._config.base_url}. Is LM Studio running?"
                ) from exc
            except openai.APIStatusError as exc:
                raise LLMError(f"the LLM server returned HTTP {exc.status_code}: {exc.message}") from exc
            except openai.APIError as exc:
                raise LLMError(f"LLM request failed: {exc}") from exc
        finally:
            _close(stream)
        return "".join(raw_parts), _ordered(calls), False


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
