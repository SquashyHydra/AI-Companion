"""Decorator-based tool registry with schema generation and safe execution."""

from __future__ import annotations

import inspect
import json
import logging
import re
import threading
import time
import typing
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from companion.safety.confirm import Confirmer

logger = logging.getLogger(__name__)
audit_logger = logging.getLogger("companion.audit")

DEFAULT_TIMEOUT_S = 10.0
MAX_AUDIT_TEXT_CHARS = 500

_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_ARG_LINE_RE = re.compile(r"^\s+(\w+)\s*:\s*(.+)$")


@dataclass(eq=False)
class Tool:
    name: str
    description: str
    func: Callable[..., Any]
    args_model: type[BaseModel]
    timeout_s: float
    requires_confirmation: bool = False
    worker: threading.Thread | None = None

    def schema(self) -> dict[str, Any]:
        """Return the OpenAI function-calling schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }


class ToolRegistry:
    """Holds tools, exposes their schemas, and runs calls with timeouts and audit logs."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._lock = threading.Lock()
        self._confirmer: Confirmer | None = None

    @property
    def tools(self) -> list[Tool]:
        return list(self._tools.values())

    def set_confirmer(self, confirmer: Confirmer | None) -> None:
        """Set the function that asks the user before a tool with requires_confirmation runs."""
        self._confirmer = confirmer

    def tool(
        self,
        func: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
        requires_confirmation: bool = False,
    ) -> Any:
        """Decorator that registers a function as a tool. Usable with or without arguments."""

        def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
            return self.register(
                f, name=name, timeout=timeout, requires_confirmation=requires_confirmation
            )

        return decorator if func is None else decorator(func)

    def register(
        self,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
        requires_confirmation: bool = False,
    ) -> Callable[..., Any]:
        tool_name = name or func.__name__
        if not _NAME_RE.match(tool_name):
            raise ValueError(f"invalid tool name '{tool_name}': use letters, digits, '_' or '-'")
        if tool_name in self._tools:
            raise ValueError(f"tool '{tool_name}' is already registered")
        if timeout <= 0:
            raise ValueError(f"tool '{tool_name}' timeout must be greater than 0")
        description, arg_docs = _parse_docstring(func.__doc__)
        if not description:
            raise ValueError(f"tool '{tool_name}' needs a docstring")
        args_model = _build_args_model(tool_name, func, arg_docs)
        self._tools[tool_name] = Tool(
            tool_name, description, func, args_model, timeout, requires_confirmation
        )
        return func

    def unregister(self, name: str) -> None:
        """Remove a registered tool, if present."""
        self._tools.pop(name, None)

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def execute(self, name: str, arguments: str | dict[str, Any] | None) -> str:
        """Validate and run a tool call. Always returns text, including errors. Every call is audited."""
        started = time.monotonic()
        status, text = self._run(name, arguments)
        _audit(name, arguments, status, text, time.monotonic() - started)
        return text

    def _run(self, name: str, arguments: str | dict[str, Any] | None) -> tuple[str, str]:
        tool = self._tools.get(name)
        if tool is None:
            available = ", ".join(self._tools) or "none"
            return "unknown", f"Error: unknown tool '{name}'. Available tools: {available}."

        try:
            raw = _load_arguments(arguments)
        except ValueError as exc:
            return "invalid", f"Error: invalid arguments for '{name}': {exc}"
        try:
            args = tool.args_model.model_validate(raw)
        except ValidationError as exc:
            return "invalid", f"Error: invalid arguments for '{name}': {_summarize(exc)}"

        kwargs = {key: getattr(args, key) for key in tool.args_model.model_fields}
        if tool.requires_confirmation and not self._confirmed(tool, kwargs):
            return "declined", f"Error: the user did not allow '{name}', so it was not run."
        return self._start(tool, kwargs)

    def _confirmed(self, tool: Tool, kwargs: dict[str, Any]) -> bool:
        decision = self._decide(tool, kwargs)
        _write_audit(
            {"timestamp": _now(), "tool": tool.name, "event": "confirmation", "decision": decision}
        )
        return decision == "approved"

    def _decide(self, tool: Tool, kwargs: dict[str, Any]) -> str:
        if self._confirmer is None:
            logger.warning("Tool '%s' needs confirmation but no confirmer is set", tool.name)
            return "unavailable"
        shown = ", ".join(f"{key}={value!r}" for key, value in kwargs.items())
        question = f"Allow the companion to run {tool.name}({shown})?"
        try:
            return "approved" if self._confirmer.confirm(question) else "declined"
        except Exception:
            logger.warning("Confirmation for '%s' failed; treating as declined", tool.name, exc_info=True)
            return "error"

    def _start(self, tool: Tool, kwargs: dict[str, Any]) -> tuple[str, str]:
        """Run the tool in a daemon thread and wait up to its timeout.

        A timed-out thread cannot be stopped. Until it ends, new calls to the same tool are refused.
        """
        outcome: dict[str, Any] = {}

        def target() -> None:
            try:
                outcome["result"] = tool.func(**kwargs)
            except Exception as exc:
                outcome["error"] = exc

        with self._lock:
            if tool.worker is not None and tool.worker.is_alive():
                return "busy", (
                    f"Error: '{tool.name}' is still running from an earlier call that timed out. "
                    "Try again later."
                )
            worker = threading.Thread(target=target, name=f"tool-{tool.name}", daemon=True)
            tool.worker = worker
            worker.start()

        worker.join(tool.timeout_s)
        if worker.is_alive():
            logger.warning(
                "Tool '%s' timed out after %g seconds and is still running", tool.name, tool.timeout_s
            )
            return "timeout", f"Error: '{tool.name}' timed out after {tool.timeout_s:g} seconds."
        if "error" in outcome:
            exc = outcome["error"]
            logger.warning("Tool '%s' raised an exception", tool.name, exc_info=exc)
            return "failed", f"Error: '{tool.name}' failed: {type(exc).__name__}: {exc}"
        result = outcome.get("result")
        return "ok", "" if result is None else str(result)


default_registry = ToolRegistry()
tool = default_registry.tool


def _parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Split a Google-style docstring into a one-line description and per-argument docs."""
    text = inspect.cleandoc(doc or "")
    head, _, args_block = text.partition("\nArgs:")
    arg_docs: dict[str, str] = {}
    for line in args_block.splitlines():
        if line and not line[0].isspace():
            break
        match = _ARG_LINE_RE.match(line)
        if match:
            arg_docs[match.group(1)] = match.group(2).strip()
    return " ".join(head.split()), arg_docs


def _build_args_model(
    tool_name: str, func: Callable[..., Any], arg_docs: dict[str, str]
) -> type[BaseModel]:
    hints = typing.get_type_hints(func)
    fields: dict[str, Any] = {}
    for param in inspect.signature(func).parameters.values():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise TypeError(f"tool '{tool_name}' cannot use *args or **kwargs")
        if param.name not in hints:
            raise TypeError(f"parameter '{param.name}' of tool '{tool_name}' needs a type hint")
        default = ... if param.default is inspect.Parameter.empty else param.default
        field = Field(default, description=arg_docs.get(param.name))
        fields[param.name] = (hints[param.name], field)
    return create_model(
        f"{tool_name}_args",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


def _load_arguments(arguments: str | dict[str, Any] | None) -> dict[str, Any]:
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    text = arguments.strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"not valid JSON ({exc.msg})") from exc
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    return data


def _summarize(exc: ValidationError) -> str:
    parts = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "arguments"
        parts.append(f"{location}: {error['msg']}")
    return "; ".join(parts)


def _audit(
    name: str,
    arguments: str | dict[str, Any] | None,
    status: str,
    text: str,
    seconds: float,
) -> None:
    arguments_text = (
        arguments if isinstance(arguments, str) else json.dumps(arguments or {}, default=str)
    )
    record = {
        "timestamp": _now(),
        "tool": name,
        "arguments": _shorten(arguments_text),
        "status": status,
        "result": _shorten(text),
        "duration_ms": round(seconds * 1000),
    }
    _write_audit(record)


def _write_audit(record: dict[str, Any]) -> None:
    audit_logger.info(json.dumps(record, ensure_ascii=False))


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _shorten(text: str) -> str:
    if len(text) <= MAX_AUDIT_TEXT_CHARS:
        return text
    return text[:MAX_AUDIT_TEXT_CHARS] + "..."
