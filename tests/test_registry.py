"""Tests for the tool registry: schema generation, validation, and safe execution."""

import json
import logging
import threading
import time

import pytest

from companion.tools.registry import ToolRegistry


def _registry() -> ToolRegistry:
    return ToolRegistry()


def test_decorator_registers_and_returns_function_unchanged():
    registry = _registry()

    @registry.tool
    def ping() -> str:
        """Reply with pong."""
        return "pong"

    assert ping() == "pong"
    assert [t.name for t in registry.tools] == ["ping"]


def test_schema_comes_from_signature_and_docstring():
    registry = _registry()

    @registry.tool
    def add(a: int, b: int = 2) -> str:
        """Add two integers.

        Args:
            a: The first number.
            b: The second number.
        """
        return str(a + b)

    schema = registry.schemas()[0]["function"]
    params = schema["parameters"]
    assert schema["description"] == "Add two integers."
    assert params["properties"]["a"]["description"] == "The first number."
    assert params["properties"]["b"]["description"] == "The second number."
    assert params["required"] == ["a"]
    assert params["additionalProperties"] is False


def test_arguments_are_validated_and_coerced():
    registry = _registry()

    @registry.tool
    def add(a: int, b: int = 2) -> str:
        """Add two integers."""
        return str(a + b)

    assert registry.execute("add", '{"a": "3"}') == "5"
    assert registry.execute("add", {"a": 1, "b": 1}) == "2"


def test_invalid_arguments_return_error_text():
    registry = _registry()

    @registry.tool
    def add(a: int) -> str:
        """Add one."""
        return str(a)

    missing = registry.execute("add", "{}")
    assert missing.startswith("Error: invalid arguments for 'add'")
    assert "a:" in missing

    extra = registry.execute("add", '{"a": 1, "rogue": true}')
    assert extra.startswith("Error: invalid arguments for 'add'")

    assert registry.execute("add", "not json").startswith("Error: invalid arguments")
    assert registry.execute("add", "[1, 2]").startswith("Error: invalid arguments")


def test_unknown_tool_is_reported():
    registry = _registry()
    result = registry.execute("nope", "{}")
    assert result.startswith("Error: unknown tool 'nope'")


def test_exceptions_are_caught_and_reported():
    registry = _registry()

    @registry.tool
    def boom() -> str:
        """Always fails."""
        raise RuntimeError("kaput")

    assert registry.execute("boom", "{}") == "Error: 'boom' failed: RuntimeError: kaput"


def test_slow_tool_times_out():
    registry = _registry()

    @registry.tool(timeout=0.05)
    def slow() -> str:
        """Sleeps too long."""
        time.sleep(0.5)
        return "done"

    assert registry.execute("slow", "{}") == "Error: 'slow' timed out after 0.05 seconds."


def test_missing_type_hint_is_rejected():
    registry = _registry()

    def untyped(value):
        """No hint."""
        return str(value)

    with pytest.raises(TypeError, match="type hint"):
        registry.register(untyped)


def test_duplicate_and_invalid_names_are_rejected():
    registry = _registry()

    @registry.tool
    def first() -> str:
        """First."""
        return "1"

    with pytest.raises(ValueError, match="already registered"):
        registry.register(first)
    with pytest.raises(ValueError, match="invalid tool name"):
        registry.register(first, name="bad name!")


def test_docstring_is_required():
    registry = _registry()

    def nodoc() -> str:
        return "x"

    with pytest.raises(ValueError, match="docstring"):
        registry.register(nodoc)


def test_timeout_must_be_positive():
    registry = _registry()

    with pytest.raises(ValueError, match="timeout"):

        @registry.tool(timeout=0)
        def bad() -> str:
            """Bad timeout."""
            return "x"


def test_tools_run_in_daemon_threads():
    registry = _registry()
    seen = {}

    @registry.tool
    def who() -> str:
        """Report thread info."""
        seen["daemon"] = threading.current_thread().daemon
        return "ok"

    assert registry.execute("who", "{}") == "ok"
    assert seen["daemon"] is True


def test_busy_guard_refuses_calls_while_timed_out_thread_runs():
    registry = _registry()
    release = threading.Event()

    @registry.tool(timeout=0.05)
    def stuck() -> str:
        """Blocks until released."""
        release.wait(5)
        return "finally"

    try:
        assert registry.execute("stuck", "{}") == "Error: 'stuck' timed out after 0.05 seconds."
        busy = registry.execute("stuck", "{}")
        assert busy.startswith("Error: 'stuck' is still running")
    finally:
        release.set()

    worker = registry.tools[0].worker
    worker.join(2)
    assert not worker.is_alive()
    assert registry.execute("stuck", "{}") == "finally"


class FakeConfirmer:
    def __init__(self, answer: bool | Exception) -> None:
        self.answer = answer
        self.questions: list[str] = []

    def confirm(self, question: str) -> bool:
        self.questions.append(question)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def test_confirmation_tool_runs_only_when_approved():
    registry = _registry()
    ran = []

    @registry.tool(requires_confirmation=True)
    def wipe(folder: str) -> str:
        """Pretend to delete a folder."""
        ran.append(folder)
        return "deleted"

    assert registry.execute("wipe", '{"folder": "x"}').startswith("Error: the user did not allow")
    assert ran == []

    approve = FakeConfirmer(True)
    registry.set_confirmer(approve)
    assert registry.execute("wipe", '{"folder": "x"}') == "deleted"
    assert ran == ["x"]
    assert approve.questions == ["Allow the companion to run wipe(folder='x')?"]

    registry.set_confirmer(FakeConfirmer(False))
    assert registry.execute("wipe", '{"folder": "y"}').startswith("Error: the user did not allow")
    assert ran == ["x"]

    registry.set_confirmer(FakeConfirmer(OSError("no console")))
    assert registry.execute("wipe", '{"folder": "z"}').startswith("Error: the user did not allow")
    assert ran == ["x"]


def test_executor_never_calls_input(monkeypatch):
    registry = _registry()
    monkeypatch.setattr("builtins.input", lambda *args: pytest.fail("executor called input()"))

    @registry.tool(requires_confirmation=True)
    def wipe() -> str:
        """Pretend to delete."""
        return "deleted"

    registry.set_confirmer(FakeConfirmer(True))
    assert registry.execute("wipe", "{}") == "deleted"


def test_confirmation_decisions_are_audited(caplog):
    registry = _registry()

    @registry.tool(requires_confirmation=True)
    def wipe() -> str:
        """Pretend to delete."""
        return "deleted"

    with caplog.at_level(logging.INFO, logger="companion.audit"):
        registry.execute("wipe", "{}")
        registry.set_confirmer(FakeConfirmer(True))
        registry.execute("wipe", "{}")
        registry.set_confirmer(FakeConfirmer(False))
        registry.execute("wipe", "{}")
        registry.set_confirmer(FakeConfirmer(OSError("boom")))
        registry.execute("wipe", "{}")

    records = [json.loads(r.getMessage()) for r in caplog.records if r.name == "companion.audit"]
    decisions = [r["decision"] for r in records if r.get("event") == "confirmation"]
    assert decisions == ["unavailable", "approved", "declined", "error"]


def test_every_call_is_audited(caplog):
    registry = _registry()

    @registry.tool
    def add(a: int) -> str:
        """Add one."""
        return str(a + 1)

    with caplog.at_level(logging.INFO, logger="companion.audit"):
        registry.execute("add", '{"a": 1}')
        registry.execute("add", '{"a": "x"}')
        registry.execute("nope", "{}")

    records = [json.loads(r.getMessage()) for r in caplog.records if r.name == "companion.audit"]
    assert [r["status"] for r in records] == ["ok", "invalid", "unknown"]
    assert records[0]["tool"] == "add"
    assert records[0]["result"] == "2"
    assert records[0]["arguments"] == '{"a": 1}'
    assert isinstance(records[0]["duration_ms"], int)
    assert "timestamp" in records[0]
