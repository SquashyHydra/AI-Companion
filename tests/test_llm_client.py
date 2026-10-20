"""Tests for the streaming LLM client: deltas, tool-call fragments, cancellation, and errors.

Uses a fake OpenAI-compatible API that returns queued streams of chunks.
"""

import threading
from types import SimpleNamespace

import openai
import pytest

from companion.config import LLMConfig
from companion.llm.client import INTERRUPTED_MARK, LLMClient, LLMError
from companion.tools.registry import ToolRegistry


def _config(max_tool_rounds: int = 3) -> LLMConfig:
    return LLMConfig(
        base_url="http://localhost:1234/v1",
        model="test-model",
        api_key="k",
        temperature=0.1,
        timeout_s=5,
        max_tool_rounds=max_tool_rounds,
    )


def _registry() -> ToolRegistry:
    registry = ToolRegistry()

    @registry.tool
    def echo(text: str) -> str:
        """Echo the text back."""
        return f"echo: {text}"

    return registry


def _chunk(content: str | None = None, tool_calls: list | None = None) -> SimpleNamespace:
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def _tool_part(index: int, call_id=None, name=None, arguments=None) -> SimpleNamespace:
    function = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=call_id, function=function)


class FakeAPI:
    """Each create() call pops one response: a list of chunks, or an exception to raise."""

    def __init__(self, *responses):
        self._responses = list(responses)
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return iter(item)


def _history() -> list[dict]:
    return [{"role": "system", "content": "sys"}]


def _run(client, messages, text, cancel=None):
    deltas: list[str] = []
    outcome = client.stream_turn(messages, text, cancel=cancel, on_delta=deltas.append)
    return outcome, deltas


def test_plain_reply_streams_deltas_and_is_stored():
    api = FakeAPI([_chunk("Hel"), _chunk("lo.")])
    client = LLMClient(_config(), _registry(), api=api)
    messages = _history()

    outcome, deltas = _run(client, messages, "hi")

    assert outcome.reply == "Hello."
    assert not outcome.interrupted
    assert deltas == ["Hel", "lo."]
    assert [m["role"] for m in outcome.messages] == ["system", "user", "assistant"]
    assert outcome.messages[-1]["content"] == "Hello."
    assert api.requests[0]["stream"] is True
    assert api.requests[0]["model"] == "test-model"
    assert api.requests[0]["tools"][0]["function"]["name"] == "echo"


def test_input_history_is_not_modified():
    client = LLMClient(_config(), _registry(), api=FakeAPI([_chunk("Hi")]))
    messages = _history()

    _run(client, messages, "hello")

    assert messages == _history()


def test_think_blocks_are_not_streamed_or_stored():
    api = FakeAPI([_chunk("<think>hmm, greeting"), _chunk("</think>\nHello!")])
    client = LLMClient(_config(), _registry(), api=api)

    outcome, deltas = _run(client, _history(), "hi")

    assert outcome.reply == "Hello!"
    assert "<think>" not in "".join(deltas)
    assert "".join(deltas).strip() == "Hello!"


def test_tool_call_fragments_are_joined_executed_and_sent_back():
    api = FakeAPI(
        [
            _chunk(tool_calls=[_tool_part(0, call_id="c1", name="echo", arguments='{"text":')]),
            _chunk(tool_calls=[_tool_part(0, arguments=' "ping"}')]),
        ],
        [_chunk("The tool said echo: ping.")],
    )
    client = LLMClient(_config(), _registry(), api=api)

    outcome, _ = _run(client, _history(), "echo ping")

    assert outcome.reply == "The tool said echo: ping."
    tool_message = next(m for m in outcome.messages if m["role"] == "tool")
    assert tool_message == {"role": "tool", "tool_call_id": "c1", "content": "echo: ping"}
    assert len(api.requests) == 2


def test_unknown_tool_result_is_fed_back_to_model():
    api = FakeAPI(
        [_chunk(tool_calls=[_tool_part(0, call_id="c1", name="missing", arguments="{}")])],
        [_chunk("Sorry, that tool is not available.")],
    )
    client = LLMClient(_config(), _registry(), api=api)

    outcome, _ = _run(client, _history(), "do it")

    tool_message = next(m for m in outcome.messages if m["role"] == "tool")
    assert tool_message["content"].startswith("Error: unknown tool 'missing'")


def test_too_many_tool_rounds_raises_and_leaves_history_alone():
    looping = [
        [_chunk(tool_calls=[_tool_part(0, call_id=f"c{i}", name="echo", arguments='{"text": "x"}')])]
        for i in range(2)
    ]
    client = LLMClient(_config(max_tool_rounds=2), _registry(), api=FakeAPI(*looping))
    messages = _history()

    with pytest.raises(LLMError, match="tool rounds"):
        _run(client, messages, "loop")
    assert messages == _history()


def test_connection_error_is_friendly_and_leaves_history_alone():
    error = openai.APIConnectionError(request=None)
    client = LLMClient(_config(), _registry(), api=FakeAPI(error))
    messages = _history()

    with pytest.raises(LLMError, match="cannot reach the LLM server"):
        _run(client, messages, "hi")
    assert messages == _history()


def test_empty_stream_gives_empty_reply_not_an_error():
    client = LLMClient(_config(), _registry(), api=FakeAPI([]))

    outcome, deltas = _run(client, _history(), "hi")

    assert outcome.reply == ""
    assert deltas == []
    assert not outcome.interrupted


def test_cancel_during_stream_keeps_partial_reply_marked_interrupted():
    cancel = threading.Event()
    api = FakeAPI([_chunk("one "), _chunk("two "), _chunk("three")])
    client = LLMClient(_config(), _registry(), api=api)
    messages = _history()

    def on_delta(chunk: str) -> None:
        cancel.set()

    outcome = client.stream_turn(messages, "count", cancel=cancel, on_delta=on_delta)

    assert outcome.interrupted
    assert outcome.reply == "one"
    assert outcome.messages[-1]["content"] == f"one {INTERRUPTED_MARK}"
    assert messages == _history()


def test_cancel_before_start_makes_no_request():
    cancel = threading.Event()
    cancel.set()
    api = FakeAPI()
    client = LLMClient(_config(), _registry(), api=api)

    outcome = client.stream_turn(_history(), "hi", cancel=cancel)

    assert outcome.interrupted
    assert outcome.reply == ""
    assert api.requests == []
