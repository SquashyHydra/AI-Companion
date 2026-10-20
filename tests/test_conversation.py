"""Phase 5 tests: persona loading, conversation log, long-term memory, history budget, and the
conversation service that ties them to the LLM. The LLM is faked, so no server is needed.
"""

from __future__ import annotations

import threading
from datetime import datetime
from types import SimpleNamespace

from companion.conversation import Conversation, last_reply
from companion.conversation_log import ConversationLog
from companion.llm.context import HistoryCompactor, messages_tokens
from companion.llm.prompts import DEFAULT_PERSONA, build_system_prompt, load_persona
from companion.memory.store import MemoryStore


class FakeLLM:
    """Records stream_turn calls and returns a canned assistant reply."""

    def __init__(self, reply: str = "Sure.", interrupted: bool = False, window: int = 8192):
        self.reply = reply
        self.interrupted = interrupted
        self.window = window
        self.calls: list[dict] = []

    def stream_turn(self, messages, text, cancel=None, on_delta=None, **kwargs):
        self.calls.append({"messages": messages, "text": text, **kwargs})
        history = [*messages, {"role": "user", "content": text}]
        if not self.interrupted:
            history.append({"role": "assistant", "content": self.reply})
        return SimpleNamespace(
            messages=history,
            reply=self.reply,
            interrupted=self.interrupted,
        )

    def context_window(self) -> int:
        return self.window


def _conversation(tmp_path, llm=None, memory=None, log_enabled=True):
    log = ConversationLog(
        tmp_path / "conversations",
        enabled=log_enabled,
        clock=lambda: datetime(2024, 1, 2, 3, 4, 5),
    )
    return Conversation(
        llm=llm or FakeLLM(),
        system_prompt=lambda: "PERSONA TEXT",
        memory=memory,
        log=log,
        model="qwen3-8b",
    )


# ----- persona -----


def test_missing_persona_file_falls_back_to_default(tmp_path):
    assert load_persona(tmp_path / "missing.md") == DEFAULT_PERSONA


def test_blank_persona_file_falls_back_to_default(tmp_path):
    path = tmp_path / "persona.md"
    path.write_text("   \n", encoding="utf-8")
    assert load_persona(path) == DEFAULT_PERSONA


def test_persona_file_text_is_used_and_rules_come_first(tmp_path):
    path = tmp_path / "persona.md"
    path.write_text("You are Ada, a dry-witted librarian.", encoding="utf-8")

    prompt = build_system_prompt(load_persona(path))

    assert "You are Ada, a dry-witted librarian." in prompt
    assert prompt.index("Persona:") > 0
    assert "take priority over the persona" in prompt


# ----- conversation log -----


def test_log_file_is_created_on_first_turn_and_appended(tmp_path):
    log = ConversationLog(tmp_path / "conv", clock=lambda: datetime(2024, 1, 2, 3, 4, 5))
    assert log.path is None

    log.write_turn("hello", "hi there")
    log.write_turn("again", "yes")

    assert log.path is not None
    assert log.path.name == "conversation-20240102-030405.txt"
    text = log.path.read_text(encoding="utf-8")
    assert "You: hello\nCompanion: hi there" in text
    assert "You: again\nCompanion: yes" in text


def test_log_start_new_opens_a_new_file(tmp_path):
    times = iter([datetime(2024, 1, 2, 3, 4, 5), datetime(2024, 1, 2, 3, 9, 0)])
    log = ConversationLog(tmp_path / "conv", clock=lambda: next(times))
    log.write_turn("one", "a")
    first = log.path

    log.start_new()
    log.write_turn("two", "b")

    assert log.path != first
    assert log.path.exists()


def test_disabled_log_writes_nothing(tmp_path):
    log = ConversationLog(tmp_path / "conv", enabled=False)
    log.write_turn("hello", "hi")
    assert not (tmp_path / "conv").exists()


# ----- long-term memory -----


def test_memory_add_search_and_delete(tmp_path):
    store = MemoryStore(tmp_path / "memories.sqlite")
    saved = store.add("My dog is called Biscuit")

    assert [m.text for m in store.search("what is my dog called", limit=5)] == [
        "My dog is called Biscuit"
    ]
    assert store.search("weather in Paris", limit=5) == []
    assert store.delete([saved.id]) == 1
    assert store.all() == []


def test_memory_block_lists_matches_and_is_none_when_nothing_matches(tmp_path):
    store = MemoryStore(tmp_path / "memories.sqlite")
    store.add("My dog is called Biscuit")

    block = store.block_for("what is my dog called", limit=5, max_chars=300)
    assert block is not None
    assert "Biscuit" in block
    assert store.block_for("tell me a joke", limit=5, max_chars=300) is None


def test_memory_text_is_kept_to_one_line(tmp_path):
    store = MemoryStore(tmp_path / "memories.sqlite")
    saved = store.add("line one\n\n   line two")
    assert saved.text == "line one line two"


def test_forget_all_and_by_number(tmp_path):
    store = MemoryStore(tmp_path / "memories.sqlite")
    first = store.add("alpha fact")
    store.add("beta fact")

    assert [m.id for m in store.find_ids(f"#{first.id}")] == [first.id]
    assert len(store.find_ids("all")) == 2


# ----- history budget -----


def test_compactor_is_due_only_when_history_passes_the_budget():
    history = [{"role": "system", "content": "sys"}]
    for i in range(12):
        history += [
            {"role": "user", "content": f"question {i} " * 20},
            {"role": "assistant", "content": f"answer {i} " * 20},
        ]
    tokens = messages_tokens(history)

    small = HistoryCompactor(
        lambda text: "summary",
        window_tokens=lambda: tokens * 10,
        overhead_tokens=lambda: 0,
        compact_at=0.7,
        keep_recent_turns=2,
        max_summary_chars=500,
    )
    tight = HistoryCompactor(
        lambda text: "summary",
        window_tokens=lambda: tokens // 2,
        overhead_tokens=lambda: 0,
        compact_at=0.7,
        keep_recent_turns=2,
        max_summary_chars=500,
    )

    assert small.due(history) is False
    assert tight.due(history) is True


# ----- conversation service -----


def test_new_history_uses_the_current_persona(tmp_path):
    conversation = _conversation(tmp_path)
    history = conversation.new_history()
    assert history == [{"role": "system", "content": "PERSONA TEXT"}]


def test_turn_passes_recalled_memory_and_writes_the_log(tmp_path):
    store = MemoryStore(tmp_path / "memories.sqlite")
    store.add("My dog is called Biscuit")
    llm = FakeLLM(reply="Biscuit, of course.")
    conversation = _conversation(tmp_path, llm=llm, memory=store)
    history = conversation.new_history()

    outcome = conversation.stream(history, "what is my dog called", threading.Event(), print)
    conversation.after_turn(outcome.messages)

    assert "Biscuit" in llm.calls[0]["extra_system"]
    assert last_reply(outcome.messages) == "Biscuit, of course."
    log_text = conversation._log.path.read_text(encoding="utf-8")
    assert "You: what is my dog called" in log_text
    assert "Companion: Biscuit, of course." in log_text


def test_interrupted_turn_is_not_logged(tmp_path):
    llm = FakeLLM(interrupted=True)
    conversation = _conversation(tmp_path, llm=llm)
    history = conversation.new_history()

    conversation.stream(history, "hello", threading.Event(), print)

    assert conversation._log.path is None


def test_status_reports_model_context_and_memory_count(tmp_path):
    store = MemoryStore(tmp_path / "memories.sqlite")
    store.add("a fact")
    conversation = _conversation(tmp_path, memory=store)
    history = conversation.new_history()

    text = conversation.status(history)

    assert "Model: qwen3-8b" in text
    assert "of 8192 tokens" in text
    assert "Saved memories: 1." in text


def test_no_memory_store_means_no_memory_block(tmp_path):
    llm = FakeLLM()
    conversation = _conversation(tmp_path, llm=llm, memory=None)
    conversation.stream(conversation.new_history(), "hi", threading.Event(), print)
    assert llm.calls[0]["extra_system"] is None
