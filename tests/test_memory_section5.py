"""Section 5 memory tests: duplicate-safe saves, code-printed listing, ID-based forgetting, quote
checks for automatic saves, and the memory admin CLI. The LLM is faked.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from types import SimpleNamespace

from companion import memory_admin
from companion.conversation import Conversation
from companion.conversation_log import ConversationLog
from companion.memory.extract import parse_extraction
from companion.memory.guards import normalize
from companion.memory.intent import parse_id_spec, parse_intent
from companion.memory.store import MemoryStore


class FakeLLM:
    def __init__(self, reply: str = "Sure.", complete_reply: str = '{"facts": []}') -> None:
        self.reply = reply
        self.complete_reply = complete_reply
        self.calls: list[str] = []
        self.completions: list[list[dict]] = []

    def stream_turn(self, messages, text, cancel=None, on_delta=None, **kwargs):
        self.calls.append(text)
        history = [*messages, {"role": "user", "content": text}]
        history.append({"role": "assistant", "content": self.reply})
        return SimpleNamespace(messages=history, reply=self.reply, interrupted=False)

    def complete(self, messages, *, max_tokens: int = 300) -> str:
        self.completions.append(messages)
        return self.complete_reply

    def context_window(self) -> int:
        return 8192


def _store(path) -> MemoryStore:
    return MemoryStore(path)


def _conv(tmp_path, llm: FakeLLM, store: MemoryStore, *, auto: bool = False) -> Conversation:
    return Conversation(
        llm=llm,
        system_prompt=lambda: "PERSONA",
        memory=store,
        log=ConversationLog(tmp_path / "conv", enabled=False),
        model="qwen3-8b",
        auto_save=auto,
        announce=None,
        clock=time.monotonic,
        pending_timeout_s=120.0,
    )


def _say(conv: Conversation, history, text: str):
    return conv.stream(history, text, threading.Event(), lambda _chunk: None)


# ----- store: duplicate-safe saves -----


def test_add_returns_the_existing_memory_for_the_same_normalized_text(tmp_path):
    store = _store(tmp_path / "m.sqlite")
    first = store.add("User lives in Leeds.")
    second = store.add("user lives in leeds")

    assert second.id == first.id
    assert store.count() == 1
    store.close()


def test_update_into_an_existing_text_removes_the_updated_row(tmp_path):
    store = _store(tmp_path / "m.sqlite")
    leeds = store.add("User lives in Leeds.")
    york = store.add("User lives in York.")

    kept = store.update(york.id, "user lives in leeds!", source="auto")

    assert kept is not None and kept.id == leeds.id
    assert store.get(york.id) is None
    assert store.count() == 1
    store.close()


def test_existing_for_and_get_many(tmp_path):
    store = _store(tmp_path / "m.sqlite")
    a = store.add("My dog is called Biscuit")
    b = store.add("I like tea")

    assert store.existing_for("my dog is called biscuit!").id == a.id
    assert store.existing_for("I like coffee") is None
    assert [m.id for m in store.get_many([b.id, a.id, 999])] == [a.id, b.id]
    store.close()


def test_unique_index_is_created_on_a_fresh_database(tmp_path):
    store = _store(tmp_path / "m.sqlite")
    assert store.has_norm_index()
    store.close()


def test_backup_writes_a_readable_copy(tmp_path):
    store = _store(tmp_path / "m.sqlite")
    store.add("User lives in Leeds.")
    copy = store.backup(tmp_path / "backups")
    store.close()

    db = sqlite3.connect(str(copy))
    try:
        assert db.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1
    finally:
        db.close()


# ----- intent: ID specs and command routing -----


def test_parse_id_spec_accepts_common_spoken_and_typed_forms():
    assert parse_id_spec("3") == [3]
    assert parse_id_spec("memory 3") == [3]
    assert parse_id_spec("#3") == [3]
    assert parse_id_spec("number three") == [3]
    assert parse_id_spec("3-5") == [3, 4, 5]
    assert parse_id_spec("3 to 5") == [3, 4, 5]
    assert parse_id_spec("2, 4 and 6") == [2, 4, 6]
    assert parse_id_spec("twenty three") == [23]


def test_parse_id_spec_rejects_text_and_bad_ranges():
    assert parse_id_spec("garden") is None
    assert parse_id_spec("") is None
    assert parse_id_spec("5-2") is None


def test_remove_with_a_memory_number_is_a_forget_command():
    intent = parse_intent("Can you remove memory 3?")
    assert intent is not None and intent.kind == "forget_ids" and intent.ids == (3,)

    intent = parse_intent("Remove memory entry number 3")
    assert intent is not None and intent.kind == "forget_ids" and intent.ids == (3,)

    intent = parse_intent("forget 2-4")
    assert intent is not None and intent.ids == (2, 3, 4)


def test_remove_without_memory_wording_is_not_hijacked():
    assert parse_intent("remove the milk from my shopping list") is None
    assert parse_intent("delete the garden note") is None


def test_forget_by_words_still_routes_to_text_search():
    intent = parse_intent("delete my memory about the garden")
    assert intent is not None
    assert intent.kind == "forget_target"


def test_list_requests_are_recognized():
    assert parse_intent("list my memories").kind == "list_memories"
    assert parse_intent("What do you remember?").kind == "list_memories"


# ----- extraction: quotes from the user are required -----


def test_extraction_keeps_only_facts_quoted_from_the_user():
    raw = (
        '{"facts": ['
        '{"action": "add", "id": null, "text": "User works as a nurse in Leeds.", '
        '"quote": "i WORK as a nurse in Leeds"},'
        '{"action": "add", "id": null, "text": "User is a pilot.", "quote": "I am a pilot"}'
        "]}"
    )
    facts = parse_extraction(raw, set(), 200, "I work as a nurse in Leeds")

    assert [fact.text for fact in facts] == ["User works as a nurse in Leeds."]


def test_extraction_without_a_quote_field_saves_nothing():
    raw = '{"facts": [{"action": "add", "id": null, "text": "User is a nurse."}]}'
    assert parse_extraction(raw, set(), 200, "I am a nurse") == []


# ----- conversation: commands answered in code -----


def test_list_is_printed_by_code_with_real_numbers(tmp_path):
    store = _store(tmp_path / "m.sqlite")
    a = store.add("My dog is called Biscuit")
    b = store.add("I like tea")
    llm = FakeLLM()
    conv = _conv(tmp_path, llm, store)

    outcome = _say(conv, conv.new_history(), "list my memories")

    assert outcome.reply.startswith("2 saved memories:")
    assert f"[{a.id}] (manual) My dog is called Biscuit" in outcome.reply
    assert f"[{b.id}] (manual) I like tea" in outcome.reply
    assert llm.calls == []
    store.close()


def test_forget_ids_reports_missing_numbers_and_deletes_only_after_yes(tmp_path):
    store = _store(tmp_path / "m.sqlite")
    a = store.add("My dog is called Biscuit")
    conv = _conv(tmp_path, FakeLLM(), store)

    first = _say(conv, conv.new_history(), f"forget {a.id} and 99")
    assert first.reply.startswith("There is no memory 99. Forget this memory:")
    assert store.count() == 1

    second = _say(conv, first.messages, "yes")
    assert second.reply == "Forgot 1 memory."
    assert store.count() == 0
    store.close()


def test_forget_with_only_missing_numbers_stages_nothing(tmp_path):
    store = _store(tmp_path / "m.sqlite")
    conv = _conv(tmp_path, FakeLLM(), store)

    outcome = _say(conv, conv.new_history(), "forget 42")

    assert outcome.reply == "There is no memory 42."
    store.close()


def test_repeated_remember_reports_the_existing_memory(tmp_path):
    store = _store(tmp_path / "m.sqlite")
    conv = _conv(tmp_path, FakeLLM(), store)

    first = _say(conv, conv.new_history(), "remember my dog is called Biscuit")
    second = _say(conv, first.messages, "remember my dog is called biscuit")

    assert first.reply == "Saved: my dog is called Biscuit"
    assert second.reply.startswith("Already saved as memory ")
    assert store.count() == 1
    store.close()


def test_auto_save_does_not_duplicate_under_different_wording(tmp_path):
    store = _store(tmp_path / "m.sqlite")
    store.add("User lives in Leeds.")
    extraction = (
        '{"facts": [{"action": "add", "id": null, "text": "User Lives In Leeds.", '
        '"quote": "I live in Leeds"}]}'
    )
    conv = _conv(tmp_path, FakeLLM(complete_reply=extraction), store, auto=True)

    _say(conv, conv.new_history(), "I live in Leeds")
    conv.wait_background(timeout=5)

    assert store.count() == 1
    store.close()


# ----- admin CLI -----


def _duplicate_database(path):
    store = MemoryStore(path)
    store.add("User lives in Leeds.")
    store.close()
    db = sqlite3.connect(str(path))
    db.execute("DROP INDEX IF EXISTS idx_memories_norm")
    text = "User lives in Leeds."
    db.execute(
        "INSERT INTO memories (text, created, updated, source, norm) "
        "VALUES (?, '2026-01-02', '2026-01-02', 'manual', ?)",
        (text, normalize(text)),
    )
    db.commit()
    db.close()


def test_admin_dedupe_dry_run_changes_nothing(tmp_path, capsys):
    path = tmp_path / "m.sqlite"
    _duplicate_database(path)

    assert memory_admin.main(["dedupe"], db_path=path) == 0

    assert "Dry run" in capsys.readouterr().out
    store = MemoryStore(path)
    assert store.count() == 2
    store.close()


def test_admin_dedupe_apply_backs_up_and_keeps_the_oldest(tmp_path, capsys):
    path = tmp_path / "m.sqlite"
    _duplicate_database(path)

    assert memory_admin.main(["dedupe", "--apply"], db_path=path) == 0

    assert list((tmp_path / "backups").glob("memories-*.sqlite"))
    store = MemoryStore(path)
    assert store.count() == 1
    assert store.has_norm_index()
    store.close()


def test_admin_forget_dry_run_then_apply(tmp_path, capsys):
    path = tmp_path / "m.sqlite"
    store = MemoryStore(path)
    a = store.add("first note")
    store.add("second note")
    store.close()

    assert memory_admin.main(["forget", str(a.id)], db_path=path) == 0
    assert "Dry run" in capsys.readouterr().out
    store = MemoryStore(path)
    assert store.count() == 2
    store.close()

    assert memory_admin.main(["forget", str(a.id), "--apply"], db_path=path) == 0
    store = MemoryStore(path)
    assert store.count() == 1
    store.close()


def test_admin_rejects_a_bad_spec(tmp_path, capsys):
    path = tmp_path / "m.sqlite"
    assert memory_admin.main(["forget", "garden"], db_path=path) == 2
    assert "is not a list of memory numbers" in capsys.readouterr().out
