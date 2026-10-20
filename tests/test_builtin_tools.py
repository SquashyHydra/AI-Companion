"""Tests for the built-in time and notes tools."""

import json
import logging

import pytest

from companion.safety.confirm import TerminalConfirmer
from companion.tools.builtin import notes
from companion.tools.registry import default_registry


@pytest.fixture
def notes_dir(tmp_path, monkeypatch):
    target = tmp_path / "notes"
    monkeypatch.setattr(notes, "_notes_dir", lambda: target)
    return target


def test_builtin_tools_are_registered():
    names = {t.name for t in default_registry.tools}
    assert {"get_time", "take_note", "read_notes", "clear_notes"} <= names


def test_get_time_returns_current_year():
    from datetime import datetime

    result = default_registry.execute("get_time", "{}")
    assert str(datetime.now().year) in result


def test_take_note_then_read_back(notes_dir):
    assert default_registry.execute("take_note", '{"text": "buy milk"}') == (
        "Saved note to topic 'general'."
    )
    default_registry.execute("take_note", '{"text": "call  mum\\nafter lunch", "topic": "todo"}')

    general = default_registry.execute("read_notes", "{}")
    assert general.count("\n") == 0
    assert general.endswith("buy milk")

    todo_file = (notes_dir / "todo.md").read_text(encoding="utf-8")
    assert todo_file.startswith("# todo\n")
    assert "call mum after lunch" in todo_file


def test_read_notes_limit_returns_most_recent(notes_dir):
    for i in range(5):
        notes.take_note(f"note {i}")
    lines = default_registry.execute("read_notes", '{"limit": 2}').splitlines()
    assert [line.split("] ")[1] for line in lines] == ["note 3", "note 4"]


def test_read_notes_without_file_says_none(notes_dir):
    assert default_registry.execute("read_notes", '{"topic": "nothing"}') == (
        "No notes found for topic 'nothing'."
    )


@pytest.mark.parametrize("topic", ["../escape", "a/b", "con", "", "x" * 41])
def test_unsafe_topics_are_refused(notes_dir, topic):
    result = notes.take_note("hello", topic=topic)
    assert result.startswith("Error: topic must be")
    assert not notes_dir.exists()


def test_empty_and_oversized_notes_are_refused(notes_dir):
    assert notes.take_note("   ") == "Error: the note is empty."
    assert notes.take_note("x" * (notes.MAX_NOTE_CHARS + 1)).startswith("Error: the note is longer")


def test_read_limit_out_of_range_is_refused(notes_dir):
    assert notes.read_notes(limit=0).startswith("Error: limit must be")
    assert notes.read_notes(limit=101).startswith("Error: limit must be")


@pytest.fixture
def terminal_answers():
    """Script the terminal confirmer's answers, then restore the default (no confirmer)."""

    def use(*replies: str) -> None:
        queue = list(replies)
        default_registry.set_confirmer(TerminalConfirmer(ask=lambda prompt: queue.pop(0)))

    yield use
    default_registry.set_confirmer(None)


def test_clear_notes_is_refused_without_a_confirmer(notes_dir):
    notes.take_note("keep me")
    assert default_registry.execute("clear_notes", "{}").startswith("Error: the user did not allow")
    assert (notes_dir / "general.md").exists()


@pytest.mark.parametrize(
    ("replies", "cleared"),
    [
        (("yes",), True),
        (("Y",), True),
        (("maybe", "y"), True),
        (("no",), False),
        (("maybe", "nah", "wat"), False),
    ],
)
def test_clear_notes_follows_the_answer(notes_dir, terminal_answers, replies, cleared):
    notes.take_note("first")
    notes.take_note("second")
    terminal_answers(*replies)

    result = default_registry.execute("clear_notes", "{}")
    if cleared:
        assert result == "Cleared 2 note(s) from topic 'general'."
        assert not (notes_dir / "general.md").exists()
    else:
        assert result.startswith("Error: the user did not allow")
        assert (notes_dir / "general.md").read_text(encoding="utf-8").count("- [") == 2


def test_clear_notes_missing_topic_says_nothing_cleared(notes_dir, terminal_answers):
    terminal_answers("y")
    assert default_registry.execute("clear_notes", '{"topic": "empty"}') == (
        "No notes found for topic 'empty'. Nothing was cleared."
    )


def test_clear_notes_decisions_are_audited(notes_dir, terminal_answers, caplog):
    notes.take_note("keep")
    with caplog.at_level(logging.INFO, logger="companion.audit"):
        terminal_answers("no")
        default_registry.execute("clear_notes", "{}")
        terminal_answers("y")
        default_registry.execute("clear_notes", "{}")

    records = [json.loads(r.getMessage()) for r in caplog.records if r.name == "companion.audit"]
    decisions = [r["decision"] for r in records if r.get("event") == "confirmation"]
    statuses = [r["status"] for r in records if r.get("tool") == "clear_notes" and "status" in r]
    assert decisions == ["declined", "approved"]
    assert statuses == ["declined", "ok"]

