"""Tests for timer persistence: the JSON store, restore at startup, and missed-timer announcements."""

import json
import time

import pytest

from companion.notices import notices
from companion.tools.builtin import timers
from companion.tools.registry import default_registry


def run(tool_name: str, /, **arguments) -> str:
    return default_registry.execute(tool_name, json.dumps(arguments))


def _forget_in_memory() -> None:
    """Simulate a restart: drop running timers from memory without touching the store file."""
    with timers._lock:
        for item in timers._scheduled.values():
            item.handle.cancel()
        timers._scheduled.clear()


@pytest.fixture
def store(tmp_path, monkeypatch):
    path = tmp_path / "timers.json"
    monkeypatch.setattr(timers, "_next_id", 1)
    monkeypatch.setattr(timers, "_store_path", None)
    monkeypatch.setattr(timers, "_announce_missed", True)
    notices.drain()
    yield path
    _forget_in_memory()
    notices.drain()


def _saved(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_no_store_means_nothing_is_written(store, tmp_path):
    assert timers.configure_store(None) == 0
    run("set_timer", minutes=5, label="tea")
    _forget_in_memory()
    assert not any(tmp_path.iterdir())


def test_new_timer_is_saved_and_removed_on_cancel(store):
    assert timers.configure_store(store) == 0
    run("set_timer", minutes=5, label="tea")
    data = _saved(store)
    assert [item["label"] for item in data["items"]] == ["tea"]
    assert data["items"][0]["due"] > time.time() + 290

    timer_id = data["items"][0]["id"]
    run("cancel_timer", timer_id=timer_id)
    assert _saved(store)["items"] == []


def test_fired_timer_is_removed_from_the_store(store):
    timers.configure_store(store)
    run("set_timer", minutes=5, label="tea")
    timer_id = _saved(store)["items"][0]["id"]
    timers._fire(timer_id)
    assert _saved(store)["items"] == []
    assert notices.drain() == ["Timer done: tea."]


def test_running_timer_is_restored_after_restart(store):
    timers.configure_store(store)
    run("set_timer", minutes=5, label="laundry")
    _forget_in_memory()

    assert timers.configure_store(store) == 1
    listing = run("list_timers")
    assert "timer 'laundry'" in listing
    assert notices.drain() == []


def test_missed_timer_is_announced_once_then_dropped(store):
    store.write_text(
        json.dumps(
            {
                "version": 1,
                "next_id": 7,
                "items": [{"id": 6, "kind": "timer", "label": "tea", "due": time.time() - 30}],
            }
        ),
        encoding="utf-8",
    )
    assert timers.configure_store(store) == 0
    announced = notices.drain()
    assert len(announced) == 1
    assert announced[0].startswith("Timer missed while the companion was closed: tea (due ")

    _forget_in_memory()
    assert timers.configure_store(store) == 0
    assert notices.drain() == []


def test_missed_reminder_uses_reminder_wording(store):
    store.write_text(
        json.dumps(
            {
                "next_id": 2,
                "items": [{"id": 1, "kind": "reminder", "label": "call mum", "due": time.time() - 5}],
            }
        ),
        encoding="utf-8",
    )
    timers.configure_store(store)
    assert notices.drain()[0].startswith("Reminder missed while the companion was closed: call mum")


def test_missed_timer_is_silent_when_announce_is_off(store):
    store.write_text(
        json.dumps({"next_id": 2, "items": [{"id": 1, "kind": "timer", "label": "tea", "due": 0}]}),
        encoding="utf-8",
    )
    timers.configure_store(store, announce_missed=False)
    assert notices.drain() == []
    assert _saved(store)["items"] == []


def test_restored_ids_continue_from_saved_counter(store):
    store.write_text(json.dumps({"next_id": 42, "items": []}), encoding="utf-8")
    timers.configure_store(store)
    reply = run("set_timer", minutes=1, label="tea")
    assert reply.startswith("Timer 42 ")


def test_unreadable_store_starts_empty_and_does_not_crash(store):
    store.write_text("{not json", encoding="utf-8")
    assert timers.configure_store(store) == 0
    assert "No timers or reminders are running." == run("list_timers")


def test_bad_entries_are_skipped_and_good_ones_kept(store):
    store.write_text(
        json.dumps(
            {
                "next_id": 5,
                "items": [
                    {"id": 1, "kind": "alarm", "label": "x", "due": time.time() + 600},
                    {"id": 2, "kind": "timer", "label": "ok", "due": time.time() + 600},
                    {"kind": "timer"},
                ],
            }
        ),
        encoding="utf-8",
    )
    assert timers.configure_store(store) == 1
    assert "timer 'ok'" in run("list_timers")
