"""Tests for the Phase 6 tools: timers, media keys, file search, web search, startup, and the wake word gate."""

import json

import numpy as np
import pytest

from companion import startup
from companion.audio.wakeword import WakeGate
from companion.config import ConfigError, _file_roots, _wake_word_config
from companion.notices import notices
from companion.tools.builtin import files, media, timers, web
from companion.tools.registry import default_registry

FRAME = np.zeros(512, dtype=np.float32)


def run(tool_name: str, /, **arguments) -> str:
    return default_registry.execute(tool_name, json.dumps(arguments))


def test_phase6_tools_are_registered():
    names = {t.name for t in default_registry.tools}
    expected = {
        "set_timer", "set_reminder", "list_timers", "cancel_timer",
        "volume", "media", "find_file", "open_file", "web_search",
    }
    assert expected <= names


# Timers

@pytest.fixture
def clean_timers():
    notices.drain()
    yield
    with timers._lock:
        for item in timers._scheduled.values():
            item.handle.cancel()
        timers._scheduled.clear()
    notices.drain()


def test_set_timer_then_fire_posts_notice(clean_timers):
    reply = run("set_timer", minutes=2, label="  tea   time ")
    assert reply.startswith("Timer ") and "2 minutes: tea time." in reply
    timer_id = next(iter(timers._scheduled))
    timers._fire(timer_id)
    assert notices.drain() == ["Timer done: tea time."]
    assert timer_id not in timers._scheduled


def test_fire_after_cancel_does_nothing(clean_timers):
    run("set_timer", minutes=2, label="tea")
    timer_id = next(iter(timers._scheduled))
    assert "Cancelled timer" in run("cancel_timer", timer_id=timer_id)
    timers._fire(timer_id)
    assert notices.drain() == []


def test_reminder_uses_its_own_wording(clean_timers):
    run("set_reminder", at="18:30", label="call mum")
    timer_id = next(iter(timers._scheduled))
    timers._fire(timer_id)
    assert notices.drain() == ["Reminder: call mum."]


@pytest.mark.parametrize("minutes", [0, -1, 2000])
def test_timer_minutes_out_of_range_are_refused(clean_timers, minutes):
    assert run("set_timer", minutes=minutes).startswith("Error:")
    assert not timers._scheduled


@pytest.mark.parametrize("at", ["25:00", "12:60", "noon", "7pm"])
def test_reminder_bad_clock_time_is_refused(clean_timers, at):
    assert run("set_reminder", at=at).startswith("Error:")
    assert not timers._scheduled


def test_cancel_unknown_number_is_an_error(clean_timers):
    assert run("cancel_timer", timer_id=999).startswith("Error:")


def test_list_timers_empty_and_running(clean_timers):
    assert run("list_timers") == "No timers or reminders are running."
    run("set_timer", minutes=5, label="laundry")
    listing = run("list_timers")
    assert "timer 'laundry'" in listing


def test_timer_limit_is_enforced(clean_timers):
    for _ in range(timers.MAX_ACTIVE):
        run("set_timer", minutes=5)
    assert run("set_timer", minutes=5).startswith("Error:")


def test_clean_label_collapses_space_and_uses_default():
    assert timers._clean_label("  tea   time  ", "timer") == "tea time"
    assert timers._clean_label("   ", "timer") == "timer"
    assert len(timers._clean_label("x" * 200, "timer")) == timers.MAX_LABEL_CHARS


# Media keys

@pytest.fixture
def presses(monkeypatch):
    sent: list[int] = []
    monkeypatch.setattr(media, "_press", lambda key: sent.append(key))
    return sent


def test_volume_up_sends_one_key_per_step(presses):
    assert run("volume", action="up", steps=3) == "Volume up by 3 step(s)."
    assert presses == [media.VOLUME_KEYS["up"]] * 3


def test_volume_mute_toggles_once(presses):
    assert run("volume", action="mute") == "Toggled mute."
    assert presses == [media.VOLUME_KEYS["mute"]]


def test_volume_steps_out_of_range_are_refused(presses):
    assert "steps must be from 1 to 20" in run("volume", action="down", steps=0)
    assert "steps must be from 1 to 20" in run("volume", action="down", steps=21)
    assert presses == []


def test_media_key_sends_its_code(presses):
    assert run("media", action="next") == "Sent media key: next."
    assert presses == [media.MEDIA_KEYS["next"]]


def test_media_press_failure_is_reported(monkeypatch):
    def broken(_key):
        raise OSError("no keyboard")

    monkeypatch.setattr(media, "_press", broken)
    assert run("media", action="play_pause").startswith("Error: could not send the key press")


# File search

@pytest.fixture
def file_root(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setattr(files, "_roots", lambda: [root.resolve()])
    return root


def test_find_file_matches_name_and_skips_hidden_and_build_folders(file_root):
    (file_root / "Budget 2024.xlsx").write_text("x")
    (file_root / "notes.txt").write_text("x")
    (file_root / ".budget-cache").write_text("x")
    (file_root / "node_modules").mkdir()
    (file_root / "node_modules" / "budget.js").write_text("x")

    reply = run("find_file", name="BUDGET")
    assert "Budget 2024.xlsx" in reply
    assert "budget-cache" not in reply
    assert "node_modules" not in reply
    assert "notes.txt" not in reply


def test_find_file_no_match_and_bad_input(file_root):
    assert run("find_file", name="missing") == "No files with 'missing' in their name."
    assert run("find_file", name="   ").startswith("Error:")
    assert run("find_file", name="a", limit=0).startswith("Error:")


def test_checked_file_accepts_file_inside_root(file_root):
    doc = file_root / "plan.txt"
    doc.write_text("x")
    assert files._checked_file(str(doc)) == doc.resolve()


def test_checked_file_refuses_outside_root(file_root, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("x")
    assert "outside your file folders" in files._checked_file(str(outside))


def test_checked_file_refuses_code_suffixes(file_root):
    script = file_root / "setup.exe"
    script.write_text("x")
    assert "can run code" in files._checked_file(str(script))


def test_checked_file_refuses_relative_and_missing_paths(file_root):
    assert files._checked_file("plan.txt").startswith("Error:")
    assert files._checked_file(str(file_root / "gone.txt")).startswith("Error:")


def test_checked_file_accepts_folder_inside_root(file_root):
    music = file_root / "Music"
    music.mkdir()
    assert files._checked_file(str(music)) == music.resolve()
    assert files._checked_file(str(file_root)) == file_root.resolve()


# Web search (the parser, limits, and privacy rules are tested in test_search_sections.py)


class _RecordingSearch:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def model_search(self, query: str) -> str:
        self.queries.append(query)
        return "service reply"


def test_web_search_tool_delegates_to_the_active_service(monkeypatch):
    fake = _RecordingSearch()
    monkeypatch.setattr(web, "active_search", lambda: fake)
    assert run("web_search", query="weather") == "service reply"
    assert fake.queries == ["weather"]


# Config parsing

def test_wake_word_defaults_are_off():
    cfg = _wake_word_config({})
    assert cfg.enabled is False
    assert cfg.model == "hey_jarvis"
    assert cfg.threshold == 0.5


@pytest.mark.parametrize(
    "raw",
    [{"enabled": "yes"}, {"model": "  "}, {"threshold": 1.5}, {"window_s": 0.5}],
)
def test_wake_word_bad_values_are_refused(raw):
    with pytest.raises(ConfigError):
        _wake_word_config(raw)


def test_file_roots_resolve_relative_paths_against_project():
    roots = _file_roots(["docs"])
    assert roots[0].is_absolute()
    assert roots[0].name == "docs"


def test_file_roots_must_be_a_list_of_strings():
    with pytest.raises(ConfigError):
        _file_roots("docs")


# Startup command

def test_launch_command_prefers_pythonw_next_to_interpreter(tmp_path):
    python = tmp_path / "python.exe"
    pythonw = tmp_path / "pythonw.exe"
    launcher = tmp_path / "launch_companion.pyw"
    python.write_text("")
    pythonw.write_text("")
    assert startup.launch_command(str(python), launcher) == f'"{pythonw}" "{launcher}"'


def test_launch_command_falls_back_to_interpreter(tmp_path):
    python = tmp_path / "python.exe"
    python.write_text("")
    launcher = tmp_path / "launch_companion.pyw"
    assert startup.launch_command(str(python), launcher) == f'"{python}" "{launcher}"'


# Wake word gate

class FakeWake:
    """Reports a hit on the frames whose 1-based numbers are in hits."""

    def __init__(self, hits: set[int]) -> None:
        self._hits = hits
        self._count = 0
        self.resets = 0

    def hit(self, _frame) -> bool:
        self._count += 1
        return self._count in self._hits

    def reset(self) -> None:
        self.resets += 1


def _gate(hits: set[int], window_s: float = 0.2, reset_detect=None):
    seen: list[int] = []

    def detect(_frame) -> float:
        seen.append(1)
        return 0.9

    wake = FakeWake(hits)
    gate = WakeGate(detect, wake, window_s=window_s, reset_detect=reset_detect)
    return gate, wake, seen


def test_wake_gate_is_closed_until_wake_word_then_opens_for_window():
    gate, _wake, seen = _gate(hits={1})
    outputs = [gate(FRAME) for _ in range(10)]
    # 512 samples is 0.032 s. The window is 0.2 s after the hit, so frames 1 to 7 are open.
    assert outputs[:7] == [0.9] * 7
    assert outputs[7:] == [0.0] * 3
    assert len(seen) == 7


def test_wake_gate_closed_before_any_hit():
    gate, _wake, seen = _gate(hits=set())
    assert [gate(FRAME) for _ in range(5)] == [0.0] * 5
    assert not gate.open
    assert seen == []


def test_second_wake_word_extends_the_window():
    gate, _wake, _seen = _gate(hits={1, 6})
    outputs = [gate(FRAME) for _ in range(14)]
    # Second hit at 0.192 s keeps the window open until about 0.392 s, so frame 12 is still open.
    assert outputs[11] == 0.9
    assert outputs[12] == 0.0


def test_wake_gate_reset_keeps_window_open_and_resets_inner_state():
    detect_resets: list[int] = []
    gate, wake, _seen = _gate(hits={1}, reset_detect=lambda: detect_resets.append(1))
    gate(FRAME)
    gate.reset()
    assert wake.resets == 1
    assert detect_resets == [1]
    assert gate.open
    assert gate(FRAME) == 0.9


def test_wake_gate_rejects_non_positive_window():
    with pytest.raises(ValueError):
        _gate(hits=set(), window_s=0)
