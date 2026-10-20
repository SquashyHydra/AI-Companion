"""Tests for the app allowlist and the open_app and list_apps tools."""

import json

import pytest

from companion.safety.allowlist import is_valid_target, load_allowlist
from companion.tools.builtin import apps
from companion.tools.registry import default_registry


@pytest.fixture
def apps_file(tmp_path, monkeypatch):
    path = tmp_path / "apps.json"
    path.write_text(
        json.dumps(
            {
                "Notepad": "notepad.exe",
                "Witcher": "steam://rungameid/292030",
                "bad entry": "rm -rf /",
                "relative": "..\\evil.exe",
                "Calc": "calc.exe",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(apps, "_apps_file", lambda: path)
    return path


@pytest.fixture
def launched(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(apps, "_launch", calls.append)
    return calls


def test_valid_targets():
    assert is_valid_target("notepad.exe")
    assert is_valid_target("steam://rungameid/292030")
    assert is_valid_target("C:\\Games\\game.exe")
    assert is_valid_target("C:\\Users\\me\\Desktop\\Game.lnk")


def test_invalid_targets():
    assert not is_valid_target("rm -rf /")
    assert not is_valid_target("..\\evil.exe")
    assert not is_valid_target("cmd.com")
    assert not is_valid_target("steam://rungameid/abc")
    assert not is_valid_target("https://example.com")


def test_load_allowlist_skips_invalid_entries(apps_file):
    assert load_allowlist(apps_file) == {
        "notepad": "notepad.exe",
        "witcher": "steam://rungameid/292030",
        "calc": "calc.exe",
    }


def test_missing_allowlist_is_empty(tmp_path):
    assert load_allowlist(tmp_path / "none.json") == {}


def test_malformed_allowlist_raises(tmp_path):
    path = tmp_path / "apps.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_allowlist(path)


def test_list_apps_shows_allowlisted_names(apps_file):
    result = default_registry.execute("list_apps", "{}")
    assert result == "I can open: calc, notepad, witcher."


def test_open_app_launches_allowlisted_name(apps_file, launched):
    result = default_registry.execute("open_app", '{"name": "NOTEPAD"}')
    assert result == "Opened notepad."
    assert launched == ["notepad.exe"]


def test_open_app_refuses_unknown_name(apps_file, launched):
    result = default_registry.execute("open_app", '{"name": "powershell"}')
    assert result.startswith("Error: 'powershell' is not on the app allowlist")
    assert launched == []


def test_open_app_refuses_entries_that_failed_validation(apps_file, launched):
    result = default_registry.execute("open_app", '{"name": "bad entry"}')
    assert result.startswith("Error: 'bad entry' is not on the app allowlist")
    assert launched == []


def test_open_app_reports_missing_file(tmp_path, monkeypatch, launched):
    missing = tmp_path / "missing.exe"
    path = tmp_path / "apps.json"
    path.write_text(json.dumps({"ghost": str(missing)}), encoding="utf-8")
    monkeypatch.setattr(apps, "_apps_file", lambda: path)

    result = default_registry.execute("open_app", '{"name": "ghost"}')
    assert result.startswith("Error: the file for 'ghost' was not found")
    assert launched == []
