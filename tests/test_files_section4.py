"""Section 4: find_file and open_file for folders such as Music, and fair search across file roots."""

import json

import pytest

from companion.files import index as index_module
from companion.tools.builtin import files
from companion.tools.registry import default_registry


def run(tool_name: str, /, **arguments) -> str:
    return default_registry.execute(tool_name, json.dumps(arguments))


@pytest.fixture
def roots(tmp_path, monkeypatch):
    first = tmp_path / "Documents"
    second = tmp_path / "Music"
    first.mkdir()
    second.mkdir()
    monkeypatch.setattr(files, "_roots", lambda: [first.resolve(), second.resolve()])
    return first, second


def test_find_file_returns_the_music_folder_first(roots):
    documents, music = roots
    (documents / "music-theory.txt").write_text("x")
    (music / "Album").mkdir()

    lines = run("find_file", name="music").splitlines()

    assert lines[0] == f"1. {music.resolve()} (folder)"
    assert any("music-theory.txt" in line for line in lines)


def test_find_file_is_not_hidden_by_a_large_first_root(roots, monkeypatch):
    documents, music = roots
    monkeypatch.setattr(index_module, "MAX_ENTRIES_PER_ROOT", 3)
    for index in range(5):
        (documents / f"song {index}.txt").write_text("x")
    (music / "song final.mp3").write_text("x")

    reply = run("find_file", name="song", limit=10)

    assert "song final.mp3" in reply


def test_music_root_is_found_even_when_the_first_root_uses_its_scan_budget(roots, monkeypatch):
    documents, music = roots
    monkeypatch.setattr(index_module, "MAX_ENTRIES_PER_ROOT", 2)
    for index in range(10):
        (documents / f"clip {index}.mp4").write_text("x")

    lines = run("find_file", name="music", limit=5).splitlines()

    assert lines[0] == f"1. {music.resolve()} (folder)"


def test_find_file_limit_still_applies(roots):
    documents, _ = roots
    for index in range(4):
        (documents / f"report {index}.txt").write_text("x")

    lines = run("find_file", name="report", limit=2).splitlines()

    assert len(lines) == 2


def test_folder_inside_a_file_root_can_be_opened_after_checks(roots):
    _, music = roots
    album = music / "Album"
    album.mkdir()

    assert files._checked_file(str(music)) == music.resolve()
    assert files._checked_file(str(album)) == album.resolve()


def test_folder_outside_the_file_roots_is_refused(roots, tmp_path):
    outside = tmp_path / "Secrets"
    outside.mkdir()

    assert "outside your file folders" in files._checked_file(str(outside))


def test_executable_inside_a_folder_is_still_refused(roots):
    _, music = roots
    script = music / "run.ps1"
    script.write_text("x")

    assert "can run code" in files._checked_file(str(script))


def test_open_file_opens_the_folder_when_allowed(roots, monkeypatch):
    _, music = roots
    opened = []
    monkeypatch.setattr(files, "_open", lambda path: opened.append(path))

    target = files._checked_file(str(music))
    assert target == music.resolve()
    files._open(target)

    assert opened == [music.resolve()]
