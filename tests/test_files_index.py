"""Section 4: file index matching, media playback, confirmation rules, roots, and budgets."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from companion.config import get_config
from companion.files import index as index_module
from companion.files.index import FileIndex, Sources, make_entry
from companion.files.roots import is_known_folder_name, resolve_root, root_problems
from companion.tools.builtin import files
from companion.tools.registry import ToolRegistry

MEDIA = frozenset({".mp3", ".mp4"})


def entry_at(root: Path, rel: str, *, is_folder: bool = False):
    parts = rel.split("/")
    return make_entry(
        path=str(root.joinpath(*parts)),
        name=parts[-1],
        is_folder=is_folder,
        root=str(root),
        rel_parts=parts[:-1],
        depth=len(parts),
        media_extensions=MEDIA,
    )


@pytest.fixture
def music(tmp_path, monkeypatch):
    """A Music root, with a fresh file index that builds from disk on first use."""
    root = tmp_path / "Music"
    root.mkdir()
    monkeypatch.setattr(files, "_roots", lambda: [root])
    monkeypatch.setattr(files, "_index", FileIndex(source=files._sources))
    return root


@pytest.fixture
def playback(music, monkeypatch):
    """Records what would be opened instead of starting a media player."""
    opened: list[Path] = []
    monkeypatch.setattr(files, "_open", lambda path: opened.append(Path(path)))
    monkeypatch.setattr(files, "_has_playlist_handler", lambda: True)
    return opened


def patch_files_config(monkeypatch, **changes) -> None:
    real = get_config()
    cfg = SimpleNamespace(
        files=replace(real.files, **changes), tools=real.tools, audio=real.audio
    )
    monkeypatch.setattr(files, "get_config", lambda: cfg)


class FakeConfirmer:
    def __init__(self, answer: bool) -> None:
        self.answer = answer
        self.asked: list[str] = []

    def confirm(self, question: str) -> bool:
        self.asked.append(question)
        return self.answer


def make_registry(rule, answer: bool):
    registry = ToolRegistry()
    calls: list[int] = []

    def act(x: int) -> str:
        """Act on a number.

        Args:
            x: A number.
        """
        calls.append(x)
        return "done"

    registry.register(act, requires_confirmation=rule)
    confirmer = FakeConfirmer(answer)
    registry.set_confirmer(confirmer)
    return registry, calls, confirmer


# Matching


def test_misspelled_folder_name_is_found_as_a_close_match(music):
    (music / "Abbey Road").mkdir()

    matches = files.file_index().find("abbey rod")

    assert matches
    assert matches[0].entry.name == "Abbey Road"
    assert matches[0].score < index_module.EXACT_SCORE


def test_track_number_and_extension_are_ignored_for_exact_matches(music):
    album = music / "Beatles"
    album.mkdir()
    (album / "01 - Hey Jude.mp3").write_text("x")

    matches = files.file_index().find("hey jude")

    assert matches[0].score == index_module.EXACT_SCORE
    assert matches[0].entry.name == "01 - Hey Jude.mp3"


def test_find_result_lists_exact_matches_before_substring_matches(music):
    (music / "Live at Home.mp3").write_text("x")
    (music / "Live").mkdir()

    matches = files.file_index().find("live")

    assert matches[0].entry.name == "Live"
    assert matches[0].score == index_module.EXACT_SCORE


# Playback


def test_play_media_plays_a_whole_album_as_a_playlist(music, playback):
    album = music / "Abbey Road"
    album.mkdir()
    (album / "01 - Come Together.mp3").write_text("x")
    (album / "02 - Something.mp3").write_text("x")
    (album / "cover.jpg").write_text("x")

    reply = files.play_media("Abbey Road")

    assert reply == "Playing Abbey Road: 2 tracks."
    assert len(playback) == 1
    assert playback[0].suffix == ".m3u"
    lines = playback[0].read_text(encoding="utf-8").splitlines()
    assert lines[0] == "#EXTM3U"
    assert [Path(line).name for line in lines[1:]] == [
        "01 - Come Together.mp3",
        "02 - Something.mp3",
    ]


def test_play_media_plays_one_song_directly(music, playback):
    (music / "Hey Jude.mp3").write_text("x")

    reply = files.play_media("hey jude")

    assert reply == "Playing Hey Jude."
    assert [path.name for path in playback] == ["Hey Jude.mp3"]


def test_play_media_lists_choices_when_several_folders_match_exactly(music, playback):
    for genre in ("Rock", "Jazz"):
        folder = music / genre / "Live"
        folder.mkdir(parents=True)
        (folder / "01 - Track.mp3").write_text("x")

    reply = files.play_media("Live")

    assert reply.startswith('More than one match for "Live"')
    assert playback == []


def test_play_media_suggests_a_close_name_without_playing(music, playback):
    album = music / "Abbey Road"
    album.mkdir()
    (album / "01 - Come Together.mp3").write_text("x")

    reply = files.play_media("abbey rod")

    assert reply.startswith('No exact match for "abbey rod". Did you mean:')
    assert playback == []


def test_play_media_without_a_playlist_app_opens_only_the_first_track(
    music, playback, monkeypatch
):
    monkeypatch.setattr(files, "_has_playlist_handler", lambda: False)
    album = music / "Abbey Road"
    album.mkdir()
    (album / "01 - Come Together.mp3").write_text("x")
    (album / "02 - Something.mp3").write_text("x")

    reply = files.play_media("Abbey Road")

    assert "No playlist app is set" in reply
    assert [path.name for path in playback] == ["01 - Come Together.mp3"]


def test_play_media_respects_the_playlist_track_limit(music, playback, monkeypatch):
    patch_files_config(monkeypatch, playlist_max_tracks=2)
    album = music / "Abbey Road"
    album.mkdir()
    for number in range(1, 4):
        (album / f"0{number} - Track.mp3").write_text("x")

    reply = files.play_media("Abbey Road")

    assert reply == "Playing Abbey Road: 2 tracks (first 2 of 3 tracks)."


def test_play_media_reports_nothing_found(music, playback):
    reply = files.play_media("zzzz qqqq")

    assert reply == 'Nothing to play matches "zzzz qqqq".'
    assert playback == []


# Confirmation rules


def test_media_confirmation_follows_the_confirm_media_setting(monkeypatch):
    patch_files_config(monkeypatch, confirm_media=False)
    assert files._media_needs_confirmation({}) is False

    patch_files_config(monkeypatch, confirm_media=True)
    assert files._media_needs_confirmation({}) is True


def test_open_file_asks_for_folders_but_not_for_media_by_default(music, playback):
    song = music / "Hey Jude.mp3"
    song.write_text("x")

    assert files._open_needs_confirmation({"path": str(music)}) is True
    assert files._open_needs_confirmation({"path": str(song)}) is False


def test_open_file_asks_for_media_when_confirm_media_is_on(music, playback, monkeypatch):
    song = music / "Hey Jude.mp3"
    song.write_text("x")
    patch_files_config(monkeypatch, confirm_media=True)

    assert files._open_needs_confirmation({"path": str(song)}) is True


def test_confirmation_rule_that_says_yes_runs_the_tool_after_approval():
    registry, calls, confirmer = make_registry(lambda args: args["x"] > 1, answer=True)

    reply = registry.execute("act", {"x": 5})

    assert reply == "done"
    assert calls == [5]
    assert len(confirmer.asked) == 1


def test_confirmation_rule_that_says_no_skips_the_question():
    registry, calls, confirmer = make_registry(lambda args: args["x"] > 1, answer=False)

    assert registry.execute("act", {"x": 1}) == "done"
    assert calls == [1]
    assert confirmer.asked == []


def test_confirmation_rule_that_fails_asks_the_user_and_declines():
    def broken(args: dict) -> bool:
        raise RuntimeError("rule bug")

    registry, calls, confirmer = make_registry(broken, answer=False)

    reply = registry.execute("act", {"x": 1})

    assert reply.startswith("Error: the user did not allow")
    assert calls == []
    assert len(confirmer.asked) == 1


# Roots


def test_missing_root_folder_is_reported_with_a_warning(tmp_path):
    problems = root_problems([tmp_path / "Nope"])

    assert problems
    assert "not found" in problems[0]
    assert root_problems([tmp_path]) == []


def test_known_folder_shortcut_resolves_to_an_absolute_folder(tmp_path):
    assert is_known_folder_name("music")

    path = resolve_root("music", tmp_path)

    assert path.is_absolute()
    assert path.name.lower() == "music"


# Index behaviour


def test_rescan_finds_a_file_added_after_the_index_was_built(music):
    (music / "first.mp3").write_text("x")
    assert files.file_index().find("second") == []

    (music / "second.mp3").write_text("x")
    reply = files.rescan_files()

    assert "indexed" in reply
    assert any(m.entry.name == "second.mp3" for m in files.file_index().find("second"))


def test_one_big_root_is_capped_per_root(tmp_path, monkeypatch):
    big = tmp_path / "Big"
    big.mkdir()
    for number in range(10):
        (big / f"clip {number}.mp4").write_text("x")
    monkeypatch.setattr(index_module, "MAX_ENTRIES_PER_ROOT", 3)
    index = FileIndex(source=lambda: Sources(roots=(big,), media_extensions=MEDIA))

    assert index.build() <= 3


def test_speech_terms_lists_artist_and_album_folders_under_music():
    music_root = Path("C:/Music")
    index = FileIndex()
    index.install(
        [
            entry_at(music_root, "Beatles", is_folder=True),
            entry_at(music_root, "Beatles/Abbey Road", is_folder=True),
            entry_at(music_root, "Beatles/Abbey Road/01 - Come Together.mp3"),
        ]
    )

    terms = index.speech_terms()

    assert "Beatles" in terms
    assert "Abbey Road" in terms
    assert not any(term.endswith(".mp3") for term in terms)


def test_speech_terms_is_empty_before_a_build():
    assert FileIndex().speech_terms() == []


def test_search_on_a_large_index_stays_fast():
    root = Path("C:/Music")
    entries = [entry_at(root, f"Album {number}", is_folder=True) for number in range(5000)]
    index = FileIndex()
    index.install(entries)

    started = time.perf_counter()
    matches = index.find("albumm 4321")
    elapsed = time.perf_counter() - started

    assert matches
    assert elapsed < 2.0
