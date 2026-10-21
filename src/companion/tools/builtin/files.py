"""Find files, open them, play music and video, and rescan the file index.

Only files inside the folders listed in tools.file_roots are used. Names are matched against an
index that is built in the background at startup and refreshed on a timer.
"""

from __future__ import annotations

import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from companion.config import get_config
from companion.files.index import EXACT_SCORE, SUBSTRING_SCORE, FileIndex, Match, Sources
from companion.tools.registry import tool

MAX_RESULTS = 25
_PLAYLIST_FOLDER = "companion_playlists"
_PLAYLIST_KEEP_S = 24 * 60 * 60

Candidate = tuple[Match, list[str]]
"""A match plus the media files it would play. A folder's list is its tracks, sorted by path."""


def _roots() -> list[Path]:
    return [root.resolve() for root in get_config().tools.file_roots if root.is_dir()]


def _sources() -> Sources:
    cfg = get_config().files
    return Sources(roots=tuple(_roots()), media_extensions=frozenset(cfg.media_extensions))


_index = FileIndex(source=_sources)


def file_index() -> FileIndex:
    """The shared index. main starts it in the background; diagnose_files reads it."""
    return _index


def _no_roots() -> str:
    return "Error: none of the folders in tools.file_roots exist. Check config.yaml."


@tool(timeout=60)
def find_file(name: str, limit: int = 10) -> str:
    """Find files or folders by name. Searches your file folders (documents, desktop, downloads, videos, and music by default). Misspellings are matched too. Exact and folder matches are listed first.

    Args:
        name: Part of the file or folder name, such as "budget", "invoice", or "beatles". Not case sensitive.
        limit: The most results to return, from 1 to 25.
    """
    needle = " ".join(name.split())
    if not needle:
        return "Error: give part of a file name to search for."
    if not 1 <= limit <= MAX_RESULTS:
        return "Error: limit must be from 1 to 25."
    if not _roots():
        return _no_roots()
    matches = _index.find(needle, limit)
    if not matches:
        return f"No files with '{needle}' in their name."
    lines = []
    for index, match in enumerate(matches, 1):
        suffix = " (folder)" if match.entry.is_folder else ""
        if match.score < SUBSTRING_SCORE:
            suffix += " (close match)"
        lines.append(f"{index}. {match.entry.path}{suffix}")
    return "\n".join(lines)


def _open(path: Path) -> None:
    """Open a file with its default program. Replaced in tests."""
    os.startfile(path)  # type: ignore[attr-defined]


def _checked_file(path: str) -> Path | str:
    """Return the resolved file path, or an error message if it must not be opened."""
    text = path.strip().strip('"')
    if not text:
        return "Error: give the full path of the file, as shown by find_file."
    candidate = Path(text)
    if not candidate.is_absolute():
        return "Error: give the full path of the file, as shown by find_file."
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        return "Error: no file exists at that path."
    if not resolved.is_file() and not resolved.is_dir():
        return "Error: that path is not a file or folder."
    if not any(resolved.is_relative_to(root) for root in _roots()):
        return "Error: that is outside your file folders, so it cannot be opened."
    blocked = set(get_config().files.blocked_extensions)
    if resolved.is_file() and resolved.suffix.lower() in blocked:
        return f"Error: {resolved.suffix} files can run code, so the companion will not open them."
    return resolved


def _open_needs_confirmation(args: dict[str, Any]) -> bool:
    """Media files open without asking unless files.confirm_media is on. Folders and other files ask."""
    cfg = get_config().files
    path = Path(str(args.get("path", "")).strip().strip('"'))
    is_media = path.suffix.lower() in cfg.media_extensions and path.is_file()
    return cfg.confirm_media or not is_media


@tool(requires_confirmation=_open_needs_confirmation)
def open_file(path: str) -> str:
    """Open a file or folder with its default program, such as a document, picture, video, or the Music folder. Folders and documents ask first; music and video do not unless the user turned that on.

    Args:
        path: The full path of the file or folder, as shown by find_file.
    """
    target = _checked_file(path)
    if isinstance(target, str):
        return target
    try:
        _open(target)
    except OSError as exc:
        return f"Error: could not open {target.name}: {exc}"
    return f"Opened {target.name}."


def _media_needs_confirmation(args: dict[str, Any]) -> bool:
    return get_config().files.confirm_media


@tool(requires_confirmation=_media_needs_confirmation, timeout=30)
def play_media(query: str, shuffle: bool = False) -> str:
    """Play a song, video, album, or artist folder by name in the default media player. Searches your file folders. Misspellings are fine. If several things match, the choices are listed and nothing plays until the user picks one.

    Args:
        query: Name of the song, video, album, or artist, such as "Hey Jude" or "Abbey Road".
        shuffle: True to shuffle the tracks of a folder.
    """
    text = " ".join(query.split())
    if not text:
        return "Error: say what to play."
    if not _roots():
        return _no_roots()
    matches = _index.find(text, limit=8, media_only=True)
    candidates: list[Candidate] = []
    for match in matches:
        if match.entry.is_folder:
            tracks = _index.media_under(match.entry.path)
            if tracks:
                candidates.append((match, tracks))
        else:
            candidates.append((match, [match.entry.path]))
    if not candidates:
        return f'Nothing to play matches "{text}".'

    exact = [c for c in candidates if c[0].score == EXACT_SCORE]
    if exact:
        folders = [c for c in exact if c[0].entry.is_folder]
        if len(folders) == 1:
            return _play(folders[0], shuffle)
        if len(exact) == 1:
            return _play(exact[0], shuffle)
        return _choices(exact, text, fuzzy=False)

    substring = [c for c in candidates if c[0].score == SUBSTRING_SCORE]
    if len(substring) == 1:
        return _play(substring[0], shuffle)
    if substring:
        return _choices(substring, text, fuzzy=False)
    return _choices(candidates, text, fuzzy=True)


def _describe(candidate: Candidate) -> str:
    match, tracks = candidate
    if match.entry.is_folder:
        return f"{match.entry.label} (folder, {len(tracks)} tracks) at {match.entry.path}"
    return f"{match.entry.label} at {match.entry.path}"


def _choices(candidates: list[Candidate], text: str, *, fuzzy: bool) -> str:
    shown = "; ".join(f"{i}) {_describe(c)}" for i, c in enumerate(candidates[:3], 1))
    if fuzzy:
        return (
            f'No exact match for "{text}". Did you mean: {shown}? '
            "Say the name again to play one of these."
        )
    return f'More than one match for "{text}": {shown}. Say which one, or give the full name.'


def _play(candidate: Candidate, shuffle: bool) -> str:
    match, tracks = candidate
    label = match.entry.label
    limit = get_config().files.playlist_max_tracks
    order = list(tracks)
    if shuffle:
        random.shuffle(order)
    order = order[:limit]
    note = f" (first {limit} of {len(tracks)} tracks)" if len(tracks) > limit else ""
    try:
        if len(order) == 1:
            _open(Path(order[0]))
            return f"Playing {label}."
        if not _has_playlist_handler():
            _open(Path(order[0]))
            return (
                f"No playlist app is set on this PC, so I opened only the first track of {label}. "
                "Set a default app for .m3u files to play whole folders."
            )
        _open(_write_playlist(order))
    except OSError as exc:
        return f"Error: could not start playback of {label}: {exc}"
    shuffled = ", shuffled" if shuffle else ""
    return f"Playing {label}: {len(order)} tracks{shuffled}{note}."


def _has_playlist_handler() -> bool:
    """True when Windows has a default program registered for .m3u playlists."""
    if sys.platform != "win32":
        return False
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, ".m3u") as key:
            value, _ = winreg.QueryValueEx(key, "")
            return bool(value)
    except OSError:
        return False


def _write_playlist(paths: list[str]) -> Path:
    folder = Path(tempfile.gettempdir()) / _PLAYLIST_FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - _PLAYLIST_KEEP_S
    for old in folder.glob("playlist-*.m3u"):
        try:
            if old.stat().st_mtime < cutoff:
                old.unlink()
        except OSError:
            pass
    target = folder / f"playlist-{time.time_ns()}.m3u"
    target.write_text("#EXTM3U\n" + "\n".join(paths) + "\n", encoding="utf-8")
    return target


@tool(timeout=300)
def rescan_files() -> str:
    """Rescan your file folders now, so new, renamed, or removed files are found. Can take a little while on large folders."""
    if not _roots():
        return _no_roots()
    count = _index.build()
    return f"Rescanned your file folders: {count} files and folders indexed."
