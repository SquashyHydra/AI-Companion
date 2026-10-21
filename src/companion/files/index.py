"""In-memory index of the files and folders under the file roots.

The index is a snapshot. Searches read one snapshot, and rebuilds swap in a new one, so a search
never sees a half-built index. Builds run in a background thread at startup and on a timer.
"""

from __future__ import annotations

import difflib
import logging
import os
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from companion.files.match import FUZZY_MIN, leaf_key, normalize, similarity

logger = logging.getLogger(__name__)

MAX_ENTRIES_PER_ROOT = 200_000
"""Each root is capped separately, so one large folder cannot push other roots out of the index."""
MAX_ENTRIES = 400_000
"""Total cap across all roots."""
PREFILTER_SIZE = 50
"""Fuzzy matching scores at most this many candidates, chosen by shared word count."""
CLOSE_TOKEN_CUTOFF = 0.75
CLOSE_TOKENS_PER_WORD = 5
EXACT_SCORE = 1.0
SUBSTRING_SCORE = 0.9
SKIP_DIRS = frozenset(
    {"node_modules", "appdata", "$recycle.bin", "system volume information", "venv", "__pycache__"}
)
SPEECH_ROOTS = frozenset({"music", "videos"})


@dataclass(frozen=True)
class Sources:
    """What an index is built from. A change in either field triggers a rebuild."""

    roots: tuple[Path, ...]
    media_extensions: frozenset[str]


@dataclass(frozen=True, slots=True)
class Entry:
    path: str
    name: str
    is_folder: bool
    root: str
    depth: int
    """0 for a root folder, 1 for its direct children, and so on."""
    key: str
    """Normalized leaf name used for exact and fuzzy matching."""
    text: str
    """Normalized relative path, used for substring matching."""
    media: bool

    @property
    def label(self) -> str:
        """Name to show a person. File extensions are removed."""
        return self.name if self.is_folder else os.path.splitext(self.name)[0]


@dataclass(frozen=True, slots=True)
class Match:
    entry: Entry
    score: float
    """1.0 for an exact name, 0.9 for a name that contains every word, otherwise a fuzzy score."""


@dataclass(frozen=True)
class _Snapshot:
    sources: Sources | None
    entries: tuple[Entry, ...]
    by_token: dict[str, list[int]]
    vocab: tuple[str, ...]
    built_at: float


def make_entry(
    *,
    path: str,
    name: str,
    is_folder: bool,
    root: str,
    rel_parts: Sequence[str],
    depth: int,
    media_extensions: frozenset[str],
) -> Entry:
    suffix = os.path.splitext(name)[1].lower()
    return Entry(
        path=path,
        name=name,
        is_folder=is_folder,
        root=root,
        depth=depth,
        key=leaf_key(name, is_folder),
        text=normalize(" ".join([*rel_parts, name])),
        media=not is_folder and suffix in media_extensions,
    )


def _walk(root: Path, media_extensions: frozenset[str], limit: int) -> list[Entry]:
    root_text = str(root)
    out = [
        make_entry(
            path=root_text,
            name=root.name or root_text,
            is_folder=True,
            root=root_text,
            rel_parts=(),
            depth=0,
            media_extensions=media_extensions,
        )
    ]
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not _skip_dir(d)]
        rel = () if dirpath == root_text else Path(dirpath).relative_to(root).parts
        children = [(name, True) for name in dirnames] + [(name, False) for name in filenames]
        for name, is_folder in children:
            if len(out) >= limit:
                return out
            if name.startswith("."):
                continue
            out.append(
                make_entry(
                    path=os.path.join(dirpath, name),
                    name=name,
                    is_folder=is_folder,
                    root=root_text,
                    rel_parts=rel,
                    depth=len(rel) + 1,
                    media_extensions=media_extensions,
                )
            )
    return out


def _skip_dir(name: str) -> bool:
    return name.startswith(".") or name.lower() in SKIP_DIRS


def _snapshot(entries: list[Entry], sources: Sources | None) -> _Snapshot:
    by_token: dict[str, list[int]] = defaultdict(list)
    for index, entry in enumerate(entries):
        for token in set(entry.key.split()):
            by_token[token].append(index)
    return _Snapshot(
        sources=sources,
        entries=tuple(entries),
        by_token=dict(by_token),
        vocab=tuple(by_token),
        built_at=time.time(),
    )


class FileIndex:
    """Searchable list of files and folders. Pass source=None for an index that only holds
    entries installed with install(), which tests and diagnostics use."""

    def __init__(self, source: Callable[[], Sources] | None = None) -> None:
        self._source = source
        self._build_lock = threading.RLock()
        self._snap: _Snapshot | None = None
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # Building

    def install(self, entries: Iterable[Entry]) -> None:
        """Replace the index with the given entries. Used by tests and benchmarks."""
        with self._build_lock:
            self._snap = _snapshot(list(entries), None)
            self._ready.set()

    def build(self) -> int:
        """Walk every root and replace the index. Returns the number of entries."""
        if self._source is None:
            return len(self._current().entries)
        with self._build_lock:
            started = time.perf_counter()
            sources = self._source()
            entries: list[Entry] = []
            for root in sources.roots:
                if root.is_dir():
                    entries.extend(_walk(root, sources.media_extensions, MAX_ENTRIES_PER_ROOT))
            entries = entries[:MAX_ENTRIES]
            self._snap = _snapshot(entries, sources)
            self._ready.set()
            elapsed = time.perf_counter() - started
            logger.info("file index built: %d entries in %.2fs", len(entries), elapsed)
            return len(entries)

    def ensure(self) -> _Snapshot:
        """Return a current snapshot, building first if roots or extensions changed or no build exists."""
        if self._source is None:
            return self._current()
        with self._build_lock:
            snap = self._snap
            if snap is None or snap.sources != self._source():
                self.build()
                snap = self._snap
            assert snap is not None
            return snap

    def start(self, refresh_minutes: float) -> None:
        """Build in a background thread, then rebuild every refresh_minutes (0 disables refreshes)."""
        if self._source is None or self._thread is not None:
            return
        interval_s = max(0.0, refresh_minutes) * 60.0

        def run() -> None:
            self._safe_build()
            while interval_s > 0 and not self._stop.wait(interval_s):
                self._safe_build()

        self._thread = threading.Thread(target=run, name="file-index", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def wait_ready(self, timeout_s: float) -> bool:
        """Wait for the first build. Returns False right away if no build was started."""
        if self._snap is not None:
            return True
        if self._thread is None:
            return False
        return self._ready.wait(timeout_s)

    def _safe_build(self) -> None:
        try:
            self.build()
        except Exception:  # keep the refresh thread alive; the next build may succeed
            logger.exception("file index build failed")

    def _current(self) -> _Snapshot:
        return self._snap if self._snap is not None else _snapshot([], None)

    # Searching

    def find(self, query: str, limit: int = 10, *, media_only: bool = False) -> list[Match]:
        """Best matches for a name. media_only keeps folders and media files only."""
        snap = self.ensure()
        needle = normalize(query)
        if not needle or not snap.entries:
            return []
        words = needle.split()
        exact: list[Match] = []
        for entry in snap.entries:
            if media_only and not (entry.is_folder or entry.media):
                continue
            if entry.key == needle:
                exact.append(Match(entry, EXACT_SCORE))
            elif all(word in entry.text for word in words):
                exact.append(Match(entry, SUBSTRING_SCORE))
        if exact:
            return _ranked(exact, limit)
        return _ranked(self._fuzzy(snap, needle, words, media_only), limit)

    def _fuzzy(
        self, snap: _Snapshot, needle: str, words: list[str], media_only: bool
    ) -> list[Match]:
        hits: Counter[int] = Counter()
        for word in words:
            if word in snap.by_token:
                tokens = [word]
            else:
                tokens = difflib.get_close_matches(
                    word, snap.vocab, n=CLOSE_TOKENS_PER_WORD, cutoff=CLOSE_TOKEN_CUTOFF
                )
            for token in tokens:
                for index in snap.by_token[token]:
                    hits[index] += 1
        matches = []
        for index, _count in hits.most_common(PREFILTER_SIZE):
            entry = snap.entries[index]
            if media_only and not (entry.is_folder or entry.media):
                continue
            score = similarity(needle, entry.key)
            if score >= FUZZY_MIN:
                matches.append(Match(entry, round(score, 3)))
        return matches

    def media_under(self, folder: str) -> list[str]:
        """Paths of every media file below a folder, sorted by path."""
        snap = self.ensure()
        prefix = folder.rstrip("\\/").lower() + os.sep
        paths = [e.path for e in snap.entries if e.media and e.path.lower().startswith(prefix)]
        return sorted(paths, key=str.lower)

    def speech_terms(self, limit: int = 30) -> list[str]:
        """Artist and album folder names under the Music and Videos roots, for speech recognition.
        Does not build the index. Returns [] until a build has finished."""
        snap = self._snap
        if snap is None:
            return []
        seen: dict[str, str] = {}
        for entry in snap.entries:
            if not entry.is_folder or entry.depth not in (1, 2):
                continue
            if os.path.basename(entry.root).lower() not in SPEECH_ROOTS:
                continue
            name = " ".join(entry.name.split())
            if len(name) >= 3 and not name.isdigit():
                seen.setdefault(name.lower(), name)
            if len(seen) >= limit:
                break
        return list(seen.values())[:limit]

    def root_counts(self) -> dict[str, int]:
        snap = self.ensure()
        counts: Counter[str] = Counter(entry.root for entry in snap.entries)
        return dict(counts)

    @property
    def built_at(self) -> float | None:
        return self._snap.built_at if self._snap is not None else None

    @property
    def size(self) -> int:
        return len(self._current().entries)


def _ranked(matches: list[Match], limit: int) -> list[Match]:
    matches.sort(key=lambda m: (-m.score, m.entry.depth, m.entry.path.lower()))
    return matches[:limit]
