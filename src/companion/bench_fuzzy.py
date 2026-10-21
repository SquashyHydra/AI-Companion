"""File search benchmark: times exact, substring, fuzzy, and miss lookups on a synthetic index.

Run from the project root:

    python -m companion.bench_fuzzy --entries 50000

The report names the similarity scorer in use (rapidfuzz if installed, otherwise difflib), the
index build time, the median time per query, and the cost of one fuzzy word lookup over the
vocabulary. Nothing is changed. Use it to decide whether the search needs a faster scorer.
"""

from __future__ import annotations

import argparse
import difflib
import os
import random
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

from companion.files.index import (
    CLOSE_TOKEN_CUTOFF,
    CLOSE_TOKENS_PER_WORD,
    Entry,
    FileIndex,
    make_entry,
)
from companion.files.match import backend_name

BENCH_ROOT = os.path.join(os.sep, "bench")
MEDIA = frozenset({".mp3", ".flac", ".m4a"})
DEFAULT_ENTRIES = 50_000
DEFAULT_REPEATS = 3
MISS_QUERY = "zzqqxv wwpkrt"
LETTERS = "abcdefghijklmnopqrstuvwxyz"

Timer = Callable[[], float]


@dataclass(frozen=True)
class QueryResult:
    label: str
    query: str
    matches: int
    median_ms: float


@dataclass(frozen=True)
class BenchReport:
    backend: str
    entries: int
    vocabulary: int
    install_ms: float
    queries: list[QueryResult]
    word_lookup_ms: float
    typo_word: str


def _word(rng: random.Random) -> str:
    return "".join(rng.choice(LETTERS) for _ in range(rng.randint(4, 9)))


def make_entries(count: int, seed: int = 7) -> list[Entry]:
    """Synthetic music-style entries: three random words per name, one folder per first word."""
    rng = random.Random(seed)
    entries: list[Entry] = []
    for _ in range(count):
        words = [_word(rng) for _ in range(3)]
        name = " ".join(words) + ".mp3"
        rel = (words[0],)
        entries.append(
            make_entry(
                path=os.path.join(BENCH_ROOT, *rel, name),
                name=name,
                is_folder=False,
                root=BENCH_ROOT,
                rel_parts=rel,
                depth=2,
                media_extensions=MEDIA,
            )
        )
    return entries


def build_queries(entries: list[Entry]) -> tuple[list[tuple[str, str]], str]:
    """Four queries taken from one known entry: exact, substring, fuzzy (one typo), and a miss.
    Returns the (label, query) pairs and the typo word."""
    words = entries[len(entries) // 3].key.split()
    typo_source = words[1]
    typo = typo_source[:1] + ("x" if typo_source[1] != "x" else "y") + typo_source[2:]
    typo_query = " ".join(typo if word == typo_source else word for word in words)
    queries = [
        ("exact", entries[len(entries) // 3].key),
        ("substring", f"{words[0]} {words[2]}"),
        ("fuzzy (1 typo)", typo_query),
        ("miss", MISS_QUERY),
    ]
    return queries, typo


def _median_ms(
    action: Callable[[], object], repeats: int, timer: Timer
) -> float:
    samples: list[float] = []
    for _ in range(repeats):
        start = timer()
        action()
        samples.append((timer() - start) * 1000.0)
    return statistics.median(samples)


def benchmark(
    count: int,
    repeats: int = DEFAULT_REPEATS,
    *,
    timer: Timer = time.perf_counter,
) -> BenchReport:
    """Build a synthetic index of `count` entries and time each query `repeats` times."""
    if count < 1:
        raise ValueError("entries must be at least 1")
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    entries = make_entries(count)
    index = FileIndex()
    install_ms = _median_ms(lambda: index.install(entries), 1, timer)
    snap = index.ensure()

    queries, typo = build_queries(entries)
    results: list[QueryResult] = []
    for label, query in queries:
        matches = len(index.find(query))
        median = _median_ms(lambda q=query: index.find(q), repeats, timer)
        results.append(QueryResult(label, query, matches, median))

    word_ms = _median_ms(
        lambda: difflib.get_close_matches(
            typo, snap.vocab, n=CLOSE_TOKENS_PER_WORD, cutoff=CLOSE_TOKEN_CUTOFF
        ),
        repeats,
        timer,
    )
    return BenchReport(
        backend=backend_name(),
        entries=len(snap.entries),
        vocabulary=len(snap.vocab),
        install_ms=install_ms,
        queries=results,
        word_lookup_ms=word_ms,
        typo_word=typo,
    )


def format_report(report: BenchReport) -> list[str]:
    lines = [
        "File search benchmark",
        f"Scorer: {report.backend}",
        f"Entries: {report.entries:,}   Vocabulary: {report.vocabulary:,} tokens",
        f"Index build (install): {report.install_ms:.1f} ms",
        "",
        f"{'Query':<16}{'Matches':>8}{'Median ms':>12}",
    ]
    for item in report.queries:
        lines.append(f"{item.label:<16}{item.matches:>8}{item.median_ms:>12.2f}")
    lines.append("")
    lines.append(
        f"One fuzzy word lookup over the vocabulary ('{report.typo_word}'): "
        f"{report.word_lookup_ms:.2f} ms"
    )
    if report.backend == "difflib":
        lines.append(
            "rapidfuzz is not installed, so scoring uses difflib. Results do not change if it is added."
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="Benchmark file search on a synthetic index.")
    parser.add_argument("--entries", type=int, default=DEFAULT_ENTRIES, help="index size (default 50000)")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS, help="timed runs per query")
    args = parser.parse_args(argv)
    try:
        report = benchmark(args.entries, args.repeats)
    except ValueError as exc:
        parser.error(str(exc))
    for line in format_report(report):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
