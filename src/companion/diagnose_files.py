"""Check the file index and try name searches.

Usage (from the project folder, with PYTHONPATH=src):
    python -m companion.diagnose_files
    python -m companion.diagnose_files "hey jude" "abbey rod"
    python -m companion.diagnose_files --bench 20
"""

from __future__ import annotations

import argparse
import statistics
import time

from companion.config import ConfigError, get_config
from companion.files.index import FileIndex
from companion.files.roots import root_problems
from companion.tools.builtin.files import file_index

BENCH_QUERIES = ("music", "budget", "hey jude", "abbey rod", "invoice", "vacation")


def _show(index: FileIndex, query: str) -> None:
    print(f'\nSearch: "{query}"')
    matches = index.find(query, limit=5)
    if not matches:
        print("  no matches")
        return
    for match in matches:
        kind = "folder" if match.entry.is_folder else "file"
        print(f"  {match.score:.2f}  {kind:<6}  {match.entry.path}")


def _bench(index: FileIndex, runs: int, queries: tuple[str, ...]) -> None:
    print(f"\nTiming {runs} runs per query (milliseconds):")
    for query in queries:
        times: list[float] = []
        for _ in range(runs):
            started = time.perf_counter()
            index.find(query, limit=10)
            times.append((time.perf_counter() - started) * 1000)
        print(
            f'  "{query}": median {statistics.median(times):.1f}, worst {max(times):.1f}'
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the file index and try name searches.")
    parser.add_argument("queries", nargs="*", help="names to search for")
    parser.add_argument(
        "--bench", type=int, default=0, metavar="N", help="time N searches per query"
    )
    args = parser.parse_args(argv)
    if args.bench < 0:
        parser.error("--bench must be 0 or more")

    try:
        config = get_config()
    except ConfigError as exc:
        print(f"Config error: {exc}")
        return 1

    for problem in root_problems(config.tools.file_roots):
        print(problem)

    index = file_index()
    started = time.perf_counter()
    count = index.build()
    elapsed = time.perf_counter() - started
    print(f"Index built: {count} entries in {elapsed:.2f}s")
    for root, total in sorted(index.root_counts().items()):
        print(f"  {total:>8}  {root}")

    for query in args.queries:
        _show(index, query)
    if args.bench:
        _bench(index, args.bench, tuple(args.queries) or BENCH_QUERIES)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
