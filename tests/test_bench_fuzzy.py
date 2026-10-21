"""Tests for the file search benchmark. Uses small indexes so the suite stays fast."""

import pytest

from companion.bench_fuzzy import benchmark, build_queries, format_report, make_entries


def test_make_entries_is_deterministic_for_a_seed():
    first = [entry.key for entry in make_entries(50, seed=3)]
    second = [entry.key for entry in make_entries(50, seed=3)]
    assert first == second
    assert len(first) == 50


def test_make_entries_are_media_files_with_three_word_names():
    entries = make_entries(10)
    assert all(entry.media and not entry.is_folder for entry in entries)
    assert all(len(entry.key.split()) == 3 for entry in entries)


def test_build_queries_cover_exact_substring_typo_and_miss():
    entries = make_entries(100)
    queries, typo = build_queries(entries)
    labels = [label for label, _query in queries]
    assert labels == ["exact", "substring", "fuzzy (1 typo)", "miss"]
    assert typo not in " ".join(entry.key for entry in entries)


def test_benchmark_reports_expected_match_counts():
    report = benchmark(300, repeats=1)
    counts = {item.label: item.matches for item in report.queries}
    assert report.entries == 300
    assert counts["exact"] >= 1
    assert counts["substring"] >= 1
    assert counts["fuzzy (1 typo)"] >= 1
    assert counts["miss"] == 0


def test_benchmark_times_are_non_negative():
    report = benchmark(100, repeats=2)
    assert report.install_ms >= 0
    assert report.word_lookup_ms >= 0
    assert all(item.median_ms >= 0 for item in report.queries)


def test_format_report_names_the_scorer_and_each_query():
    report = benchmark(100, repeats=1)
    lines = format_report(report)
    assert f"Scorer: {report.backend}" in lines
    assert report.backend in ("difflib", "rapidfuzz")
    text = "\n".join(lines)
    for label in ("exact", "substring", "fuzzy (1 typo)", "miss"):
        assert label in text


def test_benchmark_rejects_empty_index_and_zero_repeats():
    with pytest.raises(ValueError, match="entries"):
        benchmark(0)
    with pytest.raises(ValueError, match="repeats"):
        benchmark(10, repeats=0)
