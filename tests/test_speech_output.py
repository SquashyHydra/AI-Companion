"""Tests for speech text cleanup, sentence splitting, the speech worker, and the Piper wrapper."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from companion.audio.errors import SpeechError
from companion.audio.tts import PiperSynthesizer
from companion.llm.streaming import SentenceSplitter
from companion.voice.speech import SpeechOut, sanitize_for_speech


# ----- sanitize_for_speech -----


def test_markdown_emphasis_is_removed_but_the_words_are_kept():
    assert sanitize_for_speech("**Bold** and *italic* and ~~gone~~") == "Bold and italic and gone"


def test_headings_and_bullets_are_removed():
    assert sanitize_for_speech("# Title\n- one\n- two") == "Title one two"


def test_links_keep_their_text_and_bare_urls_are_dropped():
    assert sanitize_for_speech("See [the docs](https://example.com/docs) now.") == "See the docs now."
    assert sanitize_for_speech("Visit https://example.com/a?b=1 today") == "Visit today"


def test_bracketed_stage_directions_are_not_spoken():
    assert sanitize_for_speech("I'm here! [laughs] How can I help?") == "I'm here! How can I help?"


def test_code_blocks_are_dropped_and_inline_code_keeps_its_text():
    spoken = sanitize_for_speech("Run this:\n```python\nprint(1)\n```\nDone.")
    assert "print" not in spoken
    assert spoken.endswith("Done.")
    assert sanitize_for_speech("Use `ls -la` now") == "Use ls -la now"


def test_iso_dates_are_spoken_as_month_day_year():
    assert sanitize_for_speech("Due 2025-03-04.") == "Due March 4, 2025."


def test_an_impossible_date_is_left_alone():
    assert sanitize_for_speech("2025-13-01") == "2025-13-01"


def test_degree_sign_is_spoken():
    assert sanitize_for_speech("21 °C") == "21 degrees C"


def test_emoji_and_symbols_are_dropped():
    assert sanitize_for_speech("Great job 🎉 today") == "Great job today"
    assert sanitize_for_speech("a < b | c") == "a b c"


# ----- SentenceSplitter -----


def test_sentences_are_cut_at_terminators_followed_by_space():
    splitter = SentenceSplitter()
    assert splitter.feed("Hello there. How are you? Fine!") == ["Hello there.", "How are you?"]
    assert splitter.flush() == ["Fine!"]


def test_a_sentence_split_across_chunks_is_emitted_once_complete():
    splitter = SentenceSplitter()
    assert splitter.feed("Hello the") == []
    assert splitter.feed("re. Next") == ["Hello there."]
    assert splitter.flush() == ["Next"]


def test_a_terminator_at_the_end_of_the_buffer_waits_for_more_text():
    splitter = SentenceSplitter()
    assert splitter.feed("It costs 3.") == []
    assert splitter.feed("5 dollars. ") == ["It costs 3.5 dollars."]


def test_common_abbreviations_do_not_end_a_sentence():
    assert SentenceSplitter().feed("Ask Dr. Smith today. ") == ["Ask Dr. Smith today."]


def test_newlines_end_a_sentence():
    assert SentenceSplitter().feed("Line one\nLine two\n") == ["Line one", "Line two"]


def test_a_long_run_without_full_stops_is_cut_before_max_chars():
    splitter = SentenceSplitter()
    text = "word " * 60
    pieces = splitter.feed(text) + splitter.flush()
    assert len(pieces) > 1
    assert all(len(piece) <= 200 for piece in pieces)
    assert " ".join(pieces).split() == text.split()


def test_first_words_cuts_the_first_piece_at_a_clause_after_three_words():
    splitter = SentenceSplitter(first_words=7)
    assert splitter.feed("Yes, of course, I can help with that. ") == [
        "Yes, of course,",
        "I can help with that.",
    ]


def test_first_words_falls_back_to_the_nth_word_when_there_is_no_clause():
    splitter = SentenceSplitter(first_words=7)
    assert splitter.feed("Sure I can check that for you now. ") == [
        "Sure I can check that for you",
        "now.",
    ]


def test_first_words_is_configurable():
    splitter = SentenceSplitter(first_words=2)
    assert splitter.feed("Yes, of course, I can help. ") == ["Yes, of", "course, I can help."]


def test_first_words_only_affects_the_first_piece():
    splitter = SentenceSplitter(first_words=2)
    assert splitter.feed("Yes, of course, I can help. Then more here. ") == [
        "Yes, of",
        "course, I can help.",
        "Then more here.",
    ]


def test_without_first_words_the_first_sentence_is_not_cut_early():
    assert SentenceSplitter().feed("Yes, of course, I can help with that. ") == [
        "Yes, of course, I can help with that."
    ]


# ----- SpeechOut -----


class FakeSynth:
    sample_rate = 10

    def __init__(self, gate: threading.Event | None = None, fail_on: str | None = None) -> None:
        self.calls: list[str] = []
        self._gate = gate
        self._fail_on = fail_on

    def synthesize(self, text: str, *, generation: int = 0) -> np.ndarray:
        self.calls.append(text)
        if self._gate is not None:
            self._gate.wait(2.0)
        if text == self._fail_on:
            raise RuntimeError("boom")
        return np.zeros(len(text), dtype=np.float32)


class FakePlayer:
    """Records writes. `sounding` stands in for audio still playing; `completed` for marks reached."""

    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0
        self.writes: list[tuple[object, int]] = []
        self.completed: list[object] = []
        self.sounding = False

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1

    def write(self, samples: np.ndarray, mark: object = None) -> None:
        self.writes.append((mark, samples.size))
        self.sounding = True

    def flush(self) -> list[object]:
        self.sounding = False
        completed, self.completed = self.completed, []
        return completed

    def busy(self, tail_s: float = 0.0) -> bool:
        return self.sounding


def wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def test_sentences_are_synthesized_in_order_and_written_with_their_text_as_mark():
    synth, player = FakeSynth(), FakePlayer()
    speech = SpeechOut(synth, player)
    speech.start()
    assert player.started == 1

    speech.begin_reply()
    speech.feed("Hello there. How are")
    speech.feed(" you? ")
    speech.finish()

    assert wait_until(lambda: len(player.writes) == 2)
    assert [mark for mark, _ in player.writes] == ["Hello there.", "How are you?"]
    assert synth.calls == ["Hello there.", "How are you?"]
    speech.close()
    assert player.stopped == 1


def test_finish_speaks_the_unfinished_last_sentence():
    synth, player = FakeSynth(), FakePlayer()
    speech = SpeechOut(synth, player)
    speech.start()
    speech.begin_reply()
    speech.feed("No full stop here")
    speech.finish()

    assert wait_until(lambda: [mark for mark, _ in player.writes] == ["No full stop here"])
    speech.close()


def test_sentences_with_no_letters_or_digits_are_not_spoken():
    synth, player = FakeSynth(), FakePlayer()
    speech = SpeechOut(synth, player)
    speech.start()
    speech.begin_reply()
    speech.feed("Okay. ... ")
    speech.finish()

    assert wait_until(lambda: len(player.writes) == 1)
    assert synth.calls == ["Okay."]
    speech.close()


def test_interrupt_drops_sentences_that_are_still_waiting_to_be_written():
    gate = threading.Event()
    synth, player = FakeSynth(gate=gate), FakePlayer()
    speech = SpeechOut(synth, player)
    speech.start()
    speech.begin_reply()
    speech.feed("First one here. Second one here. ")
    speech.finish()

    # The worker is now blocked inside the first sentence's synthesis.
    assert wait_until(lambda: synth.calls == ["First one here."])
    spoken, was_busy = speech.interrupt()
    assert was_busy
    assert spoken == ""

    gate.set()
    speech.close()  # joins the worker, so its late result has been handled
    assert player.writes == []
    assert synth.calls == ["First one here."]


def test_speech_after_an_interrupt_is_not_dropped():
    gate = threading.Event()
    synth, player = FakeSynth(gate=gate), FakePlayer()
    speech = SpeechOut(synth, player)
    speech.start()
    speech.begin_reply()
    speech.feed("Old reply here. ")
    assert wait_until(lambda: synth.calls == ["Old reply here."])
    speech.interrupt()

    gate.set()
    speech.begin_reply()
    speech.feed("New reply here. ")
    assert wait_until(lambda: [mark for mark, _ in player.writes] == ["New reply here."])
    speech.close()


def test_interrupt_returns_the_sentences_that_finished_playing():
    synth, player = FakeSynth(), FakePlayer()
    speech = SpeechOut(synth, player)
    speech.start()
    player.completed = ["One.", "Two."]
    player.sounding = True

    assert speech.interrupt() == ("One. Two.", True)
    assert speech.interrupt() == ("", False)
    speech.close()


def test_speaking_is_true_while_queued_or_sounding():
    gate = threading.Event()
    synth, player = FakeSynth(gate=gate), FakePlayer()
    speech = SpeechOut(synth, player)
    speech.start()
    assert not speech.speaking()

    speech.begin_reply()
    speech.feed("One here. ")
    assert wait_until(lambda: synth.calls == ["One here."])
    assert speech.speaking()  # still synthesizing

    gate.set()
    assert wait_until(lambda: len(player.writes) == 1)
    assert speech.speaking()  # now sounding
    player.sounding = False
    assert not speech.speaking()
    speech.close()


def test_a_synthesis_error_is_reported_and_the_next_sentence_still_plays():
    errors: list[str] = []
    synth, player = FakeSynth(fail_on="Broken one."), FakePlayer()
    speech = SpeechOut(synth, player, on_error=errors.append)
    speech.start()
    speech.begin_reply()
    speech.feed("Broken one. Good one. ")
    speech.finish()

    assert wait_until(lambda: len(player.writes) == 1)
    assert [mark for mark, _ in player.writes] == ["Good one."]
    assert len(errors) == 1
    assert "Could not speak that sentence" in errors[0]
    speech.close()


# ----- PiperSynthesizer (no real voice needed) -----


def test_a_missing_voice_file_says_how_to_download_one(tmp_path: Path):
    synth = PiperSynthesizer(tmp_path / "missing.onnx")
    with pytest.raises(SpeechError, match="piper.download_voices"):
        synth.load()
    assert not synth.loaded


def test_speed_outside_the_supported_range_is_rejected():
    with pytest.raises(ValueError):
        PiperSynthesizer(Path("voice.onnx"), speed=3.0)
    with pytest.raises(ValueError):
        PiperSynthesizer(Path("voice.onnx"), speed=0.1)
    assert PiperSynthesizer(Path("voice.onnx"), speed=0.5).sample_rate == 0


def test_synthesizing_before_loading_is_an_error():
    with pytest.raises(SpeechError, match="not loaded"):
        PiperSynthesizer(Path("voice.onnx")).synthesize("hello")
