"""Tests for the voice-activity segmenter. Frames are fed with explicit probabilities."""

import numpy as np
import pytest

from companion.audio.segmenter import Noise, Segmenter, SpeechOngoing, Utterance


def _frame(value: float = 0.0) -> np.ndarray:
    return np.full(512, value, dtype=np.float32)


def _segmenter(**overrides) -> Segmenter:
    settings = dict(
        vad_threshold=0.5,
        pre_roll_ms=0,
        min_speech_ms=100,
        end_silence_ms=200,
        max_utterance_s=10.0,
    )
    settings.update(overrides)
    return Segmenter(**settings)


def _feed(seg: Segmenter, probabilities) -> list:
    events = []
    for probability in probabilities:
        events += seg.feed(_frame(probability), probability)
    return events


def _utterances(events) -> list[Utterance]:
    return [event for event in events if isinstance(event, Utterance)]


def test_speech_then_silence_makes_one_utterance_with_trailing_silence():
    seg = _segmenter()
    speech = [0.9] * 10
    silence_frames = 7  # 7 * 32 ms = 224 ms, the first count at or above 200 ms
    events = _feed(seg, speech + [0.0] * silence_frames)

    assert any(isinstance(event, SpeechOngoing) for event in events)
    utterances = _utterances(events)
    assert len(utterances) == 1
    assert utterances[0].audio.size == (len(speech) + silence_frames) * 512
    assert utterances[0].cut is False
    assert not seg.active


def test_silence_alone_produces_nothing():
    seg = _segmenter()
    assert _feed(seg, [0.0] * 50) == []
    assert not seg.active


def test_a_short_blip_below_min_speech_is_dropped_as_noise():
    seg = _segmenter(min_speech_ms=250)
    events = _feed(seg, [0.9, 0.9] + [0.0] * 10)
    assert _utterances(events) == []
    assert any(isinstance(event, Noise) for event in events)


def test_brief_dips_inside_speech_do_not_end_the_utterance():
    # 0.4 is below the threshold of 0.5 but above the release level (0.35), so it still counts.
    seg = _segmenter()
    events = _feed(seg, [0.9] + [0.4] * 20)
    assert _utterances(events) == []
    assert seg.active


def test_probability_below_the_release_level_counts_as_silence():
    seg = _segmenter()
    events = _feed(seg, [0.9] * 4 + [0.1] * 10)
    assert len(_utterances(events)) == 1


def test_long_speech_is_cut_at_the_maximum_length():
    seg = _segmenter(max_utterance_s=0.1, min_speech_ms=0)  # about 3 frames
    events = _feed(seg, [0.9] * 5)
    utterances = _utterances(events)
    assert len(utterances) == 1
    assert utterances[0].cut is True


def test_pre_roll_adds_frames_from_before_speech_started():
    seg = _segmenter(pre_roll_ms=64)  # keeps the last 2 quiet frames
    events = _feed(seg, [0.0, 0.0, 0.0] + [0.9] * 4 + [0.0] * 7)
    utterance = _utterances(events)[0]
    assert utterance.audio.size == (2 + 4 + 7) * 512


def test_reset_drops_a_partial_utterance():
    seg = _segmenter()
    _feed(seg, [0.9] * 3)
    assert seg.active
    seg.reset()
    assert not seg.active
    assert _utterances(_feed(seg, [0.0] * 20)) == []


def test_invalid_threshold_is_rejected():
    with pytest.raises(ValueError):
        _segmenter(vad_threshold=1.5)
