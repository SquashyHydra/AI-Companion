"""Tests for the wake word diagnostic and WakeWordDetector.scores. The model and capture are fakes."""

import numpy as np
import pytest

from companion.audio.errors import SpeechError
from companion.audio.segmenter import FRAME_SAMPLES
from companion.audio.wakeword import WakeWordDetector
from companion.diagnose_wakeword import advice, format_report, run, scan


def _frame(value: float) -> np.ndarray:
    return np.full(FRAME_SAMPLES, value, dtype=np.float32)


def _first_sample(frame: np.ndarray) -> float:
    return float(frame[0])


FRAME_S = FRAME_SAMPLES / 16000


class FakeModel:
    """Returns the first sample of each frame as the score for one model name."""

    def __init__(self, name: str = "hey_jarvis") -> None:
        self._name = name
        self.resets = 0

    def predict(self, pcm):
        return {self._name: float(pcm[0]) / 32767.0}

    def reset(self):
        self.resets += 1


def test_scan_counts_rising_edges_and_their_times():
    scores = [0.1, 0.9, 0.9, 0.1, 0.8]
    report = scan([_frame(s) for s in scores], _first_sample, threshold=0.5)
    assert report.frames == 5
    assert len(report.triggers) == 2
    assert report.triggers[0] == 1 * FRAME_S
    assert report.triggers[1] == 4 * FRAME_S


def test_scan_sustained_high_score_is_one_trigger():
    report = scan([_frame(0.9)] * 6, _first_sample, threshold=0.5)
    assert report.triggers == [0.0]


def test_scan_reports_highest_score_even_without_a_trigger():
    report = scan([_frame(0.2), _frame(0.4)], _first_sample, threshold=0.5)
    assert report.triggers == []
    assert report.max_score == pytest.approx(0.4, abs=1e-6)


def test_scan_of_no_frames_is_empty():
    report = scan([], _first_sample, threshold=0.5)
    assert report.frames == 0
    assert report.triggers == []
    assert report.max_score == 0.0


def test_advice_for_empty_recording_points_at_microphone_check():
    report = scan([], _first_sample, threshold=0.5)
    assert "diagnose_audio" in advice(report)[0]


def test_advice_when_wake_word_triggered_says_threshold_works():
    report = scan([_frame(0.9)], _first_sample, threshold=0.5)
    assert advice(report) == ["The wake word triggered 1 time(s). The threshold is working."]


def test_advice_suggests_lower_threshold_when_best_score_is_close():
    report = scan([_frame(0.42)], _first_sample, threshold=0.5)
    text = advice(report)[0]
    assert "best score was 0.42" in text
    assert "threshold: 0.37" in text


def test_advice_suggestion_is_just_below_the_best_score():
    report = scan([_frame(0.21)], _first_sample, threshold=0.5)
    assert "threshold: 0.16" in advice(report)[0]


def test_advice_for_weak_scores_asks_to_check_microphone_and_distance():
    report = scan([_frame(0.05)], _first_sample, threshold=0.5)
    text = advice(report)[0]
    assert "No strong score" in text
    assert "diagnose_audio" in text


def test_format_report_lists_frames_score_and_trigger_times():
    report = scan([_frame(0.1), _frame(0.9)], _first_sample, threshold=0.5)
    lines = format_report(report)
    assert lines[0] == "Scored 2 frames over 0.1 s. Threshold 0.50."
    assert lines[1] == "Highest score: 0.900"
    assert lines[2] == f"Triggers: 1 (at {FRAME_S:.1f} s)"


def test_detector_scores_returns_each_model_score():
    detector = WakeWordDetector("hey_jarvis", 0.5, model=FakeModel())
    scores = detector.scores(_frame(0.6))
    assert set(scores) == {"hey_jarvis"}
    assert abs(scores["hey_jarvis"] - 0.6) < 0.01


def test_detector_hit_is_built_on_scores():
    detector = WakeWordDetector("hey_jarvis", 0.5, model=FakeModel())
    assert detector.hit(_frame(0.6)) is True
    assert detector.hit(_frame(0.4)) is False


def test_detector_scores_needs_a_loaded_model():
    detector = WakeWordDetector("hey_jarvis", 0.5)
    with pytest.raises(SpeechError, match="not loaded"):
        detector.scores(_frame(0.1))


def test_run_reports_trigger_from_recording():
    detector = WakeWordDetector("hey_jarvis", 0.5, model=FakeModel())
    audio = np.concatenate([_frame(0.1), _frame(0.9), _frame(0.1)])
    lines: list[str] = []

    code = run(
        None,
        seconds=1.0,
        threshold=0.5,
        model_name="hey_jarvis",
        detector=detector,
        capture_audio=lambda device, seconds: audio,
        out=lines.append,
    )

    assert code == 0
    assert any(line.startswith("Triggers: 1") for line in lines)
    assert detector._model.resets == 1


def test_run_returns_error_when_capture_fails():
    detector = WakeWordDetector("hey_jarvis", 0.5, model=FakeModel())
    lines: list[str] = []

    def broken(device, seconds):
        raise RuntimeError("device busy")

    code = run(
        None,
        seconds=1.0,
        threshold=0.5,
        model_name="hey_jarvis",
        detector=detector,
        capture_audio=broken,
        out=lines.append,
    )

    assert code == 1
    assert any("Recording failed" in line for line in lines)
