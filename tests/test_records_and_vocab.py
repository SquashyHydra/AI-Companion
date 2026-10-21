"""Tests for transcript logging, debug WAV saving, and speech vocabulary helpers."""

import json
import wave
from types import SimpleNamespace

import numpy as np

from companion.audio.records import TranscriptLog, save_utterance_wav
from companion.audio.stt import Transcription, build_prompt, speech_vocabulary


def test_transcript_log_appends_one_json_line_per_call(tmp_path):
    log = TranscriptLog(tmp_path / "transcripts.jsonl")
    log.append(
        transcription=Transcription("open notes", avg_logprob=-0.2, no_speech_prob=0.01),
        seconds=1.5,
        low_confidence=False,
    )
    log.append(
        transcription=Transcription("mumble", avg_logprob=-1.8, no_speech_prob=0.4),
        seconds=0.8,
        low_confidence=True,
    )

    lines = log.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first, second = (json.loads(line) for line in lines)
    assert first["text"] == "open notes"
    assert first["low_confidence"] is False
    assert second["low_confidence"] is True
    assert first["seconds"] == 1.5
    assert first["avg_logprob"] == -0.2
    assert "timestamp" in first


def test_save_utterance_wav_writes_16_bit_mono_16khz(tmp_path):
    audio = np.zeros(1600, dtype=np.float32)
    path = save_utterance_wav(audio, tmp_path)

    assert path.exists()
    assert path.suffix == ".wav"
    with wave.open(str(path), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 16000
        assert wav.getnframes() == 1600


def test_low_confidence_threshold():
    assert Transcription("x", avg_logprob=-2.0).low_confidence(-1.0) is True
    assert Transcription("x", avg_logprob=-0.2).low_confidence(-1.0) is False


def test_build_prompt_dedupes_and_returns_none_when_empty():
    assert build_prompt([]) is None
    prompt = build_prompt(["Spotify", "spotify", "Notes"])
    assert prompt is not None
    assert prompt.count("Spotify") + prompt.count("spotify") == 1
    assert "Notes" in prompt


def test_speech_vocabulary_merges_config_words_and_allowlist_names(tmp_path):
    apps = tmp_path / "apps.json"
    apps.write_text(json.dumps({"Spotify": "C:/Spotify.exe", "Notepad": "notepad.exe"}), encoding="utf-8")
    config = SimpleNamespace(
        vocabulary=("Mario", "mario", "  Zelda  "),
        include_app_names=True,
        include_file_names=False,
    )

    words = speech_vocabulary(config, apps)

    lowered = [word.lower() for word in words]
    assert lowered.count("mario") == 1
    assert "zelda" in lowered
    assert "spotify" in lowered
    assert "notepad" in lowered


def test_speech_vocabulary_skips_allowlist_when_disabled(tmp_path):
    apps = tmp_path / "apps.json"
    apps.write_text(json.dumps({"Spotify": "x"}), encoding="utf-8")
    config = SimpleNamespace(
        vocabulary=("Mario",), include_app_names=False, include_file_names=False
    )

    assert [word.lower() for word in speech_vocabulary(config, apps)] == ["mario"]
