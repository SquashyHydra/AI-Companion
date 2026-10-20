"""Tests for VAD trimming, Whisper transcription, and the push-to-talk flow. Fakes only."""

from types import SimpleNamespace

import numpy as np
import pytest

import companion.audio.push_to_talk as push_to_talk
from companion.audio.capture import SAMPLE_RATE
from companion.audio.errors import SpeechError
from companion.audio.push_to_talk import PushToTalk, build_push_to_talk
from companion.audio.stt import Transcriber
from companion.audio.vad import SpeechTrimmer
from companion.config import AudioConfig


def _audio(seconds: float) -> np.ndarray:
    return np.linspace(0, 1, int(seconds * SAMPLE_RATE), dtype=np.float32)


# ---------- VAD trimming ----------


def test_trim_keeps_the_span_from_first_to_last_speech():
    audio = np.arange(100, dtype=np.float32)
    trimmer = SpeechTrimmer(detect=lambda a: [(10, 20), (50, 60)])
    np.testing.assert_array_equal(trimmer.trim(audio), audio[10:60])


def test_trim_without_speech_is_empty():
    trimmer = SpeechTrimmer(detect=lambda a: [])
    assert trimmer.trim(_audio(1.0)).size == 0


def test_empty_audio_skips_detection():
    calls = []

    def detect(audio):
        calls.append(audio)
        return []

    trimmer = SpeechTrimmer(detect=detect)
    assert trimmer.trim(np.zeros(0, dtype=np.float32)).size == 0
    assert calls == []


# ---------- Transcription ----------


class FakeModel:
    def __init__(self, texts=(" hello", "world "), fail=None):
        self.texts = texts
        self.fail = fail
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio.size, kwargs))
        if self.fail is not None:
            raise self.fail
        return iter(SimpleNamespace(text=text) for text in self.texts), SimpleNamespace(language="en")


def _factory(loads, models, fail_on=()):
    """Return a model factory that records (size, device, compute) and fails for devices in fail_on."""

    def factory(size, device, compute_type):
        loads.append((size, device, compute_type))
        if device in fail_on:
            raise RuntimeError(f"{device} is not available")
        return models[device]

    return factory


def test_transcribe_joins_segments_and_passes_language():
    loads = []
    model = FakeModel()
    transcriber = Transcriber(
        model_size="small",
        device="cpu",
        compute_type="int8",
        language="en",
        model_factory=_factory(loads, {"cpu": model}),
    )

    assert transcriber.transcribe(_audio(1.0)) == "hello world"
    assert loads == [("small", "cpu", "int8")]
    _, kwargs = model.calls[0]
    assert kwargs["language"] == "en"
    assert kwargs["vad_filter"] is False


def test_empty_audio_is_not_transcribed():
    loads = []
    transcriber = Transcriber(model_factory=_factory(loads, {"cpu": FakeModel()}), device="cpu")
    assert transcriber.transcribe(np.zeros(0, dtype=np.float32)) == ""
    assert loads == []


def test_model_loads_once():
    loads = []
    transcriber = Transcriber(model_factory=_factory(loads, {"cpu": FakeModel()}), device="cpu")
    transcriber.load()
    transcriber.load()
    transcriber.transcribe(_audio(1.0))
    assert len(loads) == 1


def test_cuda_load_failure_falls_back_to_cpu():
    loads = []
    transcriber = Transcriber(
        device="cuda",
        compute_type="float16",
        model_factory=_factory(loads, {"cpu": FakeModel()}, fail_on=("cuda",)),
    )
    transcriber.load()
    assert loads == [("small", "cuda", "float16"), ("small", "cpu", "int8")]
    assert transcriber.device == "cpu"


def test_cuda_runtime_failure_reloads_on_cpu_and_retries():
    loads = []
    cuda_model = FakeModel(fail=RuntimeError("cublas64_12.dll not found"))
    cpu_model = FakeModel(texts=(" recovered",))
    transcriber = Transcriber(
        device="cuda",
        model_factory=_factory(loads, {"cuda": cuda_model, "cpu": cpu_model}),
    )
    assert transcriber.transcribe(_audio(1.0)) == "recovered"
    assert loads[-1] == ("small", "cpu", "int8")


def test_cpu_load_failure_is_a_speech_error():
    transcriber = Transcriber(
        device="cpu",
        model_factory=_factory([], {"cpu": FakeModel()}, fail_on=("cpu",)),
    )
    with pytest.raises(SpeechError, match="could not load Whisper model"):
        transcriber.load()


def test_cpu_runtime_failure_is_a_speech_error():
    transcriber = Transcriber(
        device="cpu",
        model_factory=_factory([], {"cpu": FakeModel(fail=RuntimeError("boom"))}),
    )
    with pytest.raises(SpeechError, match="transcription failed"):
        transcriber.transcribe(_audio(1.0))


# ---------- Push-to-talk flow ----------


class FakeRecorder:
    def __init__(self, audio=None, start_error=None):
        self.audio = _audio(1.0) if audio is None else audio
        self.start_error = start_error
        self.recording = False

    def start(self):
        if self.start_error is not None:
            raise self.start_error
        self.recording = True

    def stop(self):
        if not self.recording:
            raise SpeechError("not recording")
        self.recording = False
        return self.audio


def _transcriber(texts=(" turn on", "the lights"), fail=None):
    return Transcriber(
        device="cpu",
        model_factory=_factory([], {"cpu": FakeModel(texts=texts, fail=fail)}),
    )


def _speech_found(audio):
    return [(0, audio.size)]


def _ptt(messages, recorder=None, trimmer=None, transcriber=None, enter=None):
    return PushToTalk(
        recorder or FakeRecorder(),
        trimmer or SpeechTrimmer(detect=_speech_found),
        transcriber or _transcriber(),
        min_seconds=0.3,
        wait_for_enter=enter or (lambda prompt: ""),
        notify=messages.append,
    )


def test_listen_returns_the_transcript():
    messages = []
    assert _ptt(messages).listen() == "turn on the lights"
    assert messages == ["Listening... press Enter to stop."]


def test_too_short_recording_is_ignored():
    messages = []
    ptt = _ptt(messages, recorder=FakeRecorder(audio=_audio(0.1)))
    assert ptt.listen() == ""
    assert "too short" in messages[-1]


def test_no_speech_detected_is_reported():
    messages = []
    ptt = _ptt(messages, trimmer=SpeechTrimmer(detect=lambda a: []))
    assert ptt.listen() == ""
    assert messages[-1] == "No speech detected."


def test_unclear_speech_is_reported():
    messages = []
    assert _ptt(messages, transcriber=_transcriber(texts=("",))).listen() == ""
    assert messages[-1] == "Could not make out any words."


@pytest.mark.parametrize("interrupt", [EOFError(), KeyboardInterrupt()])
def test_cancelling_while_recording_discards_the_audio(interrupt):
    messages = []
    recorder = FakeRecorder()

    def enter(prompt):
        raise interrupt

    assert _ptt(messages, recorder=recorder, enter=enter).listen() == ""
    assert not recorder.recording
    assert messages[-1] == "Recording cancelled."


def test_microphone_start_error_is_reported():
    messages = []
    recorder = FakeRecorder(start_error=SpeechError("could not open the microphone: no device"))
    assert _ptt(messages, recorder=recorder).listen() == ""
    assert messages[-1].startswith("Microphone error")


def test_transcription_error_is_reported():
    messages = []
    assert _ptt(messages, transcriber=_transcriber(fail=RuntimeError("boom"))).listen() == ""
    assert messages[-1].startswith("Transcription error")


def _audio_config(**overrides) -> AudioConfig:
    values = dict(
        enabled=True,
        input_device=None,
        whisper_model="small",
        whisper_device="cpu",
        whisper_compute_type="int8",
        language="en",
        vad_threshold=0.5,
        min_seconds=0.3,
    )
    values.update(overrides)
    return AudioConfig(**values)


@pytest.fixture
def no_silero(monkeypatch):
    """Skip loading the Silero model and the real microphone lookup in build tests."""
    monkeypatch.setattr(SpeechTrimmer, "load", lambda self: None)
    monkeypatch.setattr(push_to_talk, "resolve_input_device", lambda value: 3)


def test_build_loads_models_and_reports_ready(no_silero, monkeypatch):
    import companion.audio.stt as stt

    loads = []
    monkeypatch.setattr(stt, "_whisper_factory", _factory(loads, {"cpu": FakeModel()}))
    messages = []

    ptt = build_push_to_talk(_audio_config(), notify=messages.append)

    assert isinstance(ptt, PushToTalk)
    assert loads == [("small", "cpu", "int8")]
    assert messages[-1] == "Speech models ready (Whisper small on cpu)."


def test_build_raises_when_whisper_cannot_load(no_silero, monkeypatch):
    import companion.audio.stt as stt

    monkeypatch.setattr(stt, "_whisper_factory", _factory([], {}, fail_on=("cpu",)))
    with pytest.raises(SpeechError, match="could not load Whisper model"):
        build_push_to_talk(_audio_config(), notify=lambda message: None)


def test_build_stops_before_loading_models_when_no_microphone_matches(no_silero, monkeypatch):
    import companion.audio.stt as stt

    def no_mic(value):
        raise SpeechError("no WASAPI microphone matches 'x'; see /mics")

    loads = []
    monkeypatch.setattr(push_to_talk, "resolve_input_device", no_mic)
    monkeypatch.setattr(stt, "_whisper_factory", _factory(loads, {"cpu": FakeModel()}))
    with pytest.raises(SpeechError, match="no WASAPI microphone matches"):
        build_push_to_talk(_audio_config(input_device="x"), notify=lambda message: None)
    assert loads == []
