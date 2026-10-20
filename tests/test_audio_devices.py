"""Tests for WASAPI-only microphone listing, device resolution, and stream options. No hardware needed."""

import pytest

import companion.audio.capture as capture
from companion.audio.capture import list_input_devices, resolve_input_device
from companion.audio.errors import SpeechError

# Mirrors the shape of sounddevice's query_hostapis() and query_devices().
HOST_APIS = [
    {"name": "MME", "default_input_device": 0},
    {"name": "Windows WASAPI", "default_input_device": 3},
]
DEVICES = [
    {"name": "Microsoft Sound Mapper - Input", "hostapi": 0, "max_input_channels": 2},
    {"name": "Microphone (Realtek(R) Audio)", "hostapi": 1, "max_input_channels": 2},
    {"name": "Microphone (Realtek(R) Audio)", "hostapi": 0, "max_input_channels": 2},
    {
        "name": "SteelSeries Sonar - Microphone (SteelSeries Sonar Virtual Audio Device)",
        "hostapi": 1,
        "max_input_channels": 2,
    },
    {"name": "Speakers (Realtek)", "hostapi": 1, "max_input_channels": 0},
]


@pytest.fixture
def devices(monkeypatch):
    monkeypatch.setattr(capture, "_host_apis", lambda: HOST_APIS)
    monkeypatch.setattr(capture, "_devices", lambda: DEVICES)


def test_listing_shows_only_wasapi_microphones(devices):
    assert list_input_devices() == [
        (1, "Microphone (Realtek(R) Audio)"),
        (3, "SteelSeries Sonar - Microphone (SteelSeries Sonar Virtual Audio Device)"),
    ]


def test_listing_is_empty_without_wasapi(devices, monkeypatch):
    monkeypatch.setattr(capture, "_host_apis", lambda: [HOST_APIS[0]])
    assert list_input_devices() == []


def test_default_is_the_wasapi_default_input(devices):
    assert resolve_input_device(None) == 3


def test_default_falls_back_to_first_wasapi_mic(devices, monkeypatch):
    monkeypatch.setattr(capture, "_host_apis", lambda: [HOST_APIS[0], {**HOST_APIS[1], "default_input_device": -1}])
    assert resolve_input_device(None) == 1


def test_number_must_be_a_wasapi_microphone(devices):
    assert resolve_input_device(1) == 1
    with pytest.raises(SpeechError, match="not a WASAPI microphone"):
        resolve_input_device(0)
    with pytest.raises(SpeechError, match="not a WASAPI microphone"):
        resolve_input_device(99)


def test_name_fragment_matches_one_wasapi_microphone(devices):
    assert resolve_input_device("realtek") == 1
    assert resolve_input_device("SONAR") == 3


def test_name_fragment_must_match_something(devices):
    with pytest.raises(SpeechError, match="no WASAPI microphone matches 'sound mapper'"):
        resolve_input_device("sound mapper")


def test_ambiguous_name_fragment_is_refused(devices):
    with pytest.raises(SpeechError, match="matches 2 microphones"):
        resolve_input_device("microphone")


def test_resolve_without_any_wasapi_microphone_fails(devices, monkeypatch):
    monkeypatch.setattr(capture, "_host_apis", lambda: [HOST_APIS[0]])
    with pytest.raises(SpeechError, match="no WASAPI microphones found"):
        resolve_input_device(None)


class _Recorded:
    def __init__(self):
        self.kwargs = None


@pytest.fixture
def opened(devices, monkeypatch):
    import sounddevice as sd

    record = _Recorded()

    def fake_input_stream(**kwargs):
        record.kwargs = kwargs
        return "stream"

    monkeypatch.setattr(sd, "InputStream", fake_input_stream)
    monkeypatch.setattr(sd, "WasapiSettings", lambda **kwargs: ("wasapi", kwargs))
    return record


def test_wasapi_streams_ask_windows_to_convert_the_sample_rate(opened):
    assert capture._default_stream_factory(samplerate=16000, device=3) == "stream"
    assert opened.kwargs["extra_settings"] == ("wasapi", {"auto_convert": True})
    assert opened.kwargs["samplerate"] == 16000


def test_non_wasapi_streams_get_no_extra_settings(opened):
    capture._default_stream_factory(samplerate=16000, device=0)
    assert "extra_settings" not in opened.kwargs
