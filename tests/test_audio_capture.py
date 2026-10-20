"""Tests for microphone capture. Uses a fake stream, so no microphone is needed."""

import numpy as np
import pytest

from companion.audio.capture import SAMPLE_RATE, Recorder
from companion.audio.errors import SpeechError


class FakeStream:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True

    def feed(self, samples):
        """Simulate the audio thread delivering one block of mono samples."""
        block = np.asarray(samples, dtype=np.float32).reshape(-1, 1)
        self.kwargs["callback"](block, len(block), None, None)


def _factory(streams):
    def make(**kwargs):
        stream = FakeStream(**kwargs)
        streams.append(stream)
        return stream

    return make


def test_records_mono_float32_between_start_and_stop():
    streams = []
    recorder = Recorder(device=3, stream_factory=_factory(streams))

    recorder.start()
    stream = streams[0]
    assert stream.started
    assert stream.kwargs["samplerate"] == SAMPLE_RATE
    assert stream.kwargs["channels"] == 1
    assert stream.kwargs["dtype"] == "float32"
    assert stream.kwargs["device"] == 3
    assert recorder.recording

    stream.feed([0.1, 0.2])
    stream.feed([0.3])
    audio = recorder.stop()

    assert audio.dtype == np.float32
    np.testing.assert_allclose(audio, [0.1, 0.2, 0.3], rtol=1e-6)
    assert stream.stopped and stream.closed
    assert not recorder.recording


def test_stop_with_no_audio_returns_empty_array():
    recorder = Recorder(stream_factory=_factory([]))
    recorder.start()
    assert recorder.stop().size == 0


def test_a_new_recording_starts_empty():
    streams = []
    recorder = Recorder(stream_factory=_factory(streams))

    recorder.start()
    streams[0].feed([0.9, 0.9])
    recorder.stop()

    recorder.start()
    streams[1].feed([0.5])
    np.testing.assert_allclose(recorder.stop(), [0.5], rtol=1e-6)


def test_double_start_and_stop_without_start_are_errors():
    recorder = Recorder(stream_factory=_factory([]))
    with pytest.raises(SpeechError, match="not recording"):
        recorder.stop()

    recorder.start()
    with pytest.raises(SpeechError, match="already recording"):
        recorder.start()


def test_stream_open_failure_is_a_speech_error():
    def broken(**kwargs):
        raise OSError("no device")

    recorder = Recorder(stream_factory=broken)
    with pytest.raises(SpeechError, match="could not open the microphone"):
        recorder.start()
    assert not recorder.recording
