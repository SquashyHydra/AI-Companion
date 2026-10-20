"""Tests for the speaker player and WASAPI device selection. No sound device is opened."""

from __future__ import annotations

import numpy as np
import pytest

from companion.audio import playback
from companion.audio.errors import SpeechError
from companion.audio.playback import SpeakerPlayer, describe_output_error, resolve_output_device


class FakeClock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class FakeStream:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


def make_player(clock: FakeClock | None = None, device: int | None = None, sample_rate: int = 10):
    """Sample rate 10 keeps the arithmetic simple: one sample is 0.1 s."""
    streams: list[FakeStream] = []

    def factory(**kwargs):
        stream = FakeStream(**kwargs)
        streams.append(stream)
        return stream

    player = SpeakerPlayer(
        sample_rate,
        device=device,
        stream_factory=factory,
        clock=clock or FakeClock(),
    )
    return player, streams


def pull(player: SpeakerPlayer, frames: int) -> np.ndarray:
    """Act as the audio device asking for `frames` samples."""
    outdata = np.full((frames, 1), 9.0, dtype=np.float32)
    player._on_audio(outdata, frames, None, None)
    return outdata[:, 0].copy()


def test_start_opens_one_mono_stream_at_the_requested_rate_and_device():
    player, streams = make_player(device=3, sample_rate=22050)
    player.start()

    stream = streams[0]
    assert stream.started
    assert stream.kwargs["samplerate"] == 22050
    assert stream.kwargs["device"] == 3
    assert stream.kwargs["channels"] == 1
    assert stream.kwargs["dtype"] == "float32"
    assert stream.kwargs["callback"] == player._on_audio
    assert player.running

    player.start()  # a second start must not open another stream
    assert len(streams) == 1


def test_a_failed_open_raises_a_readable_speech_error():
    def factory(**kwargs):
        raise RuntimeError("-9997 Invalid sample rate")

    player = SpeakerPlayer(22050, stream_factory=factory, clock=FakeClock())
    with pytest.raises(SpeechError, match="sample rate"):
        player.start()
    assert not player.running


def test_sentences_play_back_to_back_and_marks_arrive_in_order():
    player, _ = make_player()
    player.start()
    player.write(np.array([0.1, 0.2, 0.3], dtype=np.float32), mark="a")
    player.write(np.array([0.4, 0.5], dtype=np.float32), mark="b")

    out = pull(player, 4)
    np.testing.assert_allclose(out, [0.1, 0.2, 0.3, 0.4], rtol=1e-6)
    assert player.take_completed() == ["a"]

    out = pull(player, 4)
    np.testing.assert_allclose(out, [0.5, 0.0, 0.0, 0.0], atol=1e-6)
    assert player.take_completed() == ["b"]


def test_a_mark_is_reported_only_when_the_callback_reaches_it():
    player, _ = make_player()
    player.start()
    player.write(np.zeros(2, dtype=np.float32), mark="x")

    pull(player, 1)
    assert player.take_completed() == []
    pull(player, 1)
    assert player.take_completed() == []  # all audio played, but the mark is not reached yet
    pull(player, 1)
    assert player.take_completed() == ["x"]


def test_an_empty_queue_gives_silence():
    player, _ = make_player()
    player.start()
    assert not player.busy()
    np.testing.assert_array_equal(pull(player, 4), np.zeros(4, dtype=np.float32))


def test_a_mark_without_audio_is_still_reported():
    player, _ = make_player()
    player.start()
    player.write(np.zeros(0, dtype=np.float32), mark="empty")
    assert player.busy()
    pull(player, 2)
    assert player.take_completed() == ["empty"]
    assert not player.busy()


def test_busy_covers_queued_audio_and_the_tail_after_the_last_sample():
    clock = FakeClock(100.0)
    player, _ = make_player(clock=clock)
    player.start()
    player.write(np.zeros(2, dtype=np.float32), mark="x")
    assert player.busy()

    pull(player, 2)  # 2 samples at 10 Hz: audible until 100.2
    pull(player, 1)  # consumes the mark, so the queue is empty
    assert not playback_items(player)

    clock.now = 100.0 + 0.2
    assert not player.busy()
    assert player.busy(tail_s=0.5)  # the tail keeps the mic gated after the audio ends
    clock.now = 101.0
    assert not player.busy(tail_s=0.5)


def playback_items(player: SpeakerPlayer) -> bool:
    """Return True if the player still has queued items (samples or marks)."""
    with player._lock:
        return bool(player._items)


def test_flush_drops_the_queue_and_returns_marks_that_already_played():
    player, _ = make_player()
    player.start()
    player.write(np.zeros(2, dtype=np.float32), mark="a")
    player.write(np.zeros(2, dtype=np.float32), mark="b")
    pull(player, 3)  # all of a, and one sample of b

    assert player.flush() == ["a"]
    assert not player.busy()
    assert player.take_completed() == []
    np.testing.assert_array_equal(pull(player, 2), np.zeros(2, dtype=np.float32))


def test_flush_ends_the_audible_time_immediately():
    clock = FakeClock(100.0)
    player, _ = make_player(clock=clock)
    player.start()
    player.write(np.zeros(10, dtype=np.float32))
    pull(player, 10)  # audible until 101.0
    assert player.busy()

    player.flush()
    assert not player.busy()


def test_stop_closes_the_stream_and_clears_the_queue():
    player, streams = make_player()
    player.start()
    player.write(np.zeros(5, dtype=np.float32), mark="x")

    player.stop()
    assert streams[0].stopped
    assert streams[0].closed
    assert not player.running
    assert not player.busy()


def test_stop_before_start_is_harmless():
    player, _ = make_player()
    player.stop()
    assert not player.running


def test_describe_output_error_gives_hints_for_common_codes():
    assert "sample rate" in describe_output_error(RuntimeError("-9997 Invalid"))
    assert "unavailable" in describe_output_error(RuntimeError("-9985 device gone"))
    assert describe_output_error(RuntimeError("")) == "RuntimeError"


@pytest.fixture
def speakers(monkeypatch):
    """Two WASAPI speakers and one USB headset, with the system default set to the headset."""
    monkeypatch.setattr(
        playback,
        "list_output_devices",
        lambda: [(1, "Speakers (Realtek)"), (4, "Headphones (USB Headset)"), (6, "Speakers (HDMI)")],
    )
    monkeypatch.setattr(playback, "_wasapi_index", lambda: 0)
    monkeypatch.setattr(playback, "_host_apis", lambda: {0: {"default_output_device": 4}})


def test_no_output_device_uses_the_system_default(speakers):
    assert resolve_output_device(None) == 4


def test_no_output_device_falls_back_to_the_first_speaker_when_default_is_missing(monkeypatch):
    monkeypatch.setattr(playback, "list_output_devices", lambda: [(1, "Speakers"), (6, "HDMI")])
    monkeypatch.setattr(playback, "_wasapi_index", lambda: 0)
    monkeypatch.setattr(playback, "_host_apis", lambda: {0: {"default_output_device": 99}})
    assert resolve_output_device(None) == 1


def test_a_device_number_must_be_a_wasapi_output(speakers):
    assert resolve_output_device(6) == 6
    with pytest.raises(SpeechError, match="not a WASAPI output"):
        resolve_output_device(5)


def test_a_name_fragment_must_match_exactly_one_speaker(speakers):
    assert resolve_output_device("usb") == 4
    with pytest.raises(SpeechError, match="matches 2 speakers"):
        resolve_output_device("speakers")
    with pytest.raises(SpeechError, match="no WASAPI speaker matches"):
        resolve_output_device("bluetooth")


def test_no_speakers_at_all_is_an_error(monkeypatch):
    monkeypatch.setattr(playback, "list_output_devices", lambda: [])
    with pytest.raises(SpeechError, match="no WASAPI speakers"):
        resolve_output_device(None)
