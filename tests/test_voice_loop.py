"""Tests for the voice loop's shutdown while a confirmation is pending."""

import queue
import threading

from companion.voice.confirm import EventConfirmer
from companion.voice.keys import LineEditor
from companion.voice.loop import VoiceLoop
from companion.voice.machine import ListenMachine, Quit


class FakePipeline:
    def __init__(self):
        self.started = 0
        self.stopped = 0

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1


class FakeKeys:
    """Returns one scripted batch of keys per poll. Quits once the script runs out."""

    def __init__(self, batches):
        self._batches = list(batches)
        self.confirming_flags: list[bool] = []

    def poll(self, confirming: bool = False):
        self.confirming_flags.append(confirming)
        if self._batches:
            return self._batches.pop(0)
        return [Quit()]


def test_shutdown_ends_a_pending_confirmation_without_waiting_for_the_timeout():
    stop = threading.Event()
    events: queue.Queue = queue.Queue()
    answers: queue.Queue[bool] = queue.Queue()
    confirmer = EventConfirmer(events.put, answers, timeout_s=30.0, stop=stop)

    result: list[bool] = []
    worker = threading.Thread(target=lambda: result.append(confirmer.confirm("Delete?")))
    worker.start()
    # Wait for the request so the loop sees it on its first event read.
    events.put(events.get(timeout=2.0))

    written: list[str] = []
    keys = FakeKeys([[], []])
    loop = VoiceLoop(
        machine=ListenMachine(barge_in_ms=400),
        pipeline=FakePipeline(),
        transcribe=lambda audio: None,
        stream=lambda *args: None,
        run_command=lambda text: "",
        new_history=lambda: [],
        keys=keys,
        editor=LineEditor(written.append),
        events=events,
        answers=answers,
        low_confidence_logprob=-1.0,
        write=lambda text: None,
        tick_s=0.01,
        stop=stop,
    )

    assert loop.run() == 0
    worker.join(timeout=2.0)
    assert not worker.is_alive()
    assert result == [False]
    assert True in keys.confirming_flags  # the loop routed keys in confirm mode


def test_loop_passes_confirming_to_the_key_reader():
    keys = FakeKeys([[]])
    loop = VoiceLoop(
        machine=ListenMachine(barge_in_ms=400),
        pipeline=FakePipeline(),
        transcribe=lambda audio: None,
        stream=lambda *args: None,
        run_command=lambda text: "",
        new_history=lambda: [],
        keys=keys,
        editor=LineEditor(lambda text: None),
        events=queue.Queue(),
        answers=queue.Queue(),
        low_confidence_logprob=-1.0,
        write=lambda text: None,
        tick_s=0.01,
    )
    assert loop.run() == 0
    assert keys.confirming_flags == [False, False]
