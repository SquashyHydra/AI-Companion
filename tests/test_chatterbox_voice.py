"""Tests for the Chatterbox-Turbo voice: tags, engine fallback, server engine, and client round trip."""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from companion.audio import chatterbox_server
from companion.audio.errors import SpeechError, SupersededError
from companion.audio.tts import ResilientSynthesizer, load_voice
from companion.audio.tts_chatterbox import ChatterboxSynthesizer
from companion.config import PROJECT_ROOT, ChatterboxConfig, ConfigError, TtsConfig, load_config
from companion.voice.tags import TagDisplayFilter, normalize_tag, tags_from_vocabulary

# Tags a model reports in this test suite. The real set is read from the model at runtime.
CHATTERBOX_TAGS = frozenset({"laugh", "clear throat", "sigh"})


# --- tags -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("laugh", "laugh"),
        ("  Laugh  ", "laugh"),
        ("laughs", "laugh"),
        ("clear  throat", "clear throat"),
        ("banana", None),
        ("", None),
    ],
)
def test_normalize_tag(raw: str, expected: str | None) -> None:
    assert normalize_tag(raw, CHATTERBOX_TAGS) == expected


def test_normalize_tag_refuses_a_tag_the_engine_does_not_render() -> None:
    assert normalize_tag("laugh", frozenset()) is None


def test_tags_are_the_bracketed_added_tokens_only() -> None:
    vocab = ["[laugh]", "[clear throat]", "<|endoftext|>", "hello", "[Bad Tag!]"]
    assert tags_from_vocabulary(vocab) == frozenset({"laugh", "clear throat"})


def test_display_filter_hides_supported_tag_split_across_chunks() -> None:
    filt = TagDisplayFilter(show=False, tags=CHATTERBOX_TAGS)
    out = "".join(filt.feed(chunk) for chunk in ["That was fun [la", "ugh], really"]) + filt.flush()
    assert out == "That was fun , really"


def test_display_filter_keeps_unknown_bracket_text() -> None:
    filt = TagDisplayFilter(show=False, tags=CHATTERBOX_TAGS)
    out = filt.feed("see [banana] now") + filt.flush()
    assert out == "see [banana] now"


def test_display_filter_passes_everything_when_shown() -> None:
    filt = TagDisplayFilter(show=True, tags=CHATTERBOX_TAGS)
    assert filt.feed("hi [laugh]") == "hi [laugh]"


def test_display_filter_flushes_unfinished_tag_at_end() -> None:
    filt = TagDisplayFilter(show=False, tags=CHATTERBOX_TAGS)
    assert filt.feed("open [lau") == "open "
    assert filt.flush() == "[lau"


# --- engine fallback --------------------------------------------------------------------------


class _FailingChatterbox:
    def __init__(self) -> None:
        self.closed = False

    def load(self) -> None:
        raise SpeechError("server did not start")

    def close(self) -> None:
        self.closed = True


class _FakePiper:
    sample_rate = 22050
    tags: frozenset[str] = frozenset()

    def __init__(self) -> None:
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def synthesize(self, text: str, *, generation: int = 0) -> np.ndarray:
        return np.zeros(1, dtype=np.float32)


class _OkChatterbox:
    sample_rate = 24000
    tags = CHATTERBOX_TAGS

    def load(self) -> None:
        return None

    def synthesize(self, text: str, *, generation: int = 0) -> np.ndarray:
        return np.zeros(1, dtype=np.float32)


def _tts(engine: str = "chatterbox_turbo") -> TtsConfig:
    cb = ChatterboxConfig(
        python=Path("python"),
        host="127.0.0.1",
        port=8765,
        voice_reference=None,
        autostart=True,
        startup_timeout_s=300,
        request_timeout_s=60,
    )
    return TtsConfig(engine=engine, device="cuda", chatterbox=cb)


def _audio(speed: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(voice_path=Path("voice.onnx"), speech_speed=speed)


AUDIO = _audio()


def test_load_voice_falls_back_to_piper_when_chatterbox_fails() -> None:
    messages: list[str] = []
    failing = _FailingChatterbox()
    piper = _FakePiper()

    voice = load_voice(
        _tts(),
        AUDIO,  # type: ignore[arg-type]
        notify=messages.append,
        make_chatterbox=lambda _tts: failing,  # type: ignore[arg-type,return-value]
        make_piper=lambda _audio: piper,  # type: ignore[arg-type,return-value]
    )

    assert voice is piper
    assert piper.loaded
    assert failing.closed
    assert any("Falling back to Piper" in m for m in messages)


def test_load_voice_uses_chatterbox_when_it_loads() -> None:
    ok = _OkChatterbox()
    voice = load_voice(
        _tts(),
        AUDIO,  # type: ignore[arg-type]
        notify=lambda _m: None,
        make_chatterbox=lambda _tts: ok,  # type: ignore[arg-type,return-value]
        make_piper=lambda _audio: pytest.fail("Piper must not load"),  # type: ignore[arg-type,return-value]
    )
    assert isinstance(voice, ResilientSynthesizer)
    assert voice._primary is ok  # noqa: SLF001


def test_load_voice_piper_setting_skips_chatterbox() -> None:
    piper = _FakePiper()
    voice = load_voice(
        _tts(engine="piper"),
        AUDIO,  # type: ignore[arg-type]
        notify=lambda _m: None,
        make_chatterbox=lambda _tts: pytest.fail("Chatterbox must not be built"),  # type: ignore[arg-type,return-value]
        make_piper=lambda _audio: piper,  # type: ignore[arg-type,return-value]
    )
    assert voice is piper
    assert piper.loaded


# --- server engine ----------------------------------------------------------------------------


class _FakeModel:
    sr = 24000

    def __init__(self, *, has_conds: bool = True, fail: bool = False) -> None:
        self.conds = object() if has_conds else None
        self._fail = fail
        self.calls: list[str] = []

    def generate(self, text: str) -> np.ndarray:
        if self._fail:
            raise RuntimeError("generate exploded")
        self.calls.append(text)
        return np.linspace(-0.5, 0.5, 240, dtype=np.float32).reshape(1, -1)


def test_engine_loads_and_synthesizes_float32() -> None:
    engine = chatterbox_server.Engine(lambda: _FakeModel())
    engine.load()
    assert engine.status == "ready"
    assert engine.sample_rate == 24000
    samples = engine.synthesize("hello [laugh]")
    assert samples.dtype == np.dtype("<f4")
    assert samples.ndim == 1
    assert samples.size == 240


def test_engine_reports_error_when_model_has_no_voice() -> None:
    engine = chatterbox_server.Engine(lambda: _FakeModel(has_conds=False))
    engine.load()
    assert engine.status == "error"
    assert "voice_reference" in engine.error


def test_engine_refuses_to_synthesize_before_ready() -> None:
    engine = chatterbox_server.Engine(lambda: _FakeModel())
    with pytest.raises(RuntimeError, match="not ready"):
        engine.synthesize("hi")


# --- client round trip against an in-process server --------------------------------------------


def test_client_synthesizes_through_running_server() -> None:
    engine = chatterbox_server.Engine(lambda: _FakeModel())
    engine.load()
    server = chatterbox_server.make_server(engine, "127.0.0.1", 0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = ChatterboxSynthesizer(
            python="python",
            project_root=Path("."),
            host="127.0.0.1",
            port=port,
            device="cpu",
            voice_reference="",
            autostart=False,
            startup_timeout_s=5,
            request_timeout_s=5,
        )
        client.load()
        assert client.sample_rate == 24000
        samples = client.synthesize("hello [laugh] there")
        assert samples.dtype == np.float32
        assert samples.size == 240
        client.close()
    finally:
        server.shutdown()
        server.server_close()


@contextlib.contextmanager
def _serving(engine: chatterbox_server.Engine) -> Iterator[int]:
    """Run an in-process server for the engine and yield its port."""
    server = chatterbox_server.make_server(engine, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


def _client(port: int) -> ChatterboxSynthesizer:
    return ChatterboxSynthesizer(
        python=Path("python"),
        project_root=Path("."),
        host="127.0.0.1",
        port=port,
        device="cpu",
        autostart=False,
        startup_timeout_s=5,
        request_timeout_s=5,
    )


# --- generations and barge-in -----------------------------------------------------------------


def test_queued_request_below_a_cancelled_generation_is_refused() -> None:
    engine = chatterbox_server.Engine(lambda: _FakeModel())
    engine.load()
    engine.cancel(2)
    engine.cancel(1)  # cancel never moves backwards
    with pytest.raises(chatterbox_server.Superseded):
        engine.synthesize("old", 1)
    assert engine.synthesize("new", 2).size == 240


def test_audio_made_after_a_barge_in_is_dropped() -> None:
    holder: dict[str, chatterbox_server.Engine] = {}

    class CancellingModel(_FakeModel):
        def generate(self, text: str) -> np.ndarray:
            wav = super().generate(text)
            if text == "late":
                holder["engine"].cancel(5)  # barge-in arrives while this sentence is generating
            return wav

    engine = chatterbox_server.Engine(lambda: CancellingModel())
    holder["engine"] = engine
    engine.load()
    with pytest.raises(chatterbox_server.Superseded):
        engine.synthesize("late", 4)
    assert engine.synthesize("next", 5).size == 240


def test_barge_in_cancel_returns_409_and_the_client_raises_superseded() -> None:
    engine = chatterbox_server.Engine(lambda: _FakeModel())
    engine.load()
    with _serving(engine) as port:
        client = _client(port)
        client.load()
        client.cancel(4)
        with pytest.raises(SupersededError):
            client.synthesize("late", generation=3)
        assert client.synthesize("fresh", generation=4).size == 240
        client.close()


def test_server_reads_its_voice_tags_from_the_model_tokenizer() -> None:
    class Tokenizer:
        def get_added_vocab(self) -> dict[str, int]:
            return {"[sigh]": 50260, "[laugh]": 50259, "<|endoftext|>": 50256}

    class TaggedModel(_FakeModel):
        tokenizer = Tokenizer()

    engine = chatterbox_server.Engine(lambda: TaggedModel())
    engine.load()
    assert engine.health()["tags"] == ["laugh", "sigh"]


# --- voice settings ---------------------------------------------------------------------------


def test_nano_engine_is_built_from_the_tts_setting() -> None:
    seen: list[str] = []

    def make(tts: TtsConfig) -> _OkChatterbox:
        seen.append(tts.engine)
        return _OkChatterbox()

    load_voice(
        _tts(engine="chatterbox_nano"),
        AUDIO,  # type: ignore[arg-type]
        notify=lambda _m: None,
        make_chatterbox=make,  # type: ignore[arg-type,return-value]
        make_piper=lambda _audio: pytest.fail("Piper must not load"),  # type: ignore[arg-type,return-value]
    )
    assert seen == ["chatterbox_nano"]


def test_speed_warning_is_shown_for_chatterbox_when_speed_is_not_one() -> None:
    messages: list[str] = []
    load_voice(
        _tts(),
        _audio(1.5),  # type: ignore[arg-type]
        notify=messages.append,
        make_chatterbox=lambda _tts: _OkChatterbox(),  # type: ignore[arg-type,return-value]
        make_piper=lambda _audio: pytest.fail("Piper must not load"),  # type: ignore[arg-type,return-value]
    )
    assert any("applies to Piper only" in m for m in messages)


def test_no_speed_warning_at_the_default_speed() -> None:
    messages: list[str] = []
    load_voice(
        _tts(),
        _audio(1.0),  # type: ignore[arg-type]
        notify=messages.append,
        make_chatterbox=lambda _tts: _OkChatterbox(),  # type: ignore[arg-type,return-value]
        make_piper=lambda _audio: pytest.fail("Piper must not load"),  # type: ignore[arg-type,return-value]
    )
    assert not any("applies to Piper only" in m for m in messages)


def test_no_speed_warning_when_piper_is_the_voice() -> None:
    messages: list[str] = []
    load_voice(
        _tts(engine="piper"),
        _audio(1.5),  # type: ignore[arg-type]
        notify=messages.append,
        make_chatterbox=lambda _tts: pytest.fail("Chatterbox must not be built"),  # type: ignore[arg-type,return-value]
        make_piper=lambda _audio: _FakePiper(),  # type: ignore[arg-type,return-value]
    )
    assert not any("applies to Piper only" in m for m in messages)


class _FakeProcess:
    stdin = None
    returncode = None

    def poll(self) -> None:
        return None

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def kill(self) -> None:
        return None


def _started_command(tmp_path: Path, **options: object) -> tuple[list[str], dict[str, str]]:
    """Start the server through the client with a fake process and return its command and env."""
    python = tmp_path / "python.exe"
    python.write_text("", encoding="utf-8")
    captured: dict[str, object] = {}

    def popen(command: list[str], **kwargs: object) -> _FakeProcess:
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return _FakeProcess()

    client = ChatterboxSynthesizer(
        python=python,
        project_root=tmp_path,
        port=8799,
        log_path=tmp_path / "server.log",
        popen=popen,  # type: ignore[arg-type]
        **options,  # type: ignore[arg-type]
    )
    client._start_process()
    client.close()
    return captured["command"], captured["env"]  # type: ignore[return-value]


def test_offline_setting_sets_hf_hub_offline_for_the_server(tmp_path: Path) -> None:
    command, env = _started_command(tmp_path, engine="chatterbox_nano", device="cpu", offline=True)
    assert env["HF_HUB_OFFLINE"] == "1"
    assert command[command.index("--engine") + 1] == "chatterbox_nano"
    assert command[command.index("--device") + 1] == "cpu"


def test_online_setting_leaves_hub_access_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    _, env = _started_command(tmp_path, offline=False)
    assert "HF_HUB_OFFLINE" not in env


# --- config -----------------------------------------------------------------------------------


def _config_file(tmp_path: Path, **tts: object) -> Path:
    """Copy the project config and replace keys in its tts block."""
    data = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    data.setdefault("tts", {}).update(tts)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_project_config_uses_nano_on_cuda_offline_with_early_first_chunk() -> None:
    tts = load_config(PROJECT_ROOT / "config.yaml", environ={}).tts
    assert tts.engine == "chatterbox_nano"
    assert tts.device == "cuda"
    assert tts.offline is True
    assert tts.first_chunk_words == 7


@pytest.mark.parametrize("engine", ["chatterbox_turbo", "chatterbox_nano", "piper"])
def test_each_engine_name_is_accepted(tmp_path: Path, engine: str) -> None:
    assert load_config(_config_file(tmp_path, engine=engine), environ={}).tts.engine == engine


def test_unknown_engine_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(_config_file(tmp_path, engine="xtts"), environ={})


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_device_is_read_from_the_tts_section(tmp_path: Path, device: str) -> None:
    assert load_config(_config_file(tmp_path, device=device), environ={}).tts.device == device


def test_device_other_than_cuda_or_cpu_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(_config_file(tmp_path, device="gpu"), environ={})


def test_old_chatterbox_device_key_is_rejected_with_a_pointer(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="moved to tts.device"):
        load_config(_config_file(tmp_path, chatterbox={"device": "cuda"}), environ={})


@pytest.mark.parametrize("words", [0, 20])
def test_first_chunk_words_bounds_are_accepted(tmp_path: Path, words: int) -> None:
    assert load_config(_config_file(tmp_path, first_chunk_words=words), environ={}).tts.first_chunk_words == words


@pytest.mark.parametrize("words", [-1, 21])
def test_first_chunk_words_out_of_range_is_rejected(tmp_path: Path, words: int) -> None:
    with pytest.raises(ConfigError):
        load_config(_config_file(tmp_path, first_chunk_words=words), environ={})


def test_offline_must_be_true_or_false(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="offline"):
        load_config(_config_file(tmp_path, offline="yes"), environ={})
