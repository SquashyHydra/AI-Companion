# PC Companion

A local, voice-driven AI companion for Windows. It talks to a local model through an OpenAI-compatible API and calls tools to act on the PC.

**Status: Phase 3 (speech input, upgraded).** Terminal chat with always-listening speech input (microphone, Silero VAD, faster-whisper), streaming replies, barge-in, push-to-talk and typed modes, LLM tool calling, a decorator-based tool registry, plugin auto-discovery, the allowlisted `open_app` and `list_apps` tools, a pluggable confirmation flow for risky tools, and audit logging. Built-in tools: `get_time`, `take_note`, `read_notes`, `clear_notes` (asks first), `open_app`, `list_apps`. Typed chat remains available as a fallback.

**Pinned dependency:** `openai==3.35.1` (exact version in [requirements.txt](requirements.txt)). The original spec mentioned the 1.x line, which is outdated. Upgrade deliberately and re-run the tests.

## Setup (PowerShell)

```powershell
cd "F:\AI Companion"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements-dev.txt   # runtime + pytest
```

`requirements.txt` includes the speech packages. `silero-vad` depends on PyTorch, which is a large download. Voice activity detection runs on the CPU.

## Speech input

The mode is set by `input.mode` in [config.yaml](config.yaml):

- `continuous` (default): the companion listens all the time. Speak naturally; each pause that lasts `end_silence_ms` ends an utterance. It prints what it heard (`Heard: "..."`) and the reply streams in as it is generated.
- `push_to_talk`: press Enter on an empty line to start recording, and Enter again to stop.
- `text`: typing only.

In continuous mode, Enter on an empty line pauses or resumes listening. Typed text is always accepted. Ctrl+C quits.

While a yes/no confirmation is pending, the keyboard only answers it: `y` approves, `n` declines, and other keys are not echoed. Enter is ignored, so it cannot pause listening or decline by accident. Ctrl+C cancels the confirmation (declined) and keeps the companion running. Ctrl+C with no confirmation pending quits.

### Barge-in and streaming

Replies stream token by token as they arrive. If you start speaking for `barge_in_ms` while the companion is replying, the reply is cancelled. The partial text is kept in history and marked `[interrupted]`. Barge-in is disabled while a yes/no confirmation is pending, so speech cannot cancel a confirmation by accident. You can still answer it by voice. Typing during a reply cancels it and sends your new text next.

### States

`PAUSED` (Enter), `LISTENING` (waiting), `RECORDING` (you are speaking), `TRANSCRIBING`, `THINKING` (model is replying). A microphone error moves to `PAUSED`; typing keeps working and Enter retries the microphone.

### Pipeline

Microphone (16 kHz mono, 512-sample frames) -> Silero VAD probability per frame -> segmenter (hysteresis, pre-roll, minimum speech length, end-of-speech silence, maximum length) -> faster-whisper -> LLM.

- Transcripts below `low_confidence_logprob` are shown as `(low confidence)` and are still sent.
- Each transcript is appended to `logs/transcripts.jsonl`.
- `vocabulary` and the names in `data/apps.json` bias Whisper toward words you use. `use_hotwords` also sends them as hotwords, which can cause over-insistence, so it is off by default.

- The first run downloads the Whisper model from Hugging Face (`small` by default). This needs internet access once.
- Whisper runs on the GPU (`cuda`, `float16`) by default. If CUDA fails to load or run, it switches to the CPU (`int8`) and logs a warning.
- If the microphone or the models are unavailable, the companion starts in typed-only mode and shows why.
- `/mics` lists microphones using the Windows WASAPI host API only. Windows shows the same physical microphone under several host APIs (MME, DirectSound, WDM-KS); those duplicates are hidden. Only WASAPI devices are ever shown.
- To pick a microphone, set `audio.input_device` in [config.yaml](config.yaml) to a number from `/mics`, or to a unique part of its name. Leave it empty to use the Windows default recording device. A value that matches no WASAPI microphone, or matches more than one, stops speech input with a message, and the companion stays in typed mode.
- Recording uses 16 kHz. WASAPI shared mode rejects 16 kHz unless Windows converts the stream, so the companion asks WASAPI to convert it automatically.
- Set `audio.language` to `null` for automatic language detection. The default is `en`.
- Set `audio.enabled: false` to turn speech off.

Audio settings (`audio:` in [config.yaml](config.yaml)): `enabled`, `input_device`, `whisper_model`, `whisper_device` (`cuda` or `cpu`), `whisper_compute_type`, `language`, `vad_threshold` (0 to 1), `min_seconds` (shortest recording that is sent, push-to-talk).

Continuous listening settings: `pre_roll_ms`, `min_speech_ms`, `end_silence_ms` (lower is faster but cuts pauses), `max_utterance_s`, `barge_in_ms`.

Voice activity detection uses Silero VAD only. faster-whisper's built-in `vad_filter` is off. `min_silence_duration_ms` is not used; pause length is set by `end_silence_ms`.

Transcription settings: `beam_size`, `vocabulary`, `include_app_names`, `use_hotwords`, `low_confidence_logprob`.

Debug: `save_utterances: true` writes each utterance as a 16 kHz mono WAV file to `utterance_dir` (default `logs/utterances`). Use it to check what the microphone captured. Remember to turn it off, since it stores your voice on disk.

### Audio diagnostic

```powershell
$env:PYTHONPATH = "src"
python -m companion.diagnose_audio
```

This lists the WASAPI microphones (the configured one is marked `*`), records 5 seconds from the configured device, and saves the result as a 16 kHz mono WAV under `logs/diagnostics/`. It prints the peak (linear and dBFS) and the average level (RMS), and warns on near silence, clipping, or buffer overflows. Play the WAV back to hear what the companion receives. Exit code is 0 on success and 1 if the device cannot be used.

## Speech output

Replies are spoken by one of three engines, chosen with `tts.engine` in [config.yaml](config.yaml):

- `chatterbox_turbo` (default): Resemble AI's Chatterbox-Turbo (350M, English, GPU). It can render paralinguistic tags such as `[laugh]`, `[chuckle]`, and `[cough]`. The model runs in a separate server process from its own virtual environment, `.venv-tts`, because it needs a newer torch than the main app.
- `chatterbox_nano`: Resemble AI's Chatterbox-Nano (110M, English). Smaller and faster to load than Turbo, with the same tag list read from the model. It runs on GPU (`tts.device: cuda`) or CPU, but CPU is too slow for live speech (see measurements below).
- `piper` (fallback): Piper on CPU. Piper cannot render tags. If Chatterbox cannot start, the app falls back to Piper and prints why. If a Chatterbox request fails mid-session, the app restarts the server once and retries. A second failure switches to Piper for the rest of the session and says so aloud. Barge-in is never counted as a failure. Piper loads lazily, so its voice file must exist for the fallback to work.

Piper setup, from the project root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install piper-tts
.\.venv\Scripts\python.exe -m piper.download_voices en_US-lessac-medium --download-dir data/voices
```

Chatterbox setup (needed for `chatterbox_turbo` and `chatterbox_nano`). The versions are pinned in [requirements-tts.txt](requirements-tts.txt), which uses the CUDA 12.8 build of torch that the RTX 50 series needs. Run from the project root in PowerShell:

```powershell
py -3.12 -m venv .venv-tts
.\.venv-tts\Scripts\python.exe -m pip install -r requirements-tts.txt
.\.venv-tts\Scripts\python.exe -m pip install --no-deps --force-reinstall --no-build-isolation "git+https://github.com/resemble-ai/chatterbox.git@5de7a54aa4e5e2baadb0182dde554908b48b85c2"
.\.venv-tts\Scripts\python.exe -m pip check
```

The chatterbox install comes from a pinned git commit, not PyPI. PyPI `chatterbox-tts` 0.1.7 has no Nano model. It is installed with `--no-deps` because its metadata pins `torch==2.6.0`, which conflicts with the pinned torch 2.7.1. Git must be on PATH for that step.

`pip check` is expected to report only chatterbox metadata conflicts: the torch and torchaudio 2.6.0 pins, and the optional gradio, pykakasi, and spacy-pkuseg packages, which are not installed. Those do not affect the server. The output from a clean install is saved in `logs/tts_verify_pip_check.txt`.

Each engine downloads its model from Hugging Face on first use (Turbo about 4 GB, Nano about 0.9 GB) and needs internet for that. `tts.offline` (default `true`) sets `HF_HUB_OFFLINE=1` for the server, so it reads only the local cache. With `tts.offline: true` and no cached model, the server fails to load with a `LocalEntryNotFoundError`. To cache an engine, set `tts.offline: false` once, or run this once, then set it back to `true`:

```powershell
$env:HF_HUB_OFFLINE = "0"
.\.venv-tts\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('ResembleAI/chatterbox-nano')"
```

Use `'ResembleAI/chatterbox-turbo'` for Turbo. Install `spacy-pkuseg` or `pykakasi` only if an import error names them.

With `tts.chatterbox.autostart: true`, the app starts the server itself on `127.0.0.1:8765` and stops it on exit. Loading took 19 to 23 s on the RTX 5060 Ti in the latest measurements, so the app waits up to `startup_timeout_s` (default 300). A server that is already running on that port is reused. The Chatterbox engines have no speed control. `speech_speed` applies to Piper only, and the app prints a startup warning if it is not 1.0 while a Chatterbox engine is active. Chatterbox tags are shown on screen unless `display.show_voice_tags` is false.

Settings under `tts:` in [config.yaml](config.yaml):

- `engine`: `chatterbox_turbo`, `chatterbox_nano`, or `piper`.
- `device`: `cuda` or `cpu`. Applies to the Chatterbox engines only.
- `offline`: `true` (default) or `false`. See the download notes above.
- `first_chunk_words`: default 7. The first sentence of a reply is spoken once it reaches a clause boundary or this many words, whichever comes first. `0` turns the early cut off. Set in `chatterbox` and Piper alike.

Settings are under `audio:` in [config.yaml](config.yaml): `speak_replies` (true or false), `output_mode` (`headphones` or `speakers`), `output_device` (empty for the Windows default, a WASAPI number from the diagnostic, or a unique part of the speaker name), `voice_path` (Piper model), `speech_speed` (Piper only, 0.5 to 2.0), and `speakers_tail_ms` (default 400).

- Sentences are synthesized and played as they complete, so playback starts after the first sentence rather than the whole reply. Markdown, code blocks, and URLs are cleaned up before speaking. Qwen3 thinking text is never spoken or stored in history.
- Headphones (default): you can barge in by speaking for `barge_in_ms`. The reply stops, and history keeps only the sentences that finished playing, followed by `[interrupted]`.
- Speakers: the microphone is muted while the companion speaks and for `speakers_tail_ms` afterwards, so it cannot hear itself. Barge-in is off in this mode.
- Text and push-to-talk modes stay silent. Only continuous mode speaks replies.
- A yes/no answer is accepted only as `yes`, `y`, `no`, or `n`. Words such as "yes please" or "ok" are not accepted.
- Ctrl+C during a confirmation cancels it (declined). A second Ctrl+C within 2 seconds quits the app.

Check the speaker and voice with:

```powershell
$env:PYTHONPATH = "src"
python -m companion.diagnose_tts
```

It lists the WASAPI speakers (the configured one is marked `*`), loads the voice, synthesizes a test sentence, plays it, and reports the load time, synthesis time, and real-time factor. Use it to choose `output_device`. Use `--engine` and `--device` to test one Chatterbox engine on a given device.

To compare engines on this machine, run:

```powershell
python -m companion.diagnose_tts --compare
```

It loads Turbo (cuda), Nano (cuda and cpu), and Piper (cpu). For each one it synthesizes a short sentence, a long sentence, and a sentence with `[laugh]`, then prints load time, synthesis time, audio length, real-time factor (RTF, where below 1.0 is faster than real time), and GPU memory. WAV files go to `logs/diagnostics`. A row is skipped while a server is already running on its port. Stop the companion before running it.

## Running the LLM server

1. In LM Studio, load a model that supports tool calling (for example a Qwen3 build).
2. Start the local server (default `http://localhost:1234/v1`).
3. Set `llm.model` in [config.yaml](config.yaml) to the identifier LM Studio shows for the loaded model.

## Running the companion

```powershell
$env:PYTHONPATH = "src"
python -m companion
```

In-chat commands: `/tools` lists tools (tools that ask first are marked), `/mics` lists microphones, `/new` (or `/reset`) starts a new conversation, `/status` shows the model, context use, memory count, and log file, `/exit` quits. Type a message and press Enter to chat. Unknown `/` commands are reported as unknown; they are not sent to the model. Ctrl+C quits at any time.

## Conversation, persona, and memory

- **Persona:** the personality is in [data/persona.md](data/persona.md). It is read at the start of each conversation (`/new` picks up edits). A missing or empty file uses a built-in default. The safety rules in the system prompt always come first.
- **Long-term memory:** facts are stored in `data/memory/memories.sqlite`. The tools `remember`, `recall`, `list_memories`, and `forget` manage them.
  - Remember flows: "remember that ..." saves at once. "remember this" saves the previous message. A bare "remember" followed by the content in the next message also saves. Vague notes such as "it is at 3" are rewritten from the earlier message when one is available. Saves and forgets are decided by the app, not the model, so the model cannot claim a save that did not happen.
  - `forget <target>` reads back the matching texts and needs an exact yes or no. "Forget that" deletes the latest automatically saved memory immediately, with no read-back.
  - `remember` refuses text that looks like a password, key, or token.
  - Auto-save (`memory.auto_save`, default true) lets the companion save facts you state about yourself after a turn. It is skipped for tool turns, low-confidence speech, and remember or forget messages. Set `memory.announce_auto: true` to hear a short notice when it saves. A later contradiction can overwrite an earlier memory.
  - Recall is hybrid: keyword matches (SQLite FTS5, Porter stemming) plus semantic matches from a small CPU embedding model (`BAAI/bge-small-en-v1.5` through fastembed). Settings are under `memory:`: `embeddings` (true or false), `embedding_model`, `embedding_cache` (default `data/models`), and `semantic_min_score` (default 0.55, the cosine cutoff). The first run downloads the model and needs internet. If the model cannot load, the app prints a notice and uses keyword recall only.
  - Measured cosine scores with this model are close together for short queries (for example, "my job" against "User works as a nurse in Leeds." scored 0.572, and an unrelated memory reached 0.594). Tune `semantic_min_score` if you see missed or extra recalls.
- **Long conversations:** when history passes `llm.compact_at` of the context window, older turns are summarized in the background by one extra LLM call. The most recent `llm.keep_recent_turns` turns are kept as they are. The window is read from LM Studio, or set `llm.context_tokens`.
- **Errors:** if the LLM server is unreachable, the companion retries `llm.connect_retries` times with a growing delay, then reports the problem and keeps the conversation intact. Tool results longer than `llm.max_tool_result_chars` are shortened before the model sees them.
- **Conversation log:** each conversation is saved as plain text under `logs/conversations/`. Set `conversation.log_enabled: false` to turn this off.

See [MANUAL_TESTS.md](MANUAL_TESTS.md) for the hands-on checks to run after changes to speech input.

On startup the companion loads plugins from `plugins/` and prints which ones loaded or were skipped.

## Configuration

Settings live in [config.yaml](config.yaml). Environment variables override them:

| Variable | Setting |
|---|---|
| `COMPANION_CONFIG` | Path to an alternate config file |
| `COMPANION_LLM_BASE_URL` | `llm.base_url` |
| `COMPANION_LLM_MODEL` | `llm.model` |
| `COMPANION_LLM_API_KEY` | `llm.api_key` |
| `COMPANION_NOTES_DIR` | `tools.notes_dir` |

Other `tools` settings: `notes_dir` (default `data/notes`) and `apps_file` (default `data/apps.json`).

Phase 5 settings:

| Section | Key | Default | Meaning |
|---|---|---|---|
| `llm` | `context_tokens` | `0` | Context window; `0` reads it from LM Studio (fallback 8192) |
| `llm` | `compact_at` | `0.7` | Summarize older turns past this share of the window |
| `llm` | `keep_recent_turns` | `4` | Recent turns kept word for word |
| `llm` | `max_tool_result_chars` | `2000` | Tool output cut-off before it goes back to the model |
| `llm` | `max_summary_chars` | `1500` | Limit on the summary text |
| `llm` | `connect_retries` / `retry_delay_s` | `2` / `1.0` | Retries when the server is down |
| `conversation` | `persona_file` | `data/persona.md` | Personality text |
| `conversation` | `log_enabled` / `log_dir` | `true` / `logs/conversations` | Plain-text conversation log |
| `memory` | `enabled` / `db_path` | `true` / `data/memory/memories.sqlite` | Long-term memory store |
| `memory` | `max_in_prompt` / `max_chars` | `5` / `300` | Memories shown per turn / longest memory |

Caveats: recall is keyword-based, so it does not match synonyms. `forget` confirmation shows the words you said, not the memories it will remove. Summaries cost one extra LLM call each time they run.

## Adding a tool

Write a function with type hints and a docstring, then decorate it. The JSON schema for the model is generated from the signature and the docstring.

```python
from companion.tools.registry import tool

@tool
def shout(text: str) -> str:
    """Return the text in upper case.

    Args:
        text: The text to convert.
    """
    return text.upper()
```

Put it in a module under `src/companion/tools/builtin/` and import that module from [builtin/__init__.py](src/companion/tools/builtin/__init__.py), or put it in a plugin file (see below).

Tools return plain text. Errors are returned as text starting with `Error:` so the model can read them.

Options for `@tool`:

- `timeout` (seconds, default `10`, must be greater than 0).
- `requires_confirmation` (default `False`). The user must approve each call before it runs. Denied or unanswered calls are not run. `clear_notes` uses this.
- `name` (optional, defaults to the function name).

### Timeouts and their limits

Each call runs in a daemon thread and is abandoned after `timeout` seconds. Python cannot stop a thread, so the limits are:

- A timed-out call keeps running in the background until it returns. While it runs, new calls to the same tool are refused with an error. This prevents a stuck tool from piling up threads.
- Tools that use subprocesses or network calls must set their own timeouts (for example `subprocess.run(..., timeout=...)` or an HTTP client timeout). Otherwise the thread may linger.
- There is no process isolation. A tool that hangs in native code or holds a lock can block only its own thread. Process isolation was not added because no current tool needs it. Add it only if a tool truly requires it.

### Plugins

Any `*.py` file in [plugins/](plugins) is imported at startup, and its `@tool` functions are registered. Files starting with `_` are skipped. A plugin that fails to load is skipped, its partly registered tools are removed, and the error is shown and logged. Use `from companion.tools.registry import tool` in plugin files.

### Confirmation

A tool with `requires_confirmation=True` asks `Allow the companion to run name(args)? [y/N]` in the terminal. Only `y` or `yes` approves, and `n` or `no` declines. Any other answer re-asks, up to 3 attempts. Then the call is denied. Closing the input (EOF or Ctrl+C) also denies.

In continuous voice mode, the question is shown on the main screen and answered with `y`/`n` or by voice. There is no blocking `input()` call. If no answer arrives within `input.confirm_timeout_s` seconds (default 30, allowed range 5 to 300), the call is declined and the timeout is logged. Ctrl+C cancels a pending confirmation (declined) and does not quit. Shutdown ends a pending confirmation at once, declined and logged, so the program does not hang. Text and push-to-talk modes still use the blocking terminal prompt (`TerminalConfirmer`), so `confirm_timeout_s` does not apply there.

The executor never calls `input()` itself. It depends on a `Confirmer` protocol in [safety/confirm.py](src/companion/safety/confirm.py) with one method, `confirm(question) -> bool`. `TerminalConfirmer` is the console version. A future voice confirmer only needs to implement that method and be set with `set_confirmer()`. Every outcome (approved, declined, unavailable, error) is written to `tool_calls.jsonl`.

### Apps allowlist

`open_app` only opens names listed in [data/apps.json](data/apps.json). Names are case-insensitive. Each value must be one of:

- A bare program name, such as `notepad.exe`.
- An absolute path to a `.exe` or `.lnk` file. It must exist when opened.
- A Steam URL of the form `steam://rungameid/<id>`, for example `"witcher": "steam://rungameid/292030"`.

Invalid entries are skipped with a warning. Missing or malformed files are handled safely: a missing file means an empty allowlist, and malformed JSON stops the tool with an error. Use `list_apps` to see what is allowed.

### Logs

Logs are written to `logs/` (git-ignored):

- `companion.log`: the app log. Warnings also appear in the console.
- `tool_calls.jsonl`: one JSON record per tool call (timestamp, tool, arguments, status, result, duration). Tool calls are kept out of the console.

## Tests

```powershell
python -m pytest
```

Tests cover config loading and overrides (including the `audio` section), registry schema generation and argument validation, timeouts and the busy guard, confirmation (including the confirmer protocol and invalid answers), audit records, plugin loading, the allowlist and app tools, the notes tools (including `clear_notes`), the LLM tool-call loop (using a fake API, so no server is needed), and the speech pipeline (microphone capture, WASAPI-only device listing and resolution, VAD trimming, transcription with CUDA-to-CPU fallback, and push-to-talk). Speech tests use fakes, so they need no microphone or model download.

## Layout

```
config.yaml                 settings
src/companion/config.py     config loading and validation
src/companion/main.py       terminal chat loop
src/companion/llm/          OpenAI-compatible client and system prompt
src/companion/tools/        registry, plugin loader, and built-in tools
src/companion/safety/       confirmer protocol, terminal confirmer, and app allowlist
src/companion/audio/        microphone capture, VAD, faster-whisper, and push-to-talk
src/companion/logging_setup.py  file logging setup
plugins/                    user tool files, auto-loaded at startup
data/apps.json              app allowlist for open_app
data/notes/                 notes written by take_note (created on first use, git-ignored)
logs/                       runtime logs (git-ignored)
tests/                      pytest suite
```
