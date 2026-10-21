# Manual tests: speech input

Automated tests cover the logic (`python -m pytest`). These checks need a real microphone, speakers, LM Studio running, and a person speaking. Run them after changes to speech input, the LLM client, or the voice loop.

## Before you start

```powershell
cd "F:\AI Companion"
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
python -m companion
```

- LM Studio server is running with the model in `llm.model` loaded.
- `audio.input_device` in `config.yaml` names a WASAPI microphone (check with `/mics`).
- Expect the first run to download the Whisper model.

## 1. Continuous listening

1. Start the companion. It should print `Listening. Speak anytime.`
2. Say a full sentence, then pause for about a second.
3. Expected: `Heard: "..."` appears with your sentence, then a reply streams in.
4. Say a second sentence without pressing anything.
5. Expected: the second sentence is also heard and answered. No Enter is needed.

Pass: both sentences are transcribed and answered in order, with no keypress.

## 2. Streaming replies

1. Ask for a longer answer, for example "Explain how a rainbow forms in about five sentences."
2. Expected: text appears word by word while the model is still generating, not all at once at the end.
3. Expected: no `<think>` text is shown.

Pass: text streams, and the final text has no thinking block.

## 3. Barge-in

1. Ask for a long answer (as in check 2).
2. While the reply is streaming, speak clearly for about one second.
3. Expected: the reply stops, `(interrupted)` is shown, and your new speech is heard and answered.
4. Expected: a short cough or click does not stop the reply (it is below `min_speech_ms` and `barge_in_ms`).
5. Ask a follow-up such as "what did you just say?" Expected: the model knows the reply was cut off.

Pass: sustained speech stops the reply within about half a second. A cough does not.

## 4. Low-confidence display

1. Mumble, or speak quietly from a distance.
2. Expected: a transcript printed as `Heard (low confidence): "..."` when the average log-probability is below `low_confidence_logprob` (default -1.0).
3. Expected: the text is still sent to the model.
4. Optional: lower `low_confidence_logprob` to `-0.5` and confirm more transcripts are flagged. Restore the setting afterwards.

Pass: quiet or unclear speech is flagged. Clear speech is not.

## 5. Debug audio saving

1. Set `audio.save_utterances: true` in `config.yaml`. Restart.
2. Say one sentence.
3. Expected: a new `.wav` file appears in `logs/utterances/`. Play it back: your voice is clear, at 16 kHz mono, and includes a little silence before and after.
4. Expected: `logs/transcripts.jsonl` has a new line with the same text.
5. Set `save_utterances: false` again and restart.

Pass: the WAV plays correctly and matches what was heard. No new WAV files are written when the setting is false.

## 6. Microphone list shows WASAPI only

1. Type `/mics`.
2. Expected: every entry is a WASAPI device. No MME, DirectSound, or WDM-KS entries appear. Each physical microphone appears once.
3. Expected: the default input device is marked, and the numbers match `audio.input_device` where it is set.

Pass: only WASAPI devices, with no duplicates.

## 7. Clean Ctrl+C and /exit

1. With the companion listening, press Ctrl+C.
2. Expected: the program exits within a second or two, with no traceback and no hang. The prompt returns.
3. Start again. Start a reply, then press Ctrl+C during it. Expected: same clean exit.
4. Start again, type `/exit`. Expected: clean exit.
5. Start again, press Enter (pause), then Enter (resume), then Ctrl+C. Expected: clean exit.

Pass: all four exits are clean and the terminal is usable afterwards.

## 8. Audio diagnostic and playback

1. Run `python -m companion.diagnose_audio` (with `$env:PYTHONPATH = "src"`).
2. Expected: the WASAPI device list is shown and the configured device is marked `*`.
3. When it says `Speak now.`, speak a normal sentence for the full 5 seconds.
4. Expected: `Peak` and `Average level` are printed, and a `Saved:` path under `logs/diagnostics/`.
5. Play the WAV back. Expected: your voice is clear, with no crackle or distortion.
6. Expected: no `Warning:` line. Near silence means the wrong device or a muted mic. Clipping means lower the mic gain.

Pass: the WAV plays back your sentence clearly, and the levels are not near silence or clipping.

## 9. First word is not clipped

1. Start the companion in continuous mode.
2. Speak a sentence that starts with a quiet word (for example "Hmm, what time is it?") after a short pause.
3. Expected: `Heard: "..."` begins with the first word. The first word is not missing or cut in half.
4. Optional: enable `audio.save_utterances` and check the start of the WAV. It should include a little silence before the speech.

Pass: the first word is present in the transcript on three tries out of three.

## 10. Confirmation: yes, no, timeout, Enter, Ctrl+C

The companion has no `delete_note` tool. Use `clear_notes`, which requires confirmation. Ask it to clear notes (for example "clear all my notes").

1. **Yes:** press `y`. Expected: `Allowed.` and the tool runs. Note the result.
2. **No:** repeat the request, then press `n`. Expected: `Declined.` and the notes are unchanged.
3. **Timeout:** repeat the request and wait without answering for `input.confirm_timeout_s` seconds (default 30). Expected: the request is declined, and `companion.log` has a line `confirmation timed out after 30 s; declined`.
4. **Enter does nothing:** repeat the request and press Enter. Expected: no pause, no decline, and no echoed text. The question is still waiting. Press `n` to finish.
5. **Ctrl+C cancels:** repeat the request and press Ctrl+C. Expected: `Confirmation cancelled. Declined.` The companion keeps running. Ask a normal question to confirm.
6. **Double Ctrl+C quits:** repeat the request and press Ctrl+C once. Expected: `Confirmation cancelled. Declined.` and the companion keeps running. Press Ctrl+C again within 2 seconds. Expected: the program exits within about a second. It does not wait for the timeout.

Pass: each path shows the right result, and every decline or answer is logged in `companion.log`.

## 11. Speech output (Phase 4, Chatterbox-Turbo, Chatterbox-Nano, Piper)

Setup: follow the README "Speech output" section for both Piper and the `.venv-tts` Chatterbox environment (pinned in `requirements-tts.txt`). Run `python -m companion.diagnose_tts --engine chatterbox_turbo` first and confirm it plays the test sentence with a laugh. Then run it with `--engine chatterbox_nano --device cuda` and with `--engine piper` to confirm the other voices.

1. **Headphones, first sentence:** start the companion in continuous mode with `output_mode: headphones`. Ask a question with a long answer. Expected: speech starts after the first sentence finishes streaming, not after the whole reply.
2. **No self-reading:** the reply is not spoken as Markdown symbols, code, or URLs, and no `<think>` text is spoken.
3. **Voice tags:** ask something that invites a reaction (for example, a joke). Expected: a real laugh or chuckle is heard where the tag was, and the tag text (`laugh`, brackets) is never spoken. Unknown tags such as `[banana]` are silent and do not break the sentence.
4. **Tag display:** with `display.show_voice_tags: true`, the tag shows on screen. With `false`, it is hidden on screen too.
5. **Barge-in:** while a reply is speaking, speak for `barge_in_ms`. Expected: speech stops promptly, `(stopped speaking)` is printed, and the next answer shows the reply truncated to the spoken part followed by `[interrupted]` (ask "what did you just say?" to check). Speech that was already synthesized during the interruption must not play late.
6. **Barge-in mid-sentence:** interrupt while a long sentence with a tag is being spoken. Expected: no late audio from that sentence after the stop.
7. **Speakers, no self-interrupt:** set `output_mode: speakers`, then ask a question. Expected: the companion speaks the whole reply without stopping itself, and barge-in does not fire from its own voice.
8. **Speakers tail:** after the reply ends, wait `speakers_tail_ms` before speaking. Expected: your first word is captured, not clipped.
9. **Fallback to Piper:** set `tts.chatterbox.python` to a path that does not exist, then start the companion. Expected: a message says Chatterbox is unavailable and falls back to Piper, replies are spoken with Piper, and tags are removed from the text. Restore the setting afterwards.
10. **Chatterbox autostart and shutdown:** start the companion with Chatterbox. Expected: the server starts on port 8765 and reports ready within about 40 s. After `/exit`, no `chatterbox_server` python process remains (check in Task Manager or with `Get-CimInstance Win32_Process`).
11. **Ctrl+C during speech:** press Ctrl+C once while it speaks. Expected: speech stops and the companion keeps running.
12. **Text and push-to-talk modes:** set `input.mode: text`. Expected: replies are printed and not spoken.
13. **Latency:** note the time from your question to the first audible word, over three questions, for both Chatterbox and Piper (set `tts.engine` to switch). Record the numbers in the table below.

14. **Engine comparison:** with the companion stopped, run `python -m companion.diagnose_tts --compare`. Expected: four engine and device rows print (Turbo cuda, Nano cuda, Nano cpu, Piper cpu), each with a short, long, and `[laugh]` WAV in `logs/diagnostics`. Listen to the Turbo and Nano cuda `laugh` WAVs and confirm the laugh is audible and the tag word is not spoken.
15. **Nano engine:** set `tts.engine: chatterbox_nano` and repeat items 1 and 3. Expected: same behavior as Turbo, with the same tags. Restore `chatterbox_turbo` afterwards.
16. **Offline start:** with `tts.offline: true` and both Chatterbox models cached, disconnect from the network and start the companion with Chatterbox. Expected: the server reports ready within the usual load time, with no network errors.
17. **Speed warning:** set `audio.speech_speed: 0.8` with a Chatterbox engine. Expected: a startup warning says the setting applies to Piper only. Restore `1.0` afterwards.
18. **Chatterbox restart and Piper fallback:** with the companion running on Chatterbox, stop the Chatterbox server (end the `chatterbox_server` python process in Task Manager) and ask a question. Expected: a notice says Chatterbox is restarting; it restarts once and the reply is spoken. Stop it again and ask another question. Expected: a notice says it is switching to the Piper voice, the reply is spoken with Piper, and later replies stay on Piper for the session. Piper needs its voice file in `audio.voice_path`.

Pass: items 1 to 18 behave as described, and latency is recorded for both engines.

Measured on this machine (automated, `diagnose_tts --compare`, RTF = synthesis time / audio length, lower is faster; GPU MB is the server's reported peak):

| Engine | Device | Load s | Short synth s / audio s / RTF | Long synth s / audio s / RTF | Laugh synth s / audio s / RTF | GPU MB |
|---|---|---|---|---|---|---|
| chatterbox_turbo | cuda | 22.80 | 1.42 / 3.04 / 0.47 | 4.07 / 9.08 / 0.45 | 1.37 / 2.96 / 0.46 | 2937 |
| chatterbox_nano | cuda | 19.33 | 0.70 / 2.16 / 0.33 | 2.14 / 8.00 / 0.27 | 0.95 / 3.28 / 0.29 | 1971 |
| chatterbox_nano | cpu | 18.80 | 3.09 / 2.00 / 1.54 | 8.92 / 7.44 / 1.20 | 4.59 / 3.60 / 1.27 | n/a |
| piper | cpu | 3.23 | 0.11 / 2.18 / 0.05 | 0.29 / 8.09 / 0.04 | 0.10 / 2.64 / 0.04 | n/a |

Live server checks (not by ear), on the pinned commit, for both Turbo and Nano on cuda: `/health` returns status `ready` with the same 19 tags (advertisement through whispering). `/synthesize` at generation 5 returns 200 with float32 audio. After `/cancel` at generation 6, a generation 5 request returns 409 ("generation 5 was cancelled by 6"), and generation 6 returns 200. A clean venv built from `requirements-tts.txt` plus the pinned chatterbox commit gives the same Turbo results.

Not yet checked by ear: whether the `laugh` tag is audible in Nano output. Item 14 covers it.


## 12. Conversation, persona, and memory (Phase 5)

Setup: LM Studio running with the model loaded. Start the companion in text mode (`input.mode: text`) so the checks are easy to read.

- **Persona:** edit [data/persona.md](data/persona.md) to a new name, type `/new`, and ask the companion its name. Expected: it uses the new persona. Restore the file afterward.
- **Remember and recall:** say "remember that my dog is called Biscuit". Expected: a confirmation and a saved memory. Start `/new`, then ask "what is my dog called?". Expected: the answer names Biscuit.
- **Secret refusal:** say "remember my password is hunter2". Expected: refused; nothing is saved.
- **List and forget:** say "list my memories". Expected: a numbered list printed by the app, with the count in the header. Ask to forget "dog". Expected: a confirmation prompt; answer no, then yes; the memory is gone.
- **Status:** type `/status`. Expected: model, context use, saved memory count, and the log path after the first reply.
- **Log:** check that a file in `logs/conversations/` contains your turns and the replies.
- **Server down:** stop LM Studio's server and send a message. Expected: a short "cannot reach" message after a few seconds, no crash. Restart the server; the next message works and the earlier history is still there.
- **Long conversation:** lower `llm.compact_at` to 0.3 for this check. Send several long messages. Expected: no error; `/status` still works; the reply quality stays reasonable.
- **Voice:** in continuous mode, say the remember and recall phrases. Expected: the same results as in text.
- **Remember in two messages:** send "remember" alone. Expected: the reply asks what to remember. Send "my car is a blue Honda" next. Expected: it is saved, and the reply confirms only after the save. Check with "list my memories" that the text is there.
- **Remember the previous message:** send "my sister lives in Leeds", then "remember this". Expected: the previous message is saved.
- **Auto-save:** with `memory.auto_save: true`, state a fact about yourself ("I have two cats") and do not ask to remember it. Expected: "list my memories" shows it, and the database row has source `auto`. Set `memory.auto_save: false`, repeat, and expect nothing new to be saved.
- **Auto-save skips questions:** ask "do I have any cats?". Expected: nothing new is saved.
- **Duplicate remember:** send "remember my dog is called Biscuit" twice, the second time with different capitals or a final period. Expected: the second reply says "Already saved as memory N", and "list my memories" shows one copy.
- **Forget by number:** with at least three memories listed, say "forget 2-3". Expected: a read-back naming those numbers and asking yes or no. Say "no"; nothing is deleted. Say "forget 99". Expected: "There is no memory 99."
- **Admin dedupe (dry run):** run `python -m companion.memory_admin dedupe`. Expected: it lists groups or says there are no duplicates, and deletes nothing. Run it again with `--apply` only after a backup has been reviewed; a file appears in `data/memory/backups/`.
- **Forget that:** after an auto-save, say "forget that". Expected: the latest auto-saved memory is deleted at once, with no read-back.
- **Forget read-back:** say "forget cats". Expected: the reply reads back the matching text and asks yes or no. Answer "yes" to delete it; repeat and answer "no" to keep it. Any other reply is handled as a new message, and nothing is deleted.
- **Synonym recall:** save "I work as a nurse in Leeds". In a new conversation (`/new`), ask "what is my job?". Expected: the nurse memory is recalled through the semantic match, even though the words "job" and "nurse" do not match.
- **Embeddings off:** set `memory.embeddings: false`, restart, and repeat the synonym check. Expected: no semantic match; keyword recall still works.
- **Embedding model missing:** set `memory.embedding_cache` to an empty folder without internet. Expected: a notice says the model is unavailable, and keyword recall still works.

## 13. Polish: timers, media, files, tray, startup, wake word (Phase 6)

Setup: `pip install -r requirements.txt`. For voice checks, use `input.mode: continuous` and a working microphone.

- **Timer:** say or type "set a timer for 0.5 minutes called tea". Expected: a confirmation with a timer number. After about 30 seconds, a `* Timer done: tea.` notice appears before the next prompt (text) or is spoken (voice).
- **Cancel:** "list my timers", then "cancel timer <number>". Expected: the listing shows the timer, and the cancel confirms it. The timer does not fire.
- **Reminder:** "remind me at a time two minutes from now to stretch". Expected: a reminder set for that clock time. Use a short gap to check it fires.
- **Volume:** say "turn the volume up". Expected: the system volume rises by about 4 percent. "Mute" toggles mute. Undo both afterward.
- **Media:** with music or a video playing, say "pause". Expected: playback toggles. Try next and previous.
- **Find file:** ask "find my file called <part of a real file name>" for a file in Documents. Expected: the path is listed.
- **Find folder (Music):** ask "find my music folder". Expected: the first result is your Music folder, marked "(folder)".
- **Open folder:** ask "open my music folder". Expected: a confirmation prompt. Answer yes; Explorer opens the Music folder. Answer no on another try; nothing opens.
- **Open file:** ask to open that file. Expected: a confirmation prompt. Answer no; nothing opens. Repeat and answer yes; the file opens in its default program.
- **Refused file types:** ask to open an `.exe` or `.bat` file in your roots. Expected: the confirmation prompt appears, then after you answer yes the call is refused with the code-file message. Nothing runs.
- **Outside roots:** ask to open a file on another drive by full path. Expected: refused as outside your file folders.
- **Web search off:** with the default config, ask to search the web. Expected: "Web search is turned off".
- **Web search on:** set `tools.web_search_enabled: true`, restart, search "python release notes". Expected: numbered titles with links. Turn it back off after the test.
- **Tray:** set `display.tray_icon: true`, start voice mode, and check that the icon appears in the notification area. Right-click: "Open logs folder" opens `logs`. "Quit" stops voice mode and the process exits with no hang. Check `logs/companion.log` for a clean shutdown.
- **Startup on sign-in:** run `python -m companion.startup --enable`, then `--status`. Expected: enabled. Sign out and back in (or run the stored command from the Run key by hand). Expected: no console window, the tray icon appears, and `logs/startup-console.log` has no traceback. Run `--disable`, then `--status` shows disabled.
- **Wake word:** set `input.wake_word.enabled: true`, start voice mode, and wait for "Wake word on". Say nothing and speak a request: nothing should happen. Say "hey jarvis", pause, then ask for the time. Expected: the request is answered. Wait longer than `window_s`, then speak without the wake word: nothing should happen.
- **Wake word false triggers:** talk normally for a few minutes with the wake word enabled. Expected: no unexpected replies. If there are some, raise `threshold` to 0.6 and retest.
- **Tool log:** check that `logs/tool_calls.jsonl` has one record per tool call above, with status ok, declined, or failed as expected.

## Other modes (regression)

- `input.mode: push_to_talk`: Enter starts a recording, Enter stops it, and the transcript is sent.
- `input.mode: text`: no microphone is opened, and typed chat works.
- Confirmation in text mode: ask the model to clear notes. Expected: it asks in the terminal; answering "no" or "yes" works. This mode still uses the blocking prompt, so there is no timeout.

## Recording results

| Check | Date | Result | Notes |
|---|---|---|---|
| 1. Continuous listening | | | |
| 2. Streaming replies | | | |
| 3. Barge-in | | | |
| 4. Low-confidence display | | | |
| 5. Debug audio saving | | | |
| 6. /mics WASAPI only | | | |
| 7. Clean Ctrl+C and /exit | | | |
| 8. Audio diagnostic and playback | | | |
| 9. First word is not clipped | | | |
| 10. Confirmation paths | | | |
