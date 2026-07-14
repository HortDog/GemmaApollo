# Implementation Plan

Work the phases in order. Each phase has acceptance criteria; do not start the
next phase until they pass. Pure-logic phases (1, 2) need no GPU.

## Phase 0 — Environment ✅
- [x] `uv sync` installs; `uv run pytest` runs (empty suite ok).
- [x] Vendor or pip-install the speech-to-LateX package (github.com/DingoOz/speech-to-LateX)
      as an optional dependency group `s2l` so pure-logic dev works without it.
      (Vendored: model-loading logic in `engine/s2l_engine.py`, CER/normalize in
      `metrics.py` — upstream pins bleeding-edge deps we don't want.)
- **Accept:** fresh clone → `uv sync && uv run pytest` green on CPU-only machine. ✅

## Phase 1 — Contracts + DocState (pure python) ✅
- [x] Finalize pydantic models in `engine/base.py` (already stubbed — extend, don't rename).
- [x] Implement `docstate.py`: apply each Action, stable e-IDs, label support,
      snapshot/undo stack (depth 50), `render_context()` producing
      `[e1] \frac{1}{2}mv^2  [e2] E = mc^2`.
- [x] Implement `router.py` transcript grammar (port the regexes from
      frontend/index.html `routeUtterance`, add ordinal targeting:
      "line two" → 2nd line's id; "the last line" → last id).
- **Accept:** `uv run pytest tests/test_docstate.py tests/test_router.py` green;
  includes cases: unknown target id → error Action rejected; undo after delete
  restores id ordering; ordinal + explicit id targeting both resolve. ✅

## Phase 2 — Server + WS protocol + mock engine ✅
- [x] `server.py`: FastAPI app, serves `frontend/`, ws endpoint `/ws`.
- [x] Implement `PROTOCOL.md` exactly (client sends utterance text or audio;
      server replies status/transcript/action frames).
- [x] `MockEngine` that echoes the frontend's toy converter behavior, so the
      full loop runs with zero models.
- [x] Modify `frontend/index.html`: when ws connects, `routeUtterance` sends to
      server instead of local parsing; falls back to local mode if no ws.
- **Accept:** `uv run scribe serve`, open browser, dictate via text box →
  pending line appears via ws round-trip; Action console shows server-issued
  Actions; disconnect server → UI falls back to local mode. ✅ (verified live
  over ws incl. error containment, ordinal commands, scratch/undo)

## Phase 3 — S2LEngine (real models) ✅ (CER validation deferred to Phase 7)
- [x] `engine/s2l_engine.py`: faster-whisper (int8) + the marsianin500
      post-correction checkpoint, lazy-loaded. Config flag to swap
      large-v3 / large-v3-turbo / distil-large-v3 (`--asr-model` on bench).
      NOTE: ct2 must run one warm-up transcribe BEFORE torch loads the
      corrector, else torch matmuls die with CUBLAS_STATUS_EXECUTION_FAILED
      (Windows, ct2 4.8 + torch 2.11 in one process) — see _lazy_load().
- [x] Command path: transcript classified as command → prompted local LLM
      (reuse the Qwen weights already resident; a strict JSON-only prompt
      emitting one Action) with regex-grammar fast path first.
- [x] `scribe bench`: fixture wavs (2 vendored from the reference repo's
      MMS-TTS clips + 2 generated with Windows SAPI) through the engine;
      per-stage latency (asr / route / total) + CER vs labels.json.
- **Accept:** measured on the RTX 4060 Ti 16 GB (actual hardware; CLAUDE.md
  says 5070 Ti): VRAM peak 4.9 GB < 10 GB ✅; median end-to-end 1.06 s < 1.5 s ✅.
  CER on the 4 synthetic fixtures: 0.34 whitespace-stripped (best clip 0.08) —
  NOT yet comparable to the paper's numbers; the synthetic voices are partly
  out-of-distribution and one clip's ASR mishears the TTS ("minus squared x").
  Real CER comparison = run the reference eval on the marsianin500/Speech2Latex
  HF test split, which is exactly Phase 7's eval harness. ⚠ carried forward.

## Phase 4 — Live audio: VAD + mic streaming
- [ ] `audio/vad.py`: Silero VAD chunker on a 16 kHz mic stream
      (sounddevice), emitting utterance wavs with configurable
      min/max length (0.5–10 s) and padding.
- [ ] Browser sends mic audio over ws (Opus/PCM frames) OR backend owns the
      mic directly (simpler, single-user local app — prefer backend mic,
      keep browser-mic as a stretch goal).
- [ ] Status frames drive UI states: listening / heard (interim transcript) /
      thinking / action.
- **Accept:** speak "e equals m c squared" into the mic → pending line within
  ~2 s; interim transcript visible; silence produces no spurious utterances
  over a 60 s idle test.

## Phase 5 — Wake-word spotters
- [ ] `audio/wakewords.py`: openWakeWord models for commit / undo /
      scratch-that running continuously on the raw stream (parallel to VAD).
- [ ] Train spotters with piper-sample-generator; negatives from recorded
      math dictation (must never fire mid-equation). Training scripts in
      `tools/wakewords/`.
- [ ] Spotter hits bypass the engine entirely → app-layer intent → ws status.
- **Accept:** commit/undo/scratch by voice work while the engine is busy
  processing an utterance; false-positive rate ~0 over a 10-minute dictation
  session (log every fire).

## Phase 6 — Training-data logger
- [ ] `datalogger.py`: per session dir under `data/sessions/<ts>/`:
      `NNN.wav` + `NNN.json` {doc_context, transcript, engine_action,
      final_action, verdict: committed|scratched|edited_after, engine, latency_ms}.
- [ ] "edited_after": if the user MathQuill-edits a line within N seconds of
      its commit, attach the corrected LaTeX as gold label.
- [ ] `scribe export` → JSONL in the (audio, doc_state) → Action format from
      the handover doc §5, ready for the GemmaApollo training mix.
- **Accept:** a 5-minute session yields a valid JSONL that a schema test
  round-trips; scratched utterances retained as negatives.

## Phase 7 — Eval + real-voice set
- [ ] `eval/`: adapt the repo's eval script; metrics per handover doc §7:
      tool-name acc, target-id acc, payload SymPy-equivalence + exact match.
- [ ] `tools/recorder.py`: prompt-flashing recorder to capture the
      real-voice corpus (500–1000 clips) and the held-out eval slice.
- **Accept:** `scribe eval --set realvoice` prints the three metrics split by
  category (dictation / edit / conversational / negative).

## Phase 8 — GemmaEngine swap (blocked on training, keep interface ready)
- [ ] `engine/gemma_engine.py`: load fine-tuned Gemma 4 12B (4-bit), build the
      prompt from doc context + system prompt (thinking-mode suppressed),
      parse Gemma function-call output → Action.
- [ ] Engine selector in config + UI dropdown actually switches engines.
- **Accept:** identical UI behavior with `--engine gemma`; eval harness runs
  against both engines and prints a comparison table.

## Non-goals (for now)
- Multi-user / auth / cloud anything.
- Streaming token-level transcription display.
- Full LaTeX document export (plain concatenation is fine until Phase 7+).
