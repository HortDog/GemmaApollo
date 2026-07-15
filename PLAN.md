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

## Phase 4 — Live audio: VAD + mic streaming ✅ (final spoken test = user)
- [x] `audio/vad.py`: Silero VAD chunker on a 16 kHz mic stream
      (sounddevice), emitting utterance arrays with configurable
      min/max length (0.5–10 s) and pre-roll padding. Pure UtteranceChunker
      (fake-VAD unit tests) + lazy SileroVAD/mic wrappers. min_s gates on
      speech duration, not buffer length (clicks stay dropped).
- [x] Backend owns the mic: `scribe serve --mic` runs mic→VAD→chunker in a
      daemon thread feeding the same utterance pipeline as typed input;
      engine inference runs off the event loop; all server frames broadcast
      to every connected client. Mic utterance wavs attach to pending
      proposals and land in the datalogger on commit/scratch.
      (Browser-mic ws `audio` frames remain the stretch goal.)
- [x] Status frames drive UI states: listening / heard / thinking / idle —
      state chip in the frontend header.
- **Accept:** verified with real Silero on fixtures: each clip → exactly 1
  utterance (2.3–7.1 s), 0 spurious utterances over 60 s silence AND 60 s
  low room noise; `--mic` serve announces `listening`, capture thread runs
  clean, typed utterances still round-trip. ⚠ The literal speak-into-the-mic
  check needs a human voice — run `uv run scribe serve --engine s2l --mic`
  and say "e equals m c squared".

## Phase 5 — Wake-word spotters ✅ runtime (model training = user/Colab step)
- [x] `audio/wakewords.py`: pure IntentSpotter (1280-sample chunking,
      per-intent thresholds, 2 s refractory) + lazy OWWScorer
      (openwakeword, onnx backend on Windows), running on every mic frame
      in parallel with the VAD chunker.
- [x] Training scripts + procedure in `tools/wakewords/`
      (piper-sample-generator positives, session-dictation negatives,
      oww training notebook).
- [x] Models TRAINED (2026-07-15, locally in WSL2 on the 4060 Ti — see the
      README's WSL gotcha list): 4000 piper-TTS positives + ACAV100M negative
      features each, 25k steps. Validation FP/hour: commit 0.18,
      scratch_that 0.0, undo 3.0 (⚠ short word — watch the app_intent audit
      log in real sessions; raise its IntentSpotter threshold if it misfires).
      Deployed to `models/wakewords/*.onnx` (+ `.onnx.data` sidecars —
      torch 2.9's exporter externalizes weights); server auto-loads them.
      Verified: 6/6 SAPI phrase clips fire exactly their own intent;
      0 fires over 60 s silence AND all 4 math-dictation fixtures.
- [x] Spotter hits bypass the engine entirely → shared app-intent path
      (same code as UI buttons) → ws status; every fire logged as
      `verdict: app_intent` for the false-positive audit.
- **Accept:** plumbing verified with the pretrained hey_jarvis model mapped
  to commit: 60 s silence → 0 fires; synthesized "hey jarvis" clip → exactly
  1 commit fire; intents consumed on a separate task so they work while the
  engine is busy. ⚠ The 10-minute-dictation false-positive test runs after
  the real models are trained (fires are logged; audit data/sessions).

## Phase 5.5 — Portability + inference split ✅ (issue #1)
- [x] OS-agnostic packaging: torch's CUDA index is marker-gated to
      Windows / x86-64 Linux; macOS and ARM Linux resolve CPU/MPS wheels
      from PyPI. `S2LEngine` device default is `"auto"` (cuda → cpu; the
      corrector may use MPS — CTranslate2 has none, so macOS ASR is CPU int8).
      CUDA warm-up hack now gated on the resolved device.
- [x] `scribe infer-serve`: standalone inference server hosting one engine
      behind HTTP (INFERENCE.md); `RemoteEngine` proxy implements the frozen
      Engine protocol over it (`scribe serve --engine remote --infer-url …`).
      Wire format is EngineResult JSON verbatim — schema untouched. VAD +
      spotters + datalogger stay in the app process.
- **Accept:** full ws loop green over the HTTP boundary with the mock engine
  (tests/test_remote_engine.py, tests/test_infer_server.py); existing 48
  tests unchanged. GPU s2l path needs a manual smoke test on the CUDA box.
  ⚠ `uv.lock` regeneration (`uv lock`) pending — the dev sandbox cannot
  reach download.pytorch.org; run it on the next `uv sync`.

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
