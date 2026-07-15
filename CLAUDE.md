# GemmaApollo Scribe

Real-time, fully-local math scribe: the user dictates mathematics which appears
line-by-line as LaTeX, edits it conversationally by voice ("change e2 to x
squared"), types directly into any line with MathQuill, and commits/undoes
changes with voice keywords.

## Architecture (do not deviate without asking)

```
Mic ─► VAD chunker (1–3 s utterances)
        ├─► openWakeWord spotters: commit / undo / scratch-that  → app layer, instant
        └─► Engine.process(audio, doc_state) → Action
                                                  │
                                                  ▼
                          DocState applies Action; app owns commit/undo/transactions
```

Two engines implement one interface (`src/scribe/engine/base.py`):

1. **S2LEngine** (now): audio → Whisper-large-v3 transcript → router →
   - dictation → Qwen2.5 post-correction checkpoint (`marsianin500/*`, from
     the ICLR 2026 Speech-to-LaTeX paper, arXiv:2508.03542) → LaTeX
   - command → prompted text LLM → edit-op Action
   Existing working CLI to reuse/vendor: https://github.com/DingoOz/speech-to-LateX
2. **GemmaEngine** (later): fine-tuned Gemma 4 12B, end-to-end audio → Action
   tool call. Placeholder only for now; the Action schema is its training
   target, so **the schema is frozen** — changes require explicit sign-off.

The Engine may run in-process (default) or behind `scribe infer-serve` via
`RemoteEngine` (`--engine remote`) — an HTTP proxy implementing the same
interface; the wire format is EngineResult JSON verbatim, frozen with the
schema (see INFERENCE.md). VAD + wake-word spotters always stay in the app
process. OS support: Windows / Linux / macOS (macOS ASR is CPU-only —
CTranslate2 has no MPS backend).

## Hard rules

- `commit`, `undo`, `scratch that` are NEVER engine Actions. They are
  app-layer intents (wake-word spotters / UI buttons). The engine only
  proposes edits; DocState owns state and history.
- Action schema is the contract (see `engine/base.py`):
  `append_math | replace | insert | delete | set_label | text_reply | clarify`.
  Frontend, backend, logger, and both engines all speak exactly this JSON.
- Raw LaTeX is the source of truth for every line. MathQuill is an editor
  *view*; lines it cannot represent (align, matrices) fall back to plain-text
  editing. Never store MathQuill-internal state.
- Every processed utterance is logged as a training triple
  (audio wav + doc context + transcript + final accepted Action + user verdict)
  via `datalogger.py`. This is the GemmaApollo dataset. Do not skip logging.
- Everything runs locally. Target GPU: RTX 5070 Ti 16 GB. VRAM budget for the
  S2L engine: Whisper via faster-whisper int8 (~3 GB; try large-v3-turbo /
  distil-large-v3 and benchmark) + Qwen2.5 4-bit (~5 GB) + openWakeWord.
- 16 kHz mono audio end to end (matches both Whisper and Gemma 4 frontends).

## Commands

```
uv sync                                   # env
uv run scribe serve                       # FastAPI + ws on :8017, serves frontend/
uv run scribe infer-serve --engine s2l    # standalone inference server on :8018 (INFERENCE.md)
uv run scribe bench --engine s2l          # latency benchmark on fixture clips
uv run pytest                             # tests (schema, router, docstate are pure-python)
```

## Conventions

- Python 3.11+, uv-managed, pydantic v2 models for all wire types.
- Pure logic (router, docstate, schema) has no torch imports — must be
  testable on CPU with no models downloaded.
- Model-loading code is lazy and behind the Engine interface only.
- WebSocket protocol is defined in `PROTOCOL.md`; keep it in sync with
  `frontend/index.html` and `server.py`.
- Tests first for router grammar and DocState edge cases (unknown target IDs,
  undo across pending, empty doc).

## Key references

- Paper: arXiv:2508.03542 (S2L datasets + post-correction checkpoints;
  66k-sample open dataset — candidate training mix for GemmaApollo)
- Working CLI to build on: github.com/DingoOz/speech-to-LateX
- Unsloth Gemma 4 docs (for the later engine): unsloth.ai/docs/models/gemma-4/train
- Full project handover doc: docs/realtime-math-scribe-handover.md
- Phased plan with acceptance criteria: PLAN.md
