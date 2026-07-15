# GemmaApollo Scribe

A real-time, **fully local** math scribe. You dictate mathematics and it
appears line-by-line as rendered LaTeX; you edit it by voice ("change e2 to x
squared"), extend the pending equation piece by piece, type into any line with
MathQuill, and commit/undo with spoken wake words. Every utterance you accept
or reject becomes training data for **GemmaApollo** — a planned end-to-end
audio→edit-action model fine-tuned from Gemma.

Nothing leaves your machine: ASR, LaTeX correction, VAD, and wake-word
spotting all run on the local GPU (developed on 16 GB cards). Runs on
Windows, Linux, and macOS; the heavy models can optionally live in a separate
inference server process — even on another machine — see
[INFERENCE.md](INFERENCE.md).

## Quickstart

```bash
uv sync --extra dev                  # pure-logic dev (no models, mock engine)
uv run pytest                        # 48 tests, CPU-only
uv run scribe serve                  # http://127.0.0.1:8017, mock engine

# the real thing (first run downloads Whisper large-v3 + the corrector, ~4 GB)
uv sync --extra dev --extra s2l --extra audio
uv run scribe serve --engine s2l --mic
```

### Per-OS notes

| | |
|---|---|
| **Windows** | works as-is; CUDA torch wheels are pulled automatically on NVIDIA machines |
| **Linux** | sounddevice needs PortAudio: `sudo apt install libportaudio2`. NVIDIA x86-64 gets CUDA wheels; other machines get CPU torch |
| **macOS** | grant mic permission to your terminal app on first `--mic` run. faster-whisper has no MPS backend, so ASR runs CPU int8 — use `--asr-model distil-large-v3`, or better, run the engine remotely on a CUDA box (below) |

The install is split so each host only pulls what it runs: base deps serve
the app with a remote engine (no torch at all), `--extra audio` adds the mic
stack (CPU torch via Silero), `--extra s2l` adds the heavy models.

### Split inference (optional)

Run the models in their own process — or on a different machine/OS entirely:

```bash
uv run scribe infer-serve --engine s2l --preload    # GPU box (:8018)
uv run scribe serve --engine remote --mic           # app, dictation, UI
```

See [INFERENCE.md](INFERENCE.md) for the API, two-machine setup, and
security notes.

Open the page, then just talk:

- **"e equals m c squared"** → a pending LaTeX line appears
- keep talking → each utterance **extends** the pending equation (the
  corrector re-derives the whole equation from your accumulated speech)
- **"commit"** → the line lands · **"scratch that"** → removes the last piece
  · **"undo"** → reverts the last committed change
- "change the last line to x squared", "delete e2", "label e1 kinetic
  energy" — voice edits, with ordinal targeting ("line two", "the last line")
- click any line to edit it with MathQuill; the pending line can be targeted
  by voice before it's even committed

The wake words (commit / undo / scratch-that) are custom openWakeWord models
running continuously beside the VAD — they work even while the engine is busy
and never fire mid-equation.

## How it works

```
Mic ─► Silero VAD chunker (utterances, 0.5–10 s)
        ├─► openWakeWord spotters: commit/undo/scratch  → app intents, instant
        └─► Engine.process(audio, doc_context) → Action (JSON)
                                                   │
                                                   ▼
                    DocState applies; the app owns commit/undo/history
```

Two engines implement one frozen contract (`src/scribe/engine/base.py` —
`append_math | replace | insert | delete | set_label | text_reply | clarify`):

1. **S2LEngine** (now): faster-whisper ASR → transcript router → the
   [Speech-to-LaTeX](https://arxiv.org/abs/2508.03542) Qwen2.5 post-correction
   checkpoint. Non-agentic — so the *app layer* provides the agency:
   pending-equation composition, ordinal resolution, command grammar.
2. **GemmaEngine** (later): fine-tuned Gemma, end-to-end audio → Action tool
   call, trained on this app's own logs. The Action schema is its training
   target, which is why it is frozen.

Every processed utterance is logged as a training triple (audio + doc context
+ transcript + accepted/rejected Action) under `data/sessions/` — accepted
edits are positives, scratches are negatives, wake-word fires are audited.

## Commands

```bash
uv run scribe serve [--engine mock|s2l|remote] [--mic] [--mic-device N] [--port 8017]
uv run scribe infer-serve [--engine s2l] [--preload] [--port 8018]  # see INFERENCE.md
uv run scribe bench --engine s2l     # per-stage latency + CER on fixture clips
uv run scribe mics                   # list audio input devices
uv run scribe mic-test --seconds 10  # console level/VAD tester
uv run pytest                        # schema/router/docstate/ws tests (no GPU)
```

The web UI includes a mic device dropdown and a live level/VAD test bar when
serving with `--mic`.

## Status

| Phase | | |
|---|---|---|
| 0–2 | env, contracts, DocState, ws server + UI | ✅ |
| 3 | real S2L engine + bench (1.06 s median e2e, 4.9 GB VRAM) | ✅ |
| 4 | live audio: VAD chunker + backend mic | ✅ |
| 5 | wake-word spotters, custom-trained locally | ✅ |
| — | mic select/tester · pending-equation composition | ✅ |
| 6 | training-data export (`scribe export` → JSONL) | next |
| 7 | eval harness + real-voice corpus | |
| 8 | GemmaEngine swap | |

Full plan with acceptance criteria: [PLAN.md](PLAN.md).

## Repo map

- [`src/scribe/`](src/scribe/) — server, DocState, router, engines, audio
  (pure logic is torch-free and tested on CPU)
- [`frontend/index.html`](frontend/index.html) — single-file UI (KaTeX +
  MathQuill), server-driven over ws with an offline local mode
- [`PROTOCOL.md`](PROTOCOL.md) — the ws protocol (kept in sync with
  `server.py` and the frontend)
- [`models/wakewords/`](models/wakewords/) — trained spotter models
- [`tools/wakewords/`](tools/wakewords/) + [`docs/wakeword-training.md`](docs/wakeword-training.md)
  — reproducible wake-word training pipeline (WSL2) and its runbook
- [`tests/`](tests/) — schema, router, DocState, chunker, spotter, ws protocol

## Key references

- Speech-to-LaTeX paper: [arXiv:2508.03542](https://arxiv.org/abs/2508.03542)
  (datasets + the `marsianin500/*` correction checkpoints)
- Reference CLI this builds on: [DingoOz/speech-to-LateX](https://github.com/DingoOz/speech-to-LateX)
- [openWakeWord](https://github.com/dscripka/openWakeWord) ·
  [Silero VAD](https://github.com/snakers4/silero-vad) ·
  [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
