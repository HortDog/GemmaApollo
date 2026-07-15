# Inference server (`scribe infer-serve`)

The heavy engine (Whisper + Qwen corrector now, Gemma later) can run in a
separate process — or on a separate machine/OS — from the scribe app. The app
talks to it over a small HTTP API; everything else (mic, VAD, wake words,
DocState, undo, datalogger, web UI) stays in the app process.

```
┌─ app host ─────────────────────────────┐      ┌─ inference host ────────────────┐
│ scribe serve --engine remote           │ HTTP │ scribe infer-serve --engine s2l │
│  mic → SileroVAD → chunker → spotters  │─────►│  POST /v1/process[_text]        │
│  DocState · datalogger · /ws · UI      │◄─────│  GET /healthz /readyz           │
│  RemoteEngine (implements Engine)      │ JSON │  S2LEngine (lazy models)        │
└────────────────────────────────────────┘      └─────────────────────────────────┘
```

The default remains single-process (`scribe serve --engine s2l`) — the split
is opt-in. The boundary is the frozen `Engine` protocol
(`src/scribe/engine/base.py`): responses are `EngineResult` JSON **verbatim**,
so the wire format is frozen together with the Action schema. The WebSocket
protocol (PROTOCOL.md), frontend, and training-data logging are unchanged.

## Quickstart (two machines)

On the GPU box (any OS with an NVIDIA card; CPU works too, slower):

```bash
uv sync --extra s2l
uv run scribe infer-serve --engine s2l --host 0.0.0.0 --preload
```

On the machine you dictate at (no torch needed unless you use `--mic`):

```bash
uv sync --extra audio          # mic host; plain `uv sync` for typed-only
uv run scribe serve --engine remote --infer-url http://gpu-box:8018 --mic
```

`--infer-url` defaults to `$SCRIBE_INFER_URL` or `http://127.0.0.1:8018`.

## API

| Method / path | Request body | Success (200) |
|---|---|---|
| `GET /healthz` | — | `{"status":"ok","engine":"s2l","loaded":bool}` |
| `GET /readyz` | — | `{"ready":true}`; `503 {"ready":false,"detail":"loading"}` while `--preload` is warming |
| `POST /v1/process_text` | `{"transcript": str, "doc_context": str}` | `EngineResult` JSON |
| `POST /v1/process` | `{"audio_wav_b64": str, "doc_context": str}` | `EngineResult` JSON |

Errors: `422` malformed body (FastAPI validation) · `400` undecodable audio
(`audio_wav_b64` must be a base64 16-bit PCM **mono 16 kHz** WAV file — the
engine contract's format; the server rejects rather than resamples) · `501`
the engine raised `NotImplementedError` (mock's audio path, gemma stub) ·
`500 {"detail":"ExcType: msg"}` any other engine failure.

Audio crosses the wire as base64 WAV inside the JSON body: it reuses the same
`wav_bytes()` encoding the datalogger already persists, is half the size of
raw float32, and keeps `doc_context` in-band. A worst-case 10 s utterance is
~430 KB of JSON — negligible next to model inference time.

Requests are serialized server-side (one GPU, engines are not thread-safe).

## Warm-up

Models load lazily on the first request (which can take minutes on a cold
cache — the client's default timeout of 120 s covers a warm disk cache).
Prefer `--preload`: models load at startup and `/readyz` flips to 200 when
inference is actually ready.

## Security

The server binds `127.0.0.1` by default and has **no authentication** —
`--host 0.0.0.0` exposes it to your LAN and is a trust decision. Everything
still runs on machines you own; nothing changes about the project's
fully-local stance. Token auth is a possible follow-up if anyone needs it.
