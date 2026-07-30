# Deployment: Electron clients + Docker model server over Tailscale

The 3-tier topology (branch `electron-3tier`):

```
┌ client machine ──────────────────────────┐      ┌ GPU box ─────────────────┐
│ Electron shell (desktop/)                │      │ Docker: model server     │
│  └► scribe-server.exe (frozen sidecar)   │ HTTP │  Whisper + Qwen S2L      │
│      app server: DocState, VAD, wake     ├──────┤  /process /process_text  │
│      words, datalogger, ws, frontend     │ tail │  /health /warmup         │
│  renderer = frontend/index.html          │ net  │  /log  ◄─ central        │
│  mic: getUserMedia → 16 kHz ws frames    │      │  GemmaApollo dataset     │
└──────────────────────────────────────────┘      └──────────────────────────┘
```

No audio streams across the network: VAD, wake words, and chunking run in the
sidecar on each client (CPU, torch-free). The only tailnet traffic is a
per-utterance wav POST → `EngineResult` JSON, plus spooled training-row
uploads to `/log`. Plain HTTP is fine on this hop — the tailnet is
WireGuard-encrypted, and a desktop app has no browser secure-context rules.
Keep everything tailnet-only: no router port-forwards, never
`tailscale funnel`.

## GPU box: model server (Docker)

Prereqs: Docker Desktop (WSL2 backend) + a current NVIDIA driver; Tailscale
with MagicDNS enabled.

```powershell
docker compose -f docker/compose.yaml up --build -d
docker compose -f docker/compose.yaml logs -f     # watch the --preload warmup
```

- First build downloads ~3 GB of cu128 wheels; first run downloads Whisper
  large-v3 + the Qwen corrector into the `hf-cache` volume (survives
  rebuilds). Confirm GPU access with
  `docker compose -f docker/compose.yaml exec model-server nvidia-smi`
  before debugging anything else.
- The central dataset lands on the host at `data/sessions/<client_id>/…`
  via the bind mount — this is the canonical GemmaApollo training store.
- Clients reach it at `http://<gpu-box-magicdns-name>:8018` (the host port
  publish is visible on the host's Tailscale address). Health check:
  `curl http://<gpu-box>:8018/health`.
- Without Docker: `uv run scribe model-server --engine s2l --preload`
  (add `--host <tailscale-100.x-ip>` for cross-machine use).

## Client machines: desktop app

Build once on the dev box:

```powershell
tools/build/build_sidecar.ps1        # PyInstaller-freeze the sidecar (~193 MB, torch-free)
cd desktop; npm install; npm run dist
```

Install `desktop/dist/GemmaApollo Scribe Setup <ver>.exe` (per-user, no
admin). First launch: menu **Scribe → Model server…** → set
`http://<gpu-box>:8018`, then **Start mic** in the UI. Windows may need
Settings → Privacy → Microphone → "let desktop apps access" enabled.

Client state lives in `%LOCALAPPDATA%\GemmaApolloScribe\` (local training
rows + spool, `client_id`) and `%APPDATA%\gemmapollo-scribe-desktop\`
(settings.json). Rows upload to the GPU box automatically and move to
`uploaded/` locally; while the box is unreachable they queue and retry.

## Dev-mode notes

- `cd desktop; npm start` runs the shell against `uv run scribe serve`
  from the checkout (no freeze needed). In a VS Code terminal, clear
  `ELECTRON_RUN_AS_NODE` first — VS Code exports it and it breaks Electron:
  `Remove-Item Env:ELECTRON_RUN_AS_NODE; npm start`.
- Browser instead of Electron: `uv run scribe serve --engine remote` and
  open `http://localhost:8017` (localhost is a secure context, so the
  browser mic works there without TLS).
- The old single-process modes still work: `scribe serve --engine s2l --mic`
  (sounddevice mic, in-process inference) and `--engine mock` for GPU-less
  dev.
- Remote *browser* access (no desktop app) needs HTTPS for getUserMedia:
  `tailscale serve --bg 8017` gives `https://<box>.<tailnet>.ts.net` with a
  tailnet cert proxying to loopback. Optional — the Electron app makes this
  unnecessary.
