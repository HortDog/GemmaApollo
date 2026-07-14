"""FastAPI app: serves frontend/ and the /ws endpoint per PROTOCOL.md.
Server owns DocState (single source of truth) and the SessionLogger.
Phase 2: full loop with MockEngine. Phase 4: backend-owned mic — VAD-chunked
utterances feed the same pipeline as typed ones; all server frames broadcast
to every connected client (single-user app, possibly several tabs)."""
from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import TypeAdapter

from .datalogger import SessionLogger
from .docstate import DocState
from .engine.base import Action, Clarify, TextReply
from .engine.mock_engine import MockEngine

ACTION_ADAPTER = TypeAdapter(Action)

def make_engine(name: str):
    if name == "mock":
        return MockEngine()
    if name == "s2l":
        from .engine.s2l_engine import S2LEngine
        return S2LEngine()
    if name == "gemma":
        from .engine.gemma_engine import GemmaEngine
        return GemmaEngine()
    raise ValueError(name)

def build_app(engine_name: str = "mock", mic: bool = False) -> FastAPI:
    app = FastAPI(title="GemmaApollo Scribe")
    engine = make_engine(engine_name)
    doc = DocState()
    logger = SessionLogger()
    clients: set[WebSocket] = set()
    pending: dict = {}  # pending_id -> {"action", "transcript", "audio_wav"}
    counter = {"n": 0}

    # ------------------------------------------------------------ transport
    async def send(ws: WebSocket, obj: dict):
        await ws.send_text(json.dumps(obj))

    async def broadcast(obj: dict):
        dead = []
        for ws in clients:
            try:
                await send(ws, obj)
            except Exception:
                dead.append(ws)
        for ws in dead:
            clients.discard(ws)

    # ------------------------------------------------------------ doc ops
    async def apply_and_broadcast(action: Action, verdict: str,
                                  transcript=None, audio_wav: bytes | None = None,
                                  source="voice"):
        # Training triples are (doc context -> action): log the context the
        # action was issued against, i.e. BEFORE the mutation.
        ctx_before = doc.render_context()
        try:
            assigned = doc.apply(action)
        except KeyError as e:
            await broadcast({"type": "error", "message": str(e),
                             "action": action.model_dump()})
            logger.log(audio_bytes=audio_wav, doc_context=ctx_before,
                       transcript=transcript, engine_action=action.model_dump(),
                       final_action=None, verdict="discarded",
                       engine=engine.name, latency_ms={})
            return
        logger.log(audio_bytes=audio_wav, doc_context=ctx_before,
                   transcript=transcript, engine_action=action.model_dump(),
                   final_action=action.model_dump(), verdict=verdict,
                   engine=engine.name, latency_ms={})
        await broadcast({"type": "applied", "action": action.model_dump(),
                         "assigned_id": assigned,
                         "doc_context": doc.render_context()})

    async def resolve_pending(pending_id: str, verdict: str):
        """Commit or scratch a pending proposal. Shared by `resolve` frames and
        `commit`/`scratch` intents (buttons now, voice spotters in Phase 5)."""
        p = pending.pop(pending_id, None)
        if not p:
            return
        if verdict == "commit":
            await apply_and_broadcast(p["action"], "committed",
                                      p["transcript"], p["audio_wav"])
        else:
            logger.log(audio_bytes=p["audio_wav"],
                       doc_context=doc.render_context(),
                       transcript=p["transcript"],
                       engine_action=p["action"].model_dump(),
                       final_action=None, verdict="scratched",
                       engine=engine.name, latency_ms={})
            await broadcast({"type": "status", "state": "idle",
                             "detail": "scratched"})

    # ------------------------------------------------------------ utterances
    async def process_utterance(text: str | None = None, audio=None):
        """Shared pipeline for typed utterances (ws text box) and mic
        utterances (VAD chunker). Engine inference runs off the event loop."""
        await broadcast({"type": "status", "state": "thinking"})
        if audio is not None:
            res = await asyncio.to_thread(engine.process, audio,
                                          doc.render_context())
        else:
            res = await asyncio.to_thread(engine.process_text, text,
                                          doc.render_context())
        if res.transcript:
            await broadcast({"type": "transcript", "text": res.transcript,
                             "interim": False})
        a = res.action
        if isinstance(a, (TextReply, Clarify)):
            await broadcast({"type": "reply", **a.model_dump()})
        else:
            # auto-commit any prior pending (continuous dictation)
            for pid in list(pending):
                await resolve_pending(pid, "commit")
            counter["n"] += 1
            pid = f"p{counter['n']}"
            audio_wav = None
            if audio is not None:
                from .audio.vad import wav_bytes
                audio_wav = wav_bytes(audio)
            pending[pid] = {"action": a, "transcript": res.transcript,
                            "audio_wav": audio_wav}
            await broadcast({"type": "proposal", "pending_id": pid,
                             "action": a.model_dump(),
                             "transcript": res.transcript})
        await broadcast({"type": "status",
                         "state": "listening" if mic else "idle"})

    # ------------------------------------------------------------ dispatch
    async def dispatch(ws: WebSocket, msg: dict):
        """Handle one client frame (see PROTOCOL.md)."""
        t = msg.get("type")

        if t == "utterance":
            await process_utterance(text=msg["text"])

        elif t == "resolve":
            await resolve_pending(msg["pending_id"], msg["verdict"])

        elif t == "intent":
            # App-layer intents (UI buttons now; voice spotters in
            # Phase 5 hit this exact path). Never engine Actions.
            name = msg["name"]
            if name == "undo":
                if doc.undo():
                    await broadcast({"type": "applied",
                                     "action": {"action": "undo"},
                                     "doc_context": doc.render_context()})
                else:
                    await broadcast({"type": "status", "state": "idle",
                                     "detail": "nothing to undo"})
            elif name in ("commit", "scratch") and pending:
                # Resolve the oldest outstanding proposal.
                pid = next(iter(pending))
                await resolve_pending(pid, "commit" if name == "commit" else "scratch")

        elif t == "edit":
            a = ACTION_ADAPTER.validate_python(msg["action"])
            await apply_and_broadcast(a, "committed", source="keyboard")
            # TODO(Phase 6): edited_after gold-label linkage.

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        clients.add(ws)
        await send(ws, {"type": "status",
                        "state": "listening" if mic else "idle",
                        "detail": f"engine={engine.name}"})
        try:
            while True:
                raw = await ws.receive_text()
                # Contain per-frame failures (bad JSON, missing keys, invalid
                # Action payloads): report an error frame, keep the socket up.
                try:
                    await dispatch(ws, json.loads(raw))
                except WebSocketDisconnect:
                    raise
                except Exception as e:
                    await send(ws, {"type": "error",
                                    "message": f"{type(e).__name__}: {e}"})
        except WebSocketDisconnect:
            pass
        finally:
            clients.discard(ws)

    # ------------------------------------------------------------ mic (Phase 4)
    if mic:
        @app.on_event("startup")
        async def start_mic():
            loop = asyncio.get_running_loop()
            utterances: asyncio.Queue = asyncio.Queue()

            def capture():  # daemon thread: mic -> VAD -> chunker -> queue
                from .audio.vad import SileroVAD, UtteranceChunker, mic_frames
                try:
                    chunker = UtteranceChunker(is_speech=SileroVAD())
                    for frame in mic_frames():
                        u = chunker.feed(frame)
                        if u is not None:
                            loop.call_soon_threadsafe(utterances.put_nowait, u)
                except Exception as e:  # no input device, driver error, ...
                    msg = f"mic capture failed: {type(e).__name__}: {e}"
                    print(msg, flush=True)
                    asyncio.run_coroutine_threadsafe(
                        broadcast({"type": "status", "state": "idle",
                                   "detail": msg}), loop)

            async def consume():
                while True:
                    u = await utterances.get()
                    await broadcast({"type": "status", "state": "heard",
                                     "detail": f"utterance {len(u)/16000:.1f}s"})
                    await process_utterance(audio=u)

            threading.Thread(target=capture, daemon=True, name="mic").start()
            asyncio.create_task(consume())

    app.mount("/", StaticFiles(directory=Path(__file__).resolve()
              .parents[2] / "frontend", html=True), name="frontend")
    return app
