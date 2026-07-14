"""FastAPI app: serves frontend/ and the /ws endpoint per PROTOCOL.md.
Server owns DocState (single source of truth) and the SessionLogger.
Phase 2 target: full loop with MockEngine, zero models."""
from __future__ import annotations
import json
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import TypeAdapter

from .docstate import DocState
from .datalogger import SessionLogger
from .engine.base import Action, TextReply, Clarify
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

def build_app(engine_name: str = "mock") -> FastAPI:
    app = FastAPI(title="GemmaApollo Scribe")
    engine = make_engine(engine_name)
    doc = DocState()
    logger = SessionLogger()
    pending: dict = {}  # pending_id -> {"action": Action, "transcript": str|None}
    counter = {"n": 0}

    async def send(ws: WebSocket, obj: dict):
        await ws.send_text(json.dumps(obj))

    async def apply_and_broadcast(ws: WebSocket, action: Action, verdict: str,
                                  transcript=None, source="voice"):
        try:
            assigned = doc.apply(action)
        except KeyError as e:
            await send(ws, {"type": "error", "message": str(e),
                            "action": action.model_dump()})
            logger.log(audio_bytes=None, doc_context=doc.render_context(),
                       transcript=transcript, engine_action=action.model_dump(),
                       final_action=None, verdict="discarded",
                       engine=engine.name, latency_ms={})
            return
        logger.log(audio_bytes=None, doc_context=doc.render_context(),
                   transcript=transcript, engine_action=action.model_dump(),
                   final_action=action.model_dump(), verdict=verdict,
                   engine=engine.name, latency_ms={})
        await send(ws, {"type": "applied", "action": action.model_dump(),
                        "assigned_id": assigned,
                        "doc_context": doc.render_context()})

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        await send(ws, {"type": "status", "state": "idle",
                        "detail": f"engine={engine.name}"})
        try:
            while True:
                msg = json.loads(await ws.receive_text())
                t = msg.get("type")

                if t == "utterance":
                    await send(ws, {"type": "status", "state": "thinking"})
                    res = engine.process_text(msg["text"], doc.render_context())
                    if res.transcript:
                        await send(ws, {"type": "transcript",
                                        "text": res.transcript, "interim": False})
                    a = res.action
                    if isinstance(a, (TextReply, Clarify)):
                        await send(ws, {"type": "reply", **a.model_dump()})
                    else:
                        # auto-commit any prior pending (continuous dictation)
                        for pid, p in list(pending.items()):
                            await apply_and_broadcast(ws, p["action"], "committed",
                                                      p["transcript"])
                            del pending[pid]
                        counter["n"] += 1
                        pid = f"p{counter['n']}"
                        pending[pid] = {"action": a, "transcript": res.transcript}
                        await send(ws, {"type": "proposal", "pending_id": pid,
                                        "action": a.model_dump(),
                                        "transcript": res.transcript})
                    await send(ws, {"type": "status", "state": "idle"})

                elif t == "resolve":
                    p = pending.pop(msg["pending_id"], None)
                    if not p:
                        continue
                    if msg["verdict"] == "commit":
                        await apply_and_broadcast(ws, p["action"], "committed",
                                                  p["transcript"])
                    else:
                        logger.log(audio_bytes=None,
                                   doc_context=doc.render_context(),
                                   transcript=p["transcript"],
                                   engine_action=p["action"].model_dump(),
                                   final_action=None, verdict="scratched",
                                   engine=engine.name, latency_ms={})

                elif t == "intent":
                    name = msg["name"]
                    if name == "undo":
                        if doc.undo():
                            await send(ws, {"type": "applied",
                                            "action": {"action": "undo"},
                                            "doc_context": doc.render_context()})
                    elif name in ("commit", "scratch") and pending:
                        pid = next(iter(pending))
                        await ws_send_resolve(ws, pid, name)
                    # TODO(Phase 5): voice spotters call the same code path.

                elif t == "edit":
                    a = ACTION_ADAPTER.validate_python(msg["action"])
                    await apply_and_broadcast(ws, a, "committed", source="keyboard")
                    # TODO(Phase 6): edited_after gold-label linkage.

        except WebSocketDisconnect:
            pass

        async def ws_send_resolve(ws, pid, name):  # helper referenced above
            verdict = "commit" if name == "commit" else "scratch"
            p = pending.pop(pid)
            if verdict == "commit":
                await apply_and_broadcast(ws, p["action"], "committed", p["transcript"])

    app.mount("/", StaticFiles(directory=Path(__file__).resolve()
              .parents[2] / "frontend", html=True), name="frontend")
    return app
