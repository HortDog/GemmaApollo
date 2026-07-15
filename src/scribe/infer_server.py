"""Standalone inference server: hosts one heavy Engine (s2l now, gemma later)
behind a small HTTP API so the scribe app can run on a different process,
machine, or OS (see INFERENCE.md). Run with `scribe infer-serve`.

No static files, no websocket, no DocState, no datalogger — those all stay in
the app process (server.py). Responses are EngineResult JSON verbatim: the
frozen schema (engine/base.py) is the wire contract.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import threading

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .engine import make_engine


class ProcessTextRequest(BaseModel):
    transcript: str
    doc_context: str = ""


class ProcessRequest(BaseModel):
    audio_wav_b64: str            # base64 16-bit PCM mono 16 kHz wav file
    doc_context: str = ""


def build_infer_app(engine_name: str = "s2l", preload: bool = False,
                    engine_kwargs: dict | None = None) -> FastAPI:
    app = FastAPI(title="GemmaApollo Scribe — inference server")
    engine = make_engine(engine_name, **(engine_kwargs or {}))
    # Engines are single-GPU and not thread-safe: serialize inference.
    infer_lock = asyncio.Lock()
    state = {"loaded": False, "loading": False}

    def warm():
        getattr(engine, "_lazy_load", lambda: None)()
        state["loaded"] = True
        state["loading"] = False

    if preload:
        state["loading"] = True

        @app.on_event("startup")
        async def start_preload():
            threading.Thread(target=warm, daemon=True, name="preload").start()

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "engine": engine.name, "loaded": state["loaded"]}

    @app.get("/readyz")
    async def readyz():
        if state["loading"]:
            return JSONResponse(status_code=503,
                                content={"ready": False, "detail": "loading"})
        return {"ready": True}

    async def run_engine(fn, *args) -> JSONResponse:
        async with infer_lock:
            try:
                res = await asyncio.to_thread(fn, *args)
            except NotImplementedError as e:
                return JSONResponse(status_code=501,
                                    content={"detail": f"{engine.name}: {e}"})
            except Exception as e:
                return JSONResponse(status_code=500,
                                    content={"detail": f"{type(e).__name__}: {e}"})
        state["loaded"] = True
        return JSONResponse(content=res.model_dump())

    @app.post("/v1/process_text")
    async def process_text(req: ProcessTextRequest):
        return await run_engine(engine.process_text, req.transcript,
                                req.doc_context)

    @app.post("/v1/process")
    async def process(req: ProcessRequest):
        from .audio.vad import wav_to_float32
        try:
            audio = wav_to_float32(base64.b64decode(req.audio_wav_b64,
                                                    validate=True))
        except (binascii.Error, ValueError, EOFError) as e:
            return JSONResponse(status_code=400,
                                content={"detail": f"bad audio_wav_b64: {e}"})
        return await run_engine(engine.process, audio, req.doc_context)

    return app
