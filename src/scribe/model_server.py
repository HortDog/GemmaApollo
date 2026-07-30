"""Model server: the Engine protocol over HTTP, plus the central home of
the GemmaApollo dataset (POST /log, fed by each app server's SpoolingLogger).

Runs on the GPU box (bare via `scribe model-server`, or in Docker); app
servers reach it through RemoteEngine over the tailnet. The frozen
EngineResult/Action pydantic models ARE the wire schema — both hops
revalidate through them, nothing new to invent. Audio arrives as complete
16 kHz mono 16-bit wav bytes (the same container the datalogger writes),
base64 in JSON, validated hard on decode.
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Optional, Union

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import Base64Bytes, BaseModel

from .engine.base import EngineResult


class ProcessRequest(BaseModel):
    audio_wav: Base64Bytes            # complete 16 kHz mono 16-bit wav file
    doc_context: str = ""


class ProcessTextRequest(BaseModel):
    transcript: str
    doc_context: str = ""


class Health(BaseModel):
    engine: str
    loaded: bool                       # flips after the first successful run
    device: Optional[str] = None


class LogRecord(BaseModel):
    client_id: str
    session_id: str
    seq: int
    audio_wav: Optional[Base64Bytes] = None   # wake-word rows have no audio
    payload: dict


def _safe(component: str) -> str:
    """Client-supplied ids become path components — flatten anything that
    isn't a plain name (tailnet peers are trusted, but not their typos)."""
    s = re.sub(r"[^A-Za-z0-9._-]", "_", component).strip("._")
    if not s:
        raise HTTPException(400, f"bad path component: {component!r}")
    return s


def build_model_app(engine: Union[str, object] = "s2l",
                    data_root: Path | str = Path("data/sessions")) -> FastAPI:
    """`engine`: a name routed through make_engine, or an instance (tests)."""
    from .engine import make_engine

    app = FastAPI(title="GemmaApollo Model Server")
    eng = make_engine(engine) if isinstance(engine, str) else engine
    gpu_lock = threading.Lock()       # one inference at a time, defensively
    log_lock = threading.Lock()
    state = {"loaded": False}
    root = Path(data_root)

    def run(fn, *args) -> EngineResult:
        with gpu_lock:
            try:
                res = fn(*args)
            except Exception as e:
                raise HTTPException(500, f"{type(e).__name__}: {e}")
        state["loaded"] = True
        return res

    # sync `def` endpoints run in FastAPI's threadpool — inference and disk
    # writes never block the event loop.
    @app.post("/process", response_model=EngineResult)
    def process(req: ProcessRequest) -> EngineResult:
        from .audio.vad import wav_to_float32
        try:
            pcm = wav_to_float32(bytes(req.audio_wav))
        except Exception as e:
            raise HTTPException(400, f"bad wav: {e}")
        return run(eng.process, pcm, req.doc_context)

    @app.post("/process_text", response_model=EngineResult)
    def process_text(req: ProcessTextRequest) -> EngineResult:
        return run(eng.process_text, req.transcript, req.doc_context)

    @app.get("/health", response_model=Health)
    def health() -> Health:
        # cheap liveness probe — must NOT trigger the lazy model load
        return Health(engine=eng.name, loaded=state["loaded"],
                      device=getattr(eng, "device", None))

    @app.post("/warmup", response_model=EngineResult)
    def warmup() -> EngineResult:
        # 1 s of silence through the full path — exercises S2L's lazy load
        # (torch-before-CTranslate2 ordering stays inside S2LEngine)
        return run(eng.process, np.zeros(16000, dtype=np.float32), "")

    app.state.warmup = warmup          # `scribe model-server --preload`

    @app.post("/log")
    def log(rec: LogRecord) -> dict:
        """Central GemmaApollo dataset:
        data/sessions/<client_id>/<session_id>/NNNN.{wav,json}.
        Idempotent on (client, session, seq) so spool retries are safe."""
        sdir = root / _safe(rec.client_id) / _safe(rec.session_id)
        stem = sdir / f"{rec.seq:04d}"
        with log_lock:
            if stem.with_suffix(".json").exists():
                return {"stored": False, "dup": True}
            sdir.mkdir(parents=True, exist_ok=True)
            if rec.audio_wav:
                stem.with_suffix(".wav").write_bytes(bytes(rec.audio_wav))
            stem.with_suffix(".json").write_text(json.dumps(rec.payload,
                                                            indent=1))
        return {"stored": True, "dup": False}

    return app
