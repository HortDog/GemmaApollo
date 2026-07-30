"""RemoteEngine: the Engine protocol over HTTP to a model server.

Sync httpx by design — server.py already wraps every engine call in
asyncio.to_thread, and a persistent Client gives keep-alive pooling on the
per-utterance hop. Failures surface as EngineUnavailable; the app layer's
containment turns that into an `error` ws frame and drops the utterance
(the pending composition survives). Module top-level stays import-light,
matching the other engines.
"""
from __future__ import annotations

import base64
from pathlib import Path

from .base import EngineResult


class EngineUnavailable(RuntimeError):
    """Model server down/unreachable, or it reported an engine failure."""


class RemoteEngine:
    def __init__(self, base_url: str, timeout_s: float = 120.0,
                 transport=None):
        import httpx
        self.base_url = base_url.rstrip("/")
        # connect fast-fails when the server is down; read must survive a
        # cold Whisper+Qwen lazy load (~20-60 s) plus inference on top
        self._client = httpx.Client(base_url=self.base_url,
                                    timeout=httpx.Timeout(timeout_s,
                                                          connect=2.0),
                                    transport=transport)
        self._name: str | None = None
        try:  # one non-fatal probe so early datalogger rows carry the real name
            self._name = self._client.get("/health").json()["engine"]
        except Exception:
            pass

    @property
    def name(self) -> str:
        # Underlying engine name ("s2l"), so training rows are consistent
        # across local/remote deployments. NEVER does network I/O here (the
        # datalogger reads it on the event loop): resolved at init and
        # refreshed from every successful response's `engine` field.
        return self._name or "remote"

    def _post(self, path: str, payload: dict) -> EngineResult:
        import httpx
        try:
            r = self._client.post(path, json=payload)
            r.raise_for_status()
        except httpx.TransportError as e:
            raise EngineUnavailable(
                f"model server unreachable at {self.base_url}: {e}") from e
        except httpx.HTTPStatusError as e:
            try:
                detail = e.response.json().get("detail", "")
            except Exception:
                detail = e.response.text[:200]
            raise EngineUnavailable(
                f"model server error {e.response.status_code}: {detail}") from e
        res = EngineResult.model_validate_json(r.content)
        self._name = res.engine
        return res

    def process(self, audio, doc_context: str) -> EngineResult:
        from ..audio.vad import wav_bytes
        if isinstance(audio, (str, Path)):
            wav = Path(audio).read_bytes()
        else:
            wav = wav_bytes(audio)
        return self._post("/process",
                          {"audio_wav": base64.b64encode(wav).decode(),
                           "doc_context": doc_context})

    def process_text(self, transcript: str, doc_context: str) -> EngineResult:
        return self._post("/process_text", {"transcript": transcript,
                                            "doc_context": doc_context})
