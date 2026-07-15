"""RemoteEngine — HTTP client proxy implementing the frozen Engine protocol
(engine/base.py) against a `scribe infer-serve` process (see INFERENCE.md).

Module import is torch-free by design: the app host running --engine remote
needs only the base dependencies. Audio crosses the wire as base64 16-bit WAV
(reusing audio/vad.wav_bytes); responses are EngineResult JSON verbatim, so
the frozen Action schema is the wire contract.
"""
from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from pathlib import Path

from ..audio.vad import wav_bytes
from .base import EngineResult


class _UrllibTransport:
    """Default transport. Tests inject a starlette-TestClient-backed stand-in
    with the same two methods instead of opening sockets."""

    def __init__(self, base_url: str, timeout: float):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _hint(self) -> str:
        return (f"is the inference server running? "
                f"(scribe infer-serve, expected at {self.base_url})")

    def get(self, path: str) -> tuple[int, str]:
        try:
            with urllib.request.urlopen(self.base_url + path,
                                        timeout=self.timeout) as r:
                return r.status, r.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8", errors="replace")
        except OSError as e:
            raise RuntimeError(f"inference server unreachable: {e} — "
                               + self._hint()) from e

    def post(self, path: str, payload: dict) -> tuple[int, str]:
        req = urllib.request.Request(
            self.base_url + path, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.status, r.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8", errors="replace")
        except OSError as e:
            raise RuntimeError(f"inference server unreachable: {e} — "
                               + self._hint()) from e


class RemoteEngine:
    # Adopts the upstream engine's name on first contact so the datalogger
    # records the engine that actually produced each Action.
    name = "remote"

    def __init__(self, url: str = "http://127.0.0.1:8018",
                 timeout: float = 120.0, transport=None):
        # timeout covers a cold server loading models on the first request;
        # start infer-serve with --preload to avoid paying it there.
        self._transport = transport or _UrllibTransport(url, timeout)
        self._hello_done = False

    def _hello(self):
        """One-time /healthz probe: fail fast with a clear message and adopt
        the upstream engine name."""
        if self._hello_done:
            return
        status, body = self._transport.get("/healthz")
        if status == 200:
            try:
                self.name = json.loads(body).get("engine", self.name)
            except json.JSONDecodeError:
                pass
        self._hello_done = True

    def _call(self, path: str, payload: dict) -> EngineResult:
        self._hello()
        status, body = self._transport.post(path, payload)
        if status != 200:
            raise RuntimeError(f"inference server {status} on {path}: {body}")
        return EngineResult.model_validate_json(body)

    # ------------------------------------------------------------------ engine
    def process(self, audio, doc_context: str) -> EngineResult:
        if hasattr(audio, "dtype"):
            data = wav_bytes(audio)
        else:
            data = Path(audio).read_bytes()   # bench fixtures are wav paths
        return self._call("/v1/process", {
            "audio_wav_b64": base64.b64encode(data).decode("ascii"),
            "doc_context": doc_context,
        })

    def process_text(self, transcript: str, doc_context: str) -> EngineResult:
        return self._call("/v1/process_text", {
            "transcript": transcript, "doc_context": doc_context,
        })
