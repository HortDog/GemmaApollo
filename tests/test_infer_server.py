"""Inference server HTTP API (infer_server.py) — mock engine, zero models.
The wire contract: responses are EngineResult JSON verbatim (frozen schema)."""
import base64

import numpy as np
from starlette.testclient import TestClient

from scribe.audio.vad import wav_bytes
from scribe.engine.base import EngineResult, TextReply
from scribe.infer_server import build_infer_app


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def test_healthz_reports_engine_and_load_state():
    client = TestClient(build_infer_app("mock"))
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "engine": "mock", "loaded": False}


def test_readyz_immediate_without_preload():
    client = TestClient(build_infer_app("mock"))
    r = client.get("/readyz")
    assert r.status_code == 200 and r.json() == {"ready": True}


def test_process_text_returns_engine_result_verbatim():
    client = TestClient(build_infer_app("mock"))
    r = client.post("/v1/process_text",
                    json={"transcript": "e equals m c squared"})
    assert r.status_code == 200
    res = EngineResult.model_validate(r.json())
    assert res.engine == "mock"
    assert res.action.action == "append_math"
    assert res.action.latex == "E = mc^2"
    # a successful inference marks the engine as loaded
    assert client.get("/healthz").json()["loaded"] is True


def test_process_audio_not_implemented_maps_to_501():
    client = TestClient(build_infer_app("mock"))   # mock engine is text-only
    wav = wav_bytes(np.zeros(16000, dtype=np.float32))
    r = client.post("/v1/process",
                    json={"audio_wav_b64": b64(wav), "doc_context": ""})
    assert r.status_code == 501


def test_bad_audio_payloads_are_400_and_422():
    client = TestClient(build_infer_app("mock"))
    r = client.post("/v1/process", json={"audio_wav_b64": "!!! not base64 !!!"})
    assert r.status_code == 400
    r = client.post("/v1/process", json={"audio_wav_b64": b64(b"not a wav")})
    assert r.status_code == 400
    r = client.post("/v1/process", json={"doc_context": "no audio field"})
    assert r.status_code == 422
    r = client.post("/v1/process_text", json={})
    assert r.status_code == 422


def test_engine_exception_maps_to_500(monkeypatch):
    class BoomEngine:
        name = "boom"

        def process(self, audio, doc_context):
            raise RuntimeError("kaput")

        def process_text(self, transcript, doc_context):
            raise RuntimeError("kaput")

    import scribe.infer_server as mod
    monkeypatch.setattr(mod, "make_engine", lambda name, **kw: BoomEngine())
    client = TestClient(mod.build_infer_app("mock"))
    r = client.post("/v1/process_text", json={"transcript": "x"})
    assert r.status_code == 500 and "kaput" in r.json()["detail"]


def test_wav_survives_the_wire_within_int16_precision(monkeypatch):
    received = {}

    class EchoEngine:
        name = "echo"

        def process(self, audio, doc_context):
            received["audio"] = audio
            received["doc_context"] = doc_context
            return EngineResult(action=TextReply(text="ok"), engine=self.name)

        def process_text(self, transcript, doc_context):
            raise NotImplementedError

    import scribe.infer_server as mod
    monkeypatch.setattr(mod, "make_engine", lambda name, **kw: EchoEngine())
    client = TestClient(mod.build_infer_app("echo"))

    pcm = np.sin(np.linspace(0, 100, 16000)).astype(np.float32) * 0.5
    r = client.post("/v1/process", json={"audio_wav_b64": b64(wav_bytes(pcm)),
                                         "doc_context": "[e1] x"})
    assert r.status_code == 200
    assert received["doc_context"] == "[e1] x"
    assert received["audio"].dtype == np.float32
    assert np.allclose(received["audio"], pcm, atol=1.0 / 32767)
