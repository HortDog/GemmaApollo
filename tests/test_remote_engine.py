"""RemoteEngine serialization + failure taxonomy via httpx.MockTransport —
no sockets, no torch. Includes a full RemoteEngine -> model-app round trip
with the transport bridged into a TestClient."""
import json

import httpx
import numpy as np
import pytest
from starlette.testclient import TestClient

from scribe.engine.base import Clarify, EngineResult, Replace
from scribe.engine.remote_engine import EngineUnavailable, RemoteEngine


def canned(action, engine="s2l", transcript=None):
    return EngineResult(action=action, transcript=transcript, engine=engine,
                        latency_ms={"total": 5.0}).model_dump_json()


def make_engine(handler):
    return RemoteEngine("http://model", transport=httpx.MockTransport(handler))


def test_process_serializes_wav_and_validates_result():
    seen = {}

    def handler(req):
        if req.url.path == "/health":
            return httpx.Response(200, json={"engine": "s2l", "loaded": True})
        seen.update(json.loads(req.content))
        return httpx.Response(200, content=canned(
            Replace(target_id="e2", latex="x^2"), transcript="change e2"))

    eng = make_engine(handler)
    res = eng.process(np.zeros(8000, dtype=np.float32), "[e1] a  [e2] b")
    assert isinstance(res.action, Replace) and res.action.target_id == "e2"
    assert seen["doc_context"] == "[e1] a  [e2] b"
    import base64
    wav = base64.b64decode(seen["audio_wav"])
    assert wav[:4] == b"RIFF" and len(wav) > 16000   # 8000 samples * 2 bytes


def test_action_union_variants_round_trip():
    actions = [Replace(target_id="e1", latex="y"),
               Clarify(question="Which line?", candidates=["e1", "e2"])]

    def handler(req):
        if req.url.path == "/health":
            return httpx.Response(200, json={"engine": "s2l", "loaded": True})
        return httpx.Response(200, content=canned(actions[0]))

    eng = make_engine(handler)
    for a in actions:
        actions[0] = a
        res = eng.process_text("t", "")
        assert type(res.action) is type(a) and res.action == a


def test_name_cached_from_health_then_responses():
    calls = {"health": 0}

    def handler(req):
        if req.url.path == "/health":
            calls["health"] += 1
            return httpx.Response(200, json={"engine": "s2l", "loaded": False})
        return httpx.Response(200, content=canned(Replace(target_id="e1", latex="z"),
                                                  engine="s2l"))

    eng = make_engine(handler)
    assert eng.name == "s2l" and calls["health"] == 1
    assert eng.name == "s2l" and calls["health"] == 1   # no I/O on re-read


def test_unreachable_server_raises_engine_unavailable():
    def handler(req):
        raise httpx.ConnectError("refused")

    eng = make_engine(handler)
    assert eng.name == "remote"                         # health probe failed
    with pytest.raises(EngineUnavailable, match="unreachable"):
        eng.process_text("x", "")


def test_http_500_raises_engine_unavailable_with_detail():
    def handler(req):
        if req.url.path == "/health":
            return httpx.Response(200, json={"engine": "s2l", "loaded": True})
        return httpx.Response(500, json={"detail": "RuntimeError: cuda exploded"})

    with pytest.raises(EngineUnavailable, match="cuda exploded"):
        make_engine(handler).process_text("x", "")


def test_remote_to_model_app_round_trip(tmp_path):
    """Full path: RemoteEngine -> (MockTransport bridge) -> model app ->
    FakeEngine, all in-process."""
    from test_model_server import FakeEngine
    from scribe.model_server import build_model_app

    tc = TestClient(build_model_app(FakeEngine(), data_root=tmp_path))

    def handler(req):
        r = tc.request(req.method, req.url.path, content=req.content,
                       headers={"content-type": "application/json"})
        return httpx.Response(r.status_code, content=r.content)

    eng = make_engine(handler)
    res = eng.process(np.full(16000, 0.1, dtype=np.float32), "[e1] x")
    assert res.action.latex == "x^2"
    assert eng.name == "fake"          # refreshed from the response's engine
