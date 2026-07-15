"""RemoteEngine proxy against an in-process inference server (mock engine,
zero models, no sockets): the frozen Engine protocol must behave identically
across the HTTP boundary, including through the full ws app loop."""
import numpy as np
from starlette.testclient import TestClient

from scribe.engine.base import EngineResult
from scribe.engine.mock_engine import MockEngine
from scribe.engine.remote_engine import RemoteEngine
from scribe.infer_server import build_infer_app
from scribe.server import build_app


class StubTransport:
    """Backs RemoteEngine's transport with a starlette TestClient."""

    def __init__(self, app):
        self.client = TestClient(app)

    def get(self, path):
        r = self.client.get(path)
        return r.status_code, r.text

    def post(self, path, payload):
        r = self.client.post(path, json=payload)
        return r.status_code, r.text


def remote_mock() -> RemoteEngine:
    return RemoteEngine(transport=StubTransport(build_infer_app("mock")))


def test_process_text_matches_direct_engine_call():
    remote = remote_mock()
    direct = MockEngine().process_text("one half m v squared", "")
    via_http = remote.process_text("one half m v squared", "")
    assert isinstance(via_http, EngineResult)
    assert via_http.action == direct.action
    assert via_http.transcript == direct.transcript
    assert via_http.engine == "mock"


def test_name_adopts_upstream_engine_for_datalogger():
    remote = remote_mock()
    assert remote.name == "remote"          # before first contact
    remote.process_text("x", "")
    assert remote.name == "mock"            # adopted from /healthz


def test_server_error_raises_runtime_error():
    remote = remote_mock()
    # mock engine's audio path is NotImplemented -> 501 -> RuntimeError
    try:
        remote.process(np.zeros(16000, dtype=np.float32), "")
    except RuntimeError as e:
        assert "501" in str(e)
    else:
        raise AssertionError("expected RuntimeError on non-200")


def _drain_until(ws, type_, **fields):
    while True:
        m = ws.receive_json()
        if m["type"] == type_ and all(m.get(k) == v for k, v in fields.items()):
            return m


def test_full_ws_loop_over_remote_engine(tmp_path, monkeypatch):
    """App loop (proposal -> commit -> applied) is unchanged when the engine
    lives behind HTTP; the datalogger records the upstream engine name."""
    import json

    monkeypatch.chdir(tmp_path)
    transport = StubTransport(build_infer_app("mock"))
    app = build_app("remote", engine_kwargs={"transport": transport})
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        _drain_until(ws, "status")
        ws.send_json({"type": "utterance", "text": "e equals m c squared"})
        prop = _drain_until(ws, "proposal")
        assert prop["action"]["latex"] == "E = mc^2"
        ws.send_json({"type": "resolve", "pending_id": prop["pending_id"],
                      "verdict": "commit"})
        applied = _drain_until(ws, "applied")
        assert applied["doc_context"] == "[e1] E = mc^2"

    session = next((tmp_path / "data" / "sessions").iterdir())
    rec = json.loads(next(iter(sorted(session.glob("*.json")))).read_text())
    assert rec["engine"] == "mock"          # adopted upstream name, not "remote"


def test_unreachable_server_raises_helpful_error():
    remote = RemoteEngine(url="http://127.0.0.1:1", timeout=0.2)
    try:
        remote.process_text("x", "")
    except RuntimeError as e:
        assert "infer-serve" in str(e)
    else:
        raise AssertionError("expected RuntimeError when server is down")
