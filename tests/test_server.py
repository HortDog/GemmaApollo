"""Phase 2 acceptance: the full ws loop runs with MockEngine, zero models.
Exercises PROTOCOL.md frames via starlette's TestClient websocket."""
import json
from starlette.testclient import TestClient
from scribe.server import build_app


def _drain_until(ws, type_, **fields):
    """Read frames until one of the given type whose fields all match; return it.
    (The server emits a trailing `status idle` after each proposal, so match on
    specific fields when you need a particular status.)"""
    while True:
        m = ws.receive_json()
        if m["type"] == type_ and all(m.get(k) == v for k, v in fields.items()):
            return m


def test_dictation_proposal_then_commit():
    client = TestClient(build_app("mock"))
    with client.websocket_connect("/ws") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "status" and hello["state"] == "idle"

        ws.send_json({"type": "utterance", "text": "e equals m c squared"})
        prop = _drain_until(ws, "proposal")
        assert prop["action"]["action"] == "append_math"
        assert prop["action"]["latex"] == "E = mc^2"
        pid = prop["pending_id"]

        ws.send_json({"type": "resolve", "pending_id": pid, "verdict": "commit"})
        applied = _drain_until(ws, "applied")
        assert applied["assigned_id"] == "e1"
        assert applied["doc_context"] == "[e1] E = mc^2"


def test_scratch_intent_discards_pending():
    client = TestClient(build_app("mock"))
    with client.websocket_connect("/ws") as ws:
        _drain_until(ws, "status")
        ws.send_json({"type": "utterance", "text": "x squared"})
        _drain_until(ws, "proposal")
        ws.send_json({"type": "intent", "name": "scratch"})
        _drain_until(ws, "status", detail="scratched")
        # nothing committed: a follow-up append should still get id e1
        ws.send_json({"type": "utterance", "text": "y"})
        prop = _drain_until(ws, "proposal")
        ws.send_json({"type": "resolve",
                      "pending_id": prop["pending_id"], "verdict": "commit"})
        applied = _drain_until(ws, "applied")
        assert applied["assigned_id"] == "e1"


def test_ordinal_command_round_trips():
    client = TestClient(build_app("mock"))
    with client.websocket_connect("/ws") as ws:
        _drain_until(ws, "status")
        for text in ("a", "b"):
            ws.send_json({"type": "utterance", "text": text})
            prop = _drain_until(ws, "proposal")
            ws.send_json({"type": "resolve",
                          "pending_id": prop["pending_id"], "verdict": "commit"})
            _drain_until(ws, "applied")
        # "change the last line to z" must resolve the ordinal to e2
        ws.send_json({"type": "utterance", "text": "change the last line to z"})
        prop = _drain_until(ws, "proposal")
        assert prop["action"]["action"] == "replace"
        assert prop["action"]["target_id"] == "e2"
        ws.send_json({"type": "resolve",
                      "pending_id": prop["pending_id"], "verdict": "commit"})
        applied = _drain_until(ws, "applied")
        assert applied["doc_context"] == "[e1] a  [e2] z"


def test_unknown_target_errors_without_mutation():
    client = TestClient(build_app("mock"))
    with client.websocket_connect("/ws") as ws:
        _drain_until(ws, "status")
        # explicit bad id -> proposal, then commit -> error frame, no mutation
        ws.send_json({"type": "utterance", "text": "delete e9"})
        prop = _drain_until(ws, "proposal")
        ws.send_json({"type": "resolve",
                      "pending_id": prop["pending_id"], "verdict": "commit"})
        err = _drain_until(ws, "error")
        assert "unknown target id" in err["message"]


def test_out_of_range_ordinal_clarifies_and_socket_survives():
    client = TestClient(build_app("mock"))
    with client.websocket_connect("/ws") as ws:
        _drain_until(ws, "status")
        # 0 lines: "delete line five" cannot resolve -> Clarify, not a crash
        ws.send_json({"type": "utterance", "text": "delete line five"})
        reply = _drain_until(ws, "reply")
        assert "Which line" in reply["question"]
        # socket still alive: a normal utterance round-trips
        ws.send_json({"type": "utterance", "text": "x"})
        prop = _drain_until(ws, "proposal")
        assert prop["action"]["action"] == "append_math"


def test_malformed_frames_error_and_socket_survives():
    client = TestClient(build_app("mock"))
    with client.websocket_connect("/ws") as ws:
        _drain_until(ws, "status")
        ws.send_text("this is not json")                      # bad JSON
        _drain_until(ws, "error")
        ws.send_json({"type": "edit", "action": {"action": "explode"}})  # bad Action
        _drain_until(ws, "error")
        ws.send_json({"type": "resolve"})                      # missing keys
        _drain_until(ws, "error")
        ws.send_json({"type": "utterance", "text": "x"})       # still alive
        _drain_until(ws, "proposal")


def test_training_record_logs_preaction_context(tmp_path, monkeypatch):
    # SessionLogger writes under cwd/data/sessions — sandbox it in tmp_path.
    monkeypatch.chdir(tmp_path)
    client = TestClient(build_app("mock"))
    with client.websocket_connect("/ws") as ws:
        _drain_until(ws, "status")
        ws.send_json({"type": "utterance", "text": "a"})
        prop = _drain_until(ws, "proposal")
        ws.send_json({"type": "resolve",
                      "pending_id": prop["pending_id"], "verdict": "commit"})
        _drain_until(ws, "applied")
        ws.send_json({"type": "utterance", "text": "b"})
        prop = _drain_until(ws, "proposal")
        ws.send_json({"type": "resolve",
                      "pending_id": prop["pending_id"], "verdict": "commit"})
        _drain_until(ws, "applied")
    session = next((tmp_path / "data" / "sessions").iterdir())
    recs = sorted(session.glob("*.json"))
    assert len(recs) == 2
    first, second = (json.loads(p.read_text()) for p in recs)
    # (doc context -> action) triples: context must be the PRE-action state
    assert first["doc_context"] == ""
    assert second["doc_context"] == "[e1] a"
    assert second["engine_action"]["latex"] == "b"


def test_mic_frames_error_when_mic_off():
    client = TestClient(build_app("mock"))          # no mic mode
    with client.websocket_connect("/ws") as ws:
        _drain_until(ws, "status")
        ws.send_json({"type": "mic", "action": "list"})
        err = _drain_until(ws, "error")
        assert "mic mode is off" in err["message"]


def test_mic_frames_with_stubbed_supervisor():
    app = build_app("mock")                          # stub micctl like start_mic would
    calls = {"selected": None, "testing": None}

    async def mics_frame():
        return {"type": "mics", "current": 1,
                "devices": [{"index": 1, "name": "Fake Mic", "default": True}]}

    async def select(device):
        calls["selected"] = device

    app.state.micctl["list"] = mics_frame
    app.state.micctl["select"] = select
    app.state.micctl["set_testing"] = lambda on: calls.update(testing=on)

    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        _drain_until(ws, "status")
        m = _drain_until(ws, "mics")                 # sent on connect
        assert m["devices"][0]["name"] == "Fake Mic" and m["current"] == 1

        ws.send_json({"type": "mic", "action": "select", "device": 3})
        ws.send_json({"type": "mic", "action": "test_start"})
        _drain_until(ws, "status", detail="mic test on")
        assert calls == {"selected": 3, "testing": True}
        ws.send_json({"type": "mic", "action": "test_stop"})
        _drain_until(ws, "status", detail="mic test off")
        assert calls["testing"] is False

        ws.send_json({"type": "mic", "action": "list"})
        _drain_until(ws, "mics")


def test_undo_after_commit():
    client = TestClient(build_app("mock"))
    with client.websocket_connect("/ws") as ws:
        _drain_until(ws, "status")
        ws.send_json({"type": "utterance", "text": "q"})
        prop = _drain_until(ws, "proposal")
        ws.send_json({"type": "resolve",
                      "pending_id": prop["pending_id"], "verdict": "commit"})
        applied = _drain_until(ws, "applied")
        assert applied["doc_context"] == "[e1] q"
        ws.send_json({"type": "intent", "name": "undo"})
        undone = _drain_until(ws, "applied")
        assert undone["doc_context"] == ""
