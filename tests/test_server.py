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


def test_composition_merges_dictation_into_pending():
    client = TestClient(build_app("mock"))
    with client.websocket_connect("/ws") as ws:
        _drain_until(ws, "status")
        ws.send_json({"type": "utterance", "text": "one half m v squared"})
        p1 = _drain_until(ws, "proposal")
        assert p1["action"]["latex"] == "\\frac{1}{2}mv^2"
        assert p1["segments"] == 1

        ws.send_json({"type": "utterance", "text": "plus m g h"})
        p2 = _drain_until(ws, "proposal")
        assert p2["pending_id"] == p1["pending_id"]      # same proposal, extended
        assert p2["segments"] == 2
        assert p2["transcript"] == "one half m v squared plus m g h"
        # merged latex comes from re-converting the JOINED transcript
        assert p2["action"]["latex"] == "one half m v^2 + m g h"


def test_scratch_pops_last_segment_then_discards():
    client = TestClient(build_app("mock"))
    with client.websocket_connect("/ws") as ws:
        _drain_until(ws, "status")
        ws.send_json({"type": "utterance", "text": "one half m v squared"})
        p1 = _drain_until(ws, "proposal")
        ws.send_json({"type": "utterance", "text": "plus m g h"})
        _drain_until(ws, "proposal")

        ws.send_json({"type": "intent", "name": "scratch"})
        p3 = _drain_until(ws, "proposal")                 # reverted, not discarded
        assert p3["pending_id"] == p1["pending_id"]
        assert p3["segments"] == 1
        assert p3["action"]["latex"] == "\\frac{1}{2}mv^2"

        ws.send_json({"type": "intent", "name": "scratch"})
        _drain_until(ws, "status", detail="scratched")    # now fully discarded
        ws.send_json({"type": "utterance", "text": "y"})
        p4 = _drain_until(ws, "proposal")
        assert p4["segments"] == 1                        # fresh composition


def test_command_ends_composition_via_auto_commit():
    client = TestClient(build_app("mock"))
    with client.websocket_connect("/ws") as ws:
        _drain_until(ws, "status")
        ws.send_json({"type": "utterance", "text": "a"})
        _drain_until(ws, "proposal")
        ws.send_json({"type": "utterance", "text": "b"})
        _drain_until(ws, "proposal")
        ws.send_json({"type": "utterance", "text": "change the last line to z"})
        applied = _drain_until(ws, "applied")             # composition committed
        assert applied["doc_context"] == "[e1] a b"
        prop = _drain_until(ws, "proposal")               # then the command
        assert prop["action"]["action"] == "replace"
        assert prop["action"]["target_id"] == "e1"


def test_commit_logs_joined_transcript(tmp_path):
    import json as _json
    from pathlib import Path
    client = TestClient(build_app("mock"))
    with client.websocket_connect("/ws") as ws:
        _drain_until(ws, "status")
        ws.send_json({"type": "utterance", "text": "a"})
        _drain_until(ws, "proposal")
        ws.send_json({"type": "utterance", "text": "b"})
        p = _drain_until(ws, "proposal")
        ws.send_json({"type": "resolve",
                      "pending_id": p["pending_id"], "verdict": "commit"})
        applied = _drain_until(ws, "applied")
        assert applied["doc_context"] == "[e1] a b"
    session = next(Path("data/sessions").iterdir())      # conftest chdir'd to tmp
    recs = [_json.loads(f.read_text()) for f in sorted(session.glob("*.json"))]
    committed = [r for r in recs if r["verdict"] == "committed"]
    assert committed and committed[-1]["transcript"] == "a b"


def test_clarify_leaves_composition_intact():
    client = TestClient(build_app("mock"))
    with client.websocket_connect("/ws") as ws:
        _drain_until(ws, "status")
        ws.send_json({"type": "utterance", "text": "a"})
        p1 = _drain_until(ws, "proposal")
        # out-of-range ordinal -> Clarify (reply), composition untouched
        ws.send_json({"type": "utterance", "text": "delete line nine"})
        _drain_until(ws, "reply")
        ws.send_json({"type": "utterance", "text": "b"})
        p2 = _drain_until(ws, "proposal")
        assert p2["pending_id"] == p1["pending_id"] and p2["segments"] == 2


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


# ---------------------------------------------------------------- mute gate
def _fake_mic_stack(monkeypatch):
    """Run start_mic without hardware: silent fake frames, fake Silero, no
    device enumeration. wakeword_models={} disables spotters entirely."""
    import numpy as np
    import scribe.audio.vad as vadmod

    class FakeVAD:
        def __call__(self, frame):
            return 0.0

        def reset(self):
            pass

    def fake_frames(device, stop=None):
        import time
        while stop is None or not stop.is_set():
            time.sleep(0.005)
            yield np.zeros(512, dtype=np.float32)

    monkeypatch.setattr(vadmod, "SileroVAD", FakeVAD)
    monkeypatch.setattr(vadmod, "mic_frames", fake_frames)
    monkeypatch.setattr(vadmod, "list_input_devices", lambda: [])


def test_mic_mode_starts_muted_wake_and_mute_intents_toggle(monkeypatch):
    _fake_mic_stack(monkeypatch)
    client = TestClient(build_app("mock", mic=True, wakeword_models={}))
    with client.websocket_connect("/ws") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "status" and hello["state"] == "muted"

        # typed utterances bypass the mic gate; trailing status stays muted
        ws.send_json({"type": "utterance", "text": "x squared"})
        _drain_until(ws, "proposal")
        _drain_until(ws, "status", state="muted")

        ws.send_json({"type": "intent", "name": "wake"})
        _drain_until(ws, "status", state="listening", detail="unmuted")
        ws.send_json({"type": "intent", "name": "mute"})
        _drain_until(ws, "status", state="muted", detail="muted")


def test_start_unmuted_skips_wake_gate(monkeypatch):
    _fake_mic_stack(monkeypatch)
    client = TestClient(build_app("mock", mic=True, wakeword_models={},
                                  start_unmuted=True))
    with client.websocket_connect("/ws") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "status" and hello["state"] == "listening"


def test_default_models_include_hey_jarvis_wake():
    from scribe.audio.wakewords import DEFAULT_MODELS
    assert DEFAULT_MODELS["wake"] == "hey_jarvis_v0.1"


# ---------------------------------------------------- browser mic streaming
def _amp_vad(monkeypatch):
    """Hub without onnxruntime models: amplitude-threshold fake Silero, so
    loud s16le frames count as speech and zeros as silence."""
    import numpy as np
    import scribe.audio.vad as vadmod

    class AmpVAD:
        def __call__(self, frame):
            return 1.0 if float(np.abs(frame).max()) > 0.1 else 0.0

        def reset(self):
            pass

    monkeypatch.setattr(vadmod, "SileroVAD", AmpVAD)


def _pcm(value, frames):
    import numpy as np
    return np.full(512 * frames, value, dtype="<i2").tobytes()


SPEECH = _pcm(20000, 20)     # ~0.64 s loud  (> min_s speech)
SILENCE = _pcm(0, 20)        # ~0.64 s quiet (> end_silence_ms)


def test_browser_stream_claim_dictate_release(monkeypatch):
    _amp_vad(monkeypatch)
    client = TestClient(build_app("mock", wakeword_models={}))
    with client.websocket_connect("/ws") as ws:
        _drain_until(ws, "status")
        ws.send_json({"type": "mic_stream", "action": "start"})
        assert _drain_until(ws, "mic_stream")["state"] == "granted"
        # stream starts muted (wake-word gate) — unmute like "hey Jarvis" would
        _drain_until(ws, "status", state="muted", detail="browser mic claimed")
        ws.send_json({"type": "intent", "name": "wake"})
        _drain_until(ws, "status", state="listening")

        ws.send_bytes(SPEECH)
        ws.send_bytes(SILENCE)
        _drain_until(ws, "status", state="heard")     # chunker closed it
        reply = _drain_until(ws, "reply")             # mock engine audio ack
        assert "(mock) heard" in reply["text"]

        ws.send_json({"type": "mic_stream", "action": "stop"})
        rel = _drain_until(ws, "mic_stream", state="released")
        assert rel["reason"] == "stopped"


def test_browser_stream_muted_audio_is_dropped(monkeypatch):
    _amp_vad(monkeypatch)
    client = TestClient(build_app("mock", wakeword_models={}))
    with client.websocket_connect("/ws") as ws:
        _drain_until(ws, "status")
        ws.send_json({"type": "mic_stream", "action": "start"})
        _drain_until(ws, "mic_stream")
        # muted (never woke): speech must not reach the engine; the socket
        # stays healthy and typed utterances still round-trip
        ws.send_bytes(SPEECH)
        ws.send_bytes(SILENCE)
        ws.send_json({"type": "utterance", "text": "x"})
        prop = _drain_until(ws, "proposal")
        assert prop["action"]["latex"] == "x"


def test_second_claim_denied_then_granted_after_release(monkeypatch):
    _amp_vad(monkeypatch)
    # `with TestClient(...)`: both ws sessions share ONE portal/event loop —
    # broadcasts between concurrently-open sockets deadlock otherwise.
    with TestClient(build_app("mock", wakeword_models={})) as client, \
         client.websocket_connect("/ws") as ws1, \
         client.websocket_connect("/ws") as ws2:
        _drain_until(ws1, "status")
        _drain_until(ws2, "status")
        ws1.send_json({"type": "mic_stream", "action": "start"})
        assert _drain_until(ws1, "mic_stream")["state"] == "granted"

        ws2.send_json({"type": "mic_stream", "action": "start"})
        denied = _drain_until(ws2, "mic_stream", state="denied")
        assert "another client" in denied["reason"]

        # non-owner audio: dropped, exactly one error frame ever
        ws2.send_bytes(SILENCE)
        err = _drain_until(ws2, "error")
        assert "not the mic-stream owner" in err["message"]
        ws2.send_bytes(SILENCE)
        ws2.send_json({"type": "edit_preview", "target_id": "e1"})  # ping
        while True:
            m = ws2.receive_json()
            assert m["type"] != "error"                # no second error
            if m["type"] == "edit_preview":
                break

        ws1.send_json({"type": "mic_stream", "action": "stop"})
        _drain_until(ws2, "mic_stream", state="released")
        ws2.send_json({"type": "mic_stream", "action": "start"})
        assert _drain_until(ws2, "mic_stream", state="granted")


def test_owner_disconnect_releases_stream(monkeypatch):
    _amp_vad(monkeypatch)
    # shared portal: see test_second_claim_denied_then_granted_after_release
    with TestClient(build_app("mock", wakeword_models={})) as client, \
         client.websocket_connect("/ws") as ws2:
        _drain_until(ws2, "status")
        with client.websocket_connect("/ws") as ws1:
            _drain_until(ws1, "status")
            ws1.send_json({"type": "mic_stream", "action": "start"})
            _drain_until(ws1, "mic_stream", state="granted")
        rel = _drain_until(ws2, "mic_stream", state="released")
        assert "disconnected" in rel["reason"]
        ws2.send_json({"type": "mic_stream", "action": "start"})
        assert _drain_until(ws2, "mic_stream")["state"] == "granted"


def test_claim_denied_in_mic_mode(monkeypatch):
    _fake_mic_stack(monkeypatch)
    client = TestClient(build_app("mock", mic=True, wakeword_models={}))
    with client.websocket_connect("/ws") as ws:
        _drain_until(ws, "status")
        ws.send_json({"type": "mic_stream", "action": "start"})
        denied = _drain_until(ws, "mic_stream", state="denied")
        assert "server owns the mic" in denied["reason"]


def test_mic_tester_works_with_browser_stream(monkeypatch):
    _amp_vad(monkeypatch)
    client = TestClient(build_app("mock", wakeword_models={}))
    with client.websocket_connect("/ws") as ws:
        _drain_until(ws, "status")
        ws.send_json({"type": "mic_stream", "action": "start"})
        _drain_until(ws, "mic_stream", state="granted")
        ws.send_json({"type": "mic", "action": "test_start"})
        _drain_until(ws, "status", detail="mic test on")
        ws.send_bytes(SILENCE)
        lvl = _drain_until(ws, "miclevel")
        assert lvl["rms"] == 0.0 and lvl["prob"] == 0.0
        ws.send_json({"type": "mic", "action": "test_stop"})
        _drain_until(ws, "status", detail="mic test off")


# ------------------------------------------------------------- edit_preview
def test_edit_preview_relays_without_mutation_or_logging():
    from pathlib import Path
    client = TestClient(build_app("mock"))
    with client.websocket_connect("/ws") as ws:
        _drain_until(ws, "status")
        ws.send_json({"type": "utterance", "text": "a"})
        p = _drain_until(ws, "proposal")
        ws.send_json({"type": "resolve",
                      "pending_id": p["pending_id"], "verdict": "commit"})
        _drain_until(ws, "applied")

        # live keystrokes relay verbatim...
        ws.send_json({"type": "edit_preview", "target_id": "e1", "latex": "x^2"})
        pv = _drain_until(ws, "edit_preview")
        assert pv["target_id"] == "e1" and pv["latex"] == "x^2"
        # ...and a missing/null latex (edit ended without save) relays as None
        ws.send_json({"type": "edit_preview", "target_id": "e1"})
        assert _drain_until(ws, "edit_preview")["latex"] is None

        # DocState untouched by previews: e1 still "a", next append gets e2
        ws.send_json({"type": "utterance", "text": "b"})
        p2 = _drain_until(ws, "proposal")
        ws.send_json({"type": "resolve",
                      "pending_id": p2["pending_id"], "verdict": "commit"})
        applied = _drain_until(ws, "applied")
        assert applied["doc_context"] == "[e1] a  [e2] b"

    # previews never become training records: only the two commits are logged
    session = next(Path("data/sessions").iterdir())      # conftest chdir'd to tmp
    assert len(list(session.glob("*.json"))) == 2
