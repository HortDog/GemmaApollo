"""SpoolingLogger: local-first writes, background upload, retry, backlog
drain. MockTransport only — no sockets."""
import json
import time
from pathlib import Path

import httpx

from scribe.datalogger import SpoolingLogger

ROW = dict(audio_bytes=b"RIFF-fake-wav", doc_context="[e1] x",
           transcript="x squared", engine_action={"action": "append_math"},
           final_action={"action": "append_math"}, verdict="committed",
           engine="s2l", latency_ms={})


def wait_for(cond, timeout=3.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.02)
    return False


def make_logger(tmp_path, handler, retry_s=0.05):
    client = httpx.Client(transport=httpx.MockTransport(handler), timeout=5)
    return SpoolingLogger("http://model/log", root=tmp_path / "sessions",
                          client=client, retry_s=retry_s)


def test_uploads_row_and_moves_to_uploaded(tmp_path):
    calls = []

    def handler(req):
        calls.append(json.loads(req.content))
        return httpx.Response(200, json={"stored": True, "dup": False})

    lg = make_logger(tmp_path, handler)
    try:
        stem = lg.log(**ROW)
        moved = stem.parent / "uploaded" / (stem.name + ".json")
        assert wait_for(moved.exists)
        assert (stem.parent / "uploaded" / (stem.name + ".wav")).exists()
        body = calls[0]
        assert body["seq"] == 1
        assert body["session_id"] == stem.parent.name
        assert body["client_id"]                      # hostname-suffix
        assert body["payload"]["verdict"] == "committed"
        assert body["audio_wav"]                      # b64 of the wav
    finally:
        lg.close()


def test_local_write_never_waits_on_dead_server(tmp_path):
    def handler(req):
        raise httpx.ConnectError("down")

    lg = make_logger(tmp_path, handler)
    try:
        t0 = time.monotonic()
        stem = lg.log(**ROW)
        assert time.monotonic() - t0 < 0.5            # log() is local-only
        assert stem.with_suffix(".json").exists()     # row is on disk
        time.sleep(0.3)
        assert stem.with_suffix(".json").exists()     # still local, not moved
        assert not (stem.parent / "uploaded").exists()
    finally:
        lg.close()


def test_retries_until_server_comes_back(tmp_path):
    state = {"up": False, "attempts": 0}

    def handler(req):
        state["attempts"] += 1
        if not state["up"]:
            raise httpx.ConnectError("down")
        return httpx.Response(200, json={"stored": True, "dup": False})

    lg = make_logger(tmp_path, handler)
    try:
        stem = lg.log(**ROW)
        assert wait_for(lambda: state["attempts"] >= 2)   # kept retrying
        state["up"] = True
        moved = stem.parent / "uploaded" / (stem.name + ".json")
        assert wait_for(moved.exists)                     # then drained
    finally:
        lg.close()


def test_drains_backlog_from_earlier_sessions(tmp_path):
    old = tmp_path / "sessions" / "20260101-000000"
    old.mkdir(parents=True)
    (old / "0001.json").write_text(json.dumps({"verdict": "committed"}))
    (old / "0001.wav").write_bytes(b"old-wav")
    calls = []

    def handler(req):
        calls.append(json.loads(req.content))
        return httpx.Response(200, json={"stored": True, "dup": False})

    lg = make_logger(tmp_path, handler)
    try:
        assert wait_for(lambda: (old / "uploaded" / "0001.json").exists())
        assert calls[0]["session_id"] == "20260101-000000"
    finally:
        lg.close()


def test_client_id_persists_across_loggers(tmp_path):
    def handler(req):
        return httpx.Response(200, json={"stored": True, "dup": False})

    lg1 = make_logger(tmp_path, handler); lg1.close()
    lg2 = make_logger(tmp_path, handler); lg2.close()
    assert lg1.client_id == lg2.client_id
    assert (tmp_path / "client_id").exists()
