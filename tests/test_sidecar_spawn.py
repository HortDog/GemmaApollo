"""Regression: the Electron-style spawn (stdin pipe + SCRIBE_PARENT_WATCH)
must not deadlock the mic claim.

Root cause it guards against: lazily importing onnxruntime->numpy on the
event loop deadlocked in OpenBLAS's DllMain while the parentwatch stdin
thread existed (Windows loader-lock hang). Fixed by pre-importing the audio
deps single-threaded in `scribe serve` startup. Spawns a real subprocess —
the only way to reproduce the process-level condition."""
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("onnxruntime")
REPO = Path(__file__).resolve().parents[1]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_mic_claim_survives_electron_style_spawn():
    port = free_port()
    env = {**os.environ, "SCRIBE_PARENT_WATCH": "1"}
    env.pop("ELECTRON_RUN_AS_NODE", None)
    scribe = Path(sys.executable).parent / ("scribe.exe" if os.name == "nt"
                                            else "scribe")
    if not scribe.exists():
        pytest.skip("scribe entry point not installed in this venv")
    p = subprocess.Popen(
        [str(scribe), "serve", "--engine", "mock", "--port", str(port)],
        cwd=REPO, env=env, stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        t0 = time.monotonic()
        while time.monotonic() - t0 < 30:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1)
                break
            except Exception:
                time.sleep(0.3)
        else:
            pytest.fail("sidecar did not boot")

        from websockets.sync.client import connect
        ws = connect(f"ws://127.0.0.1:{port}/ws")
        json.loads(ws.recv(timeout=10))                     # hello
        ws.send(json.dumps({"type": "mic_stream", "action": "start"}))
        t0 = time.monotonic()
        while time.monotonic() - t0 < 20:
            m = json.loads(ws.recv(timeout=20))
            if m["type"] == "mic_stream":
                assert m["state"] == "granted"
                break
            if m["type"] == "error":
                pytest.fail(f"claim errored: {m}")
        else:
            pytest.fail("mic claim wedged (no reply in 20 s)")

        # the loop must still dispatch other frames afterwards
        ws.send(json.dumps({"type": "utterance", "text": "x squared"}))
        t0 = time.monotonic()
        while time.monotonic() - t0 < 15:
            m = json.loads(ws.recv(timeout=15))
            if m["type"] == "proposal":
                break
        else:
            pytest.fail("dispatch dead after mic claim")
        ws.close()
    finally:
        subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"],
                       capture_output=True)
