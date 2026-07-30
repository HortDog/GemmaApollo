"""Model server wire: /process, /process_text, /health, /warmup, /log.
FakeEngine only — no torch, no models, no network."""
import base64
import json

import numpy as np
from starlette.testclient import TestClient

from scribe.audio.vad import wav_bytes, wav_to_float32
from scribe.engine.base import AppendMath, EngineResult
from scribe.model_server import build_model_app


class FakeEngine:
    name = "fake"

    def __init__(self, boom=False):
        self.calls = []
        self.boom = boom

    def process(self, audio, doc_context):
        if self.boom:
            raise RuntimeError("cuda exploded")
        self.calls.append(("audio", np.asarray(audio), doc_context))
        return EngineResult(action=AppendMath(latex="x^2"),
                            transcript="x squared", engine=self.name,
                            latency_ms={"total": 1.0})

    def process_text(self, transcript, doc_context):
        self.calls.append(("text", transcript, doc_context))
        return EngineResult(action=AppendMath(latex=transcript),
                            engine=self.name)


def b64wav(x, rate=16000):
    return base64.b64encode(wav_bytes(x, sample_rate=rate)).decode()


def test_process_round_trip(tmp_path):
    eng = FakeEngine()
    tc = TestClient(build_model_app(eng, data_root=tmp_path))
    pcm = np.full(16000, 0.25, dtype=np.float32)
    r = tc.post("/process", json={"audio_wav": b64wav(pcm),
                                  "doc_context": "[e1] y"})
    assert r.status_code == 200
    res = EngineResult.model_validate(r.json())     # frozen schema round-trips
    assert res.action.latex == "x^2" and res.engine == "fake"
    kind, got, ctx = eng.calls[0]
    assert kind == "audio" and ctx == "[e1] y"
    assert got.dtype == np.float32 and got.size == 16000
    assert np.allclose(got, 0.25, atol=1e-3)        # wav 16-bit quantization


def test_health_loaded_flips_without_triggering_load(tmp_path):
    tc = TestClient(build_model_app(FakeEngine(), data_root=tmp_path))
    assert tc.get("/health").json() == {"engine": "fake", "loaded": False,
                                        "device": None}
    tc.post("/process_text", json={"transcript": "x"})
    assert tc.get("/health").json()["loaded"] is True


def test_warmup_runs_audio_path(tmp_path):
    eng = FakeEngine()
    tc = TestClient(build_model_app(eng, data_root=tmp_path))
    assert tc.post("/warmup").status_code == 200
    assert eng.calls[0][1].size == 16000            # 1 s of zeros


def test_bad_wav_is_400_not_500(tmp_path):
    tc = TestClient(build_model_app(FakeEngine(), data_root=tmp_path))
    r = tc.post("/process", json={"audio_wav": b64wav(np.zeros(800), rate=8000)})
    assert r.status_code == 400 and "16000 Hz" in r.json()["detail"]
    r = tc.post("/process", json={"audio_wav": base64.b64encode(b"junk").decode()})
    assert r.status_code == 400


def test_engine_exception_is_500(tmp_path):
    tc = TestClient(build_model_app(FakeEngine(boom=True), data_root=tmp_path))
    r = tc.post("/process", json={"audio_wav": b64wav(np.zeros(16000))})
    assert r.status_code == 500 and "cuda exploded" in r.json()["detail"]


def test_wav_round_trip():
    x = np.sin(np.linspace(0, 30, 16000)).astype(np.float32) * 0.5
    assert np.allclose(wav_to_float32(wav_bytes(x)), x, atol=1e-4)


# ------------------------------------------------------------------- /log
def _row(seq=1, with_audio=True, client="box-abc123", session="20260730-120000"):
    return {"client_id": client, "session_id": session, "seq": seq,
            "audio_wav": b64wav(np.zeros(1600)) if with_audio else None,
            "payload": {"verdict": "committed", "transcript": "x squared"}}


def test_log_stores_and_dedupes(tmp_path):
    tc = TestClient(build_model_app(FakeEngine(), data_root=tmp_path))
    assert tc.post("/log", json=_row()).json() == {"stored": True, "dup": False}
    stem = tmp_path / "box-abc123" / "20260730-120000" / "0001"
    assert stem.with_suffix(".wav").exists()
    assert json.loads(stem.with_suffix(".json").read_text())["verdict"] == "committed"
    # idempotent retry
    assert tc.post("/log", json=_row()).json() == {"stored": False, "dup": True}


def test_log_wakeword_row_has_no_wav(tmp_path):
    tc = TestClient(build_model_app(FakeEngine(), data_root=tmp_path))
    tc.post("/log", json=_row(seq=2, with_audio=False))
    stem = tmp_path / "box-abc123" / "20260730-120000" / "0002"
    assert stem.with_suffix(".json").exists()
    assert not stem.with_suffix(".wav").exists()


def test_log_sanitizes_path_components(tmp_path):
    tc = TestClient(build_model_app(FakeEngine(), data_root=tmp_path))
    r = tc.post("/log", json=_row(client="../../evil", session="a/b"))
    assert r.status_code == 200
    written = list(tmp_path.rglob("0001.json"))
    assert len(written) == 1
    assert tmp_path in written[0].parents           # never escaped data_root
    r = tc.post("/log", json=_row(client="..."))    # nothing left after strip
    assert r.status_code == 400
