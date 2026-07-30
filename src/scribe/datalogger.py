"""Training-triple logger — every utterance becomes GemmaApollo data.
Layout: data/sessions/<session_ts>/NNN.wav + NNN.json  (see PLAN.md Phase 6).
SpoolingLogger mirrors each row to the model server's central /log store."""
from __future__ import annotations
import base64, json, queue, socket, threading, time, uuid
from pathlib import Path

class SessionLogger:
    def __init__(self, root: Path = Path("data/sessions")):
        self.dir = root / time.strftime("%Y%m%d-%H%M%S")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.n = 0

    def log(self, *, audio_bytes: bytes | None, doc_context: str,
            transcript: str | None, engine_action: dict,
            final_action: dict | None, verdict: str, engine: str,
            latency_ms: dict) -> Path:
        """verdict: committed | scratched | edited_after | app_intent"""
        self.n += 1
        stem = self.dir / f"{self.n:04d}"
        if audio_bytes:
            (stem.with_suffix(".wav")).write_bytes(audio_bytes)
        rec = dict(doc_context=doc_context, transcript=transcript,
                   engine_action=engine_action, final_action=final_action,
                   verdict=verdict, engine=engine, latency_ms=latency_ms,
                   ts=time.time())
        stem.with_suffix(".json").write_text(json.dumps(rec, indent=1))
        return stem
    # TODO(Phase 6): `scribe export` -> JSONL in the training format
    # (handover doc §5); attach edited_after gold labels.


def _client_id(state_dir: Path) -> str:
    """hostname + persisted random suffix — stable per install, collision-free
    across machines in the central dataset."""
    f = state_dir / "client_id"
    if f.exists():
        return f.read_text().strip()
    cid = f"{socket.gethostname()}-{uuid.uuid4().hex[:6]}"
    state_dir.mkdir(parents=True, exist_ok=True)
    f.write_text(cid)
    return cid


def _leftover_rows(root: Path, skip: Path) -> list[Path]:
    """Un-uploaded stems from earlier sessions (crash/offline backlog).
    uploaded/ subdirs are not descended into; the current session dir is
    skipped (its rows arrive through the live queue)."""
    stems = []
    if not root.exists():
        return stems
    for sdir in sorted(p for p in root.iterdir() if p.is_dir() and p != skip):
        stems += sorted(j.with_suffix("") for j in sdir.glob("*.json"))
    return stems


class SpoolingLogger(SessionLogger):
    """SessionLogger + best-effort central upload (electron-3tier plan).

    Rows ALWAYS land locally first — the logging hard rule never waits on
    the network. A daemon thread mirrors each row to the model server's
    POST /log and moves the local files into <session>/uploaded/ on 200;
    while the server is unreachable it retries with a delay, including any
    backlog left over from earlier runs."""

    def __init__(self, upload_url: str, root: Path = Path("data/sessions"),
                 client=None, retry_s: float = 5.0):
        super().__init__(root)
        import httpx
        self.client_id = _client_id(root.parent if root.parent != Path("")
                                    else Path("."))
        self._url = upload_url
        self._client = client or httpx.Client(timeout=10.0)
        self._retry_s = retry_s
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        for stem in _leftover_rows(root, skip=self.dir):
            self._q.put(stem)
        self._t = threading.Thread(target=self._worker, daemon=True,
                                   name="logspool")
        self._t.start()

    def log(self, **kw) -> Path:
        stem = super().log(**kw)
        self._q.put(stem)
        return stem

    def close(self):
        self._stop.set()
        self._t.join(timeout=2)

    def _worker(self):
        while not self._stop.is_set():
            try:
                stem = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if not self._upload(stem):
                self._q.put(stem)              # server unreachable: back off,
                self._stop.wait(self._retry_s)  # keep the row queued

    def _upload(self, stem: Path) -> bool:
        json_path = stem.with_suffix(".json")
        if not json_path.exists():
            return True                        # already moved (dup requeue)
        wav_path = stem.with_suffix(".wav")
        body = {
            "client_id": self.client_id,
            "session_id": stem.parent.name,
            "seq": int(stem.name),
            "audio_wav": (base64.b64encode(wav_path.read_bytes()).decode()
                          if wav_path.exists() else None),
            "payload": json.loads(json_path.read_text()),
        }
        try:
            r = self._client.post(self._url, json=body)
            r.raise_for_status()
        except Exception:
            return False
        updir = stem.parent / "uploaded"
        updir.mkdir(exist_ok=True)
        for p in (json_path, wav_path):
            if p.exists():
                p.replace(updir / p.name)
        return True
