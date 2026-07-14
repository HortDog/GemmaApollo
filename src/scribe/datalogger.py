"""Training-triple logger — every utterance becomes GemmaApollo data.
Layout: data/sessions/<session_ts>/NNN.wav + NNN.json  (see PLAN.md Phase 6)."""
from __future__ import annotations
import json, time
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
