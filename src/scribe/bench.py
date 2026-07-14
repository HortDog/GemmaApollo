"""`scribe bench` — run fixture clips through an engine and report per-stage
latency (asr / route / total) and CER against fixture labels (PLAN.md Phase 3).

Fixtures live in tests/fixtures/: <name>.wav files plus labels.json:
    [{"wav": "euler_identity.wav", "spoken": "<tts text>", "latex": "<gold>"}]

The mock engine has no audio path, so it benches the text route only (useful
as a zero-model smoke test); the s2l engine benches wav -> Action end to end.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from .metrics import cer

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


def load_labels(fixtures: Path = FIXTURES) -> list[dict]:
    return json.loads((fixtures / "labels.json").read_text(encoding="utf-8"))


def run_bench(engine_name: str = "s2l", asr_model: str | None = None,
              fixtures: Path = FIXTURES) -> dict:
    from .server import make_engine
    engine = make_engine(engine_name)
    if asr_model and hasattr(engine, "asr_model"):
        engine.asr_model = asr_model

    labels = load_labels(fixtures)
    rows, stage_ms, scores = [], {}, []
    for lab in labels:
        wav = fixtures / lab["wav"]
        if engine_name == "mock" or not wav.exists():
            res = engine.process_text(lab["spoken"], "")
        else:
            res = engine.process(wav, "")
        pred = getattr(res.action, "latex", "") or ""
        score = cer(pred, lab["latex"])
        # LaTeX rendering is (mostly) whitespace-insensitive and our refs are
        # hand-spaced, so also score with spaces stripped — closer to the
        # paper's KaTeX-normalized comparison.
        score_ns = cer(pred.replace(" ", ""), lab["latex"].replace(" ", ""))
        scores.append(score_ns)
        for stage, ms in res.latency_ms.items():
            stage_ms.setdefault(stage, []).append(ms)
        rows.append({"wav": lab["wav"], "transcript": res.transcript,
                     "pred": pred, "ref": lab["latex"], "cer": score,
                     "cer_nospace": score_ns, "latency_ms": res.latency_ms})

    summary = {
        "engine": engine.name,
        "asr_model": getattr(engine, "asr_model", None),
        "n": len(rows),
        "median_ms": {s: statistics.median(v) for s, v in stage_ms.items()},
        "mean_cer": statistics.mean(scores) if scores else None,
        "rows": rows,
    }
    return summary


def print_summary(s: dict) -> None:
    print(f"engine={s['engine']}  asr_model={s['asr_model']}  clips={s['n']}")
    for row in s["rows"]:
        lat = "  ".join(f"{k}={v:.0f}ms" for k, v in row["latency_ms"].items())
        print(f"  {row['wav']:<28} CER={row['cer']:.3f} (nospace {row['cer_nospace']:.3f})  {lat}")
        print(f"    transcript: {row['transcript']!r}")
        print(f"    pred: {row['pred']!r}")
        print(f"    ref:  {row['ref']!r}")
    med = "  ".join(f"{k}={v:.0f}ms" for k, v in s["median_ms"].items())
    print(f"median latency: {med}")
    print(f"mean CER (whitespace-stripped): {s['mean_cer']:.3f}")
