"""Generate synthetic positive samples for one wake phrase via
piper-sample-generator (see README.md; Linux/WSL/Colab recommended).

Usage:
    python tools/wakewords/generate_samples.py --phrase "scratch that" --n 4000
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

OUT_ROOT = Path("data/wakewords/positives")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phrase", required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--model", default="en_US-libritts_r-medium.pt",
                    help="piper-sample-generator voice checkpoint")
    args = ap.parse_args()

    out = OUT_ROOT / args.phrase.replace(" ", "_")
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "piper_sample_generator.generate_samples",
           args.phrase, "--max-samples", str(args.n),
           "--output-dir", str(out), "--model", args.model]
    print("running:", " ".join(cmd))
    try:
        raise SystemExit(subprocess.call(cmd))
    except FileNotFoundError:
        raise SystemExit(
            "piper-sample-generator not installed. Clone "
            "github.com/rhasspy/piper-sample-generator and install its "
            "requirements (Linux/WSL/Colab recommended), then re-run.")


if __name__ == "__main__":
    main()
