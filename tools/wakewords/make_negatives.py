"""Export recorded math-dictation audio from data/sessions/ as wake-word
NEGATIVES (PLAN.md Phase 5: spotters must never fire mid-equation, so real
dictation is the adversarial negative set).

Copies every session wav into data/wakewords/negatives/ with a flat name.

Usage:
    python tools/wakewords/make_negatives.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

SESSIONS = Path("data/sessions")
OUT = Path("data/wakewords/negatives")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for wav in sorted(SESSIONS.glob("*/*.wav")):
        shutil.copy2(wav, OUT / f"{wav.parent.name}_{wav.name}")
        n += 1
    print(f"exported {n} dictation wav(s) to {OUT}")
    if n == 0:
        print("no session audio yet — dictate with `scribe serve --mic` first")


if __name__ == "__main__":
    main()
