"""Pure-python text metrics + LaTeX cleanup. No torch — testable everywhere.

Vendored (lightly adapted) from github.com/DingoOz/speech-to-LateX
(eval/cer.py, src/speech_to_latex/normalize.py), which implements the
CER definition of arXiv:2508.03542 Appendix A.2:

    CER = (S + D + I) / N
"""
from __future__ import annotations

import re

_MATH_DELIMS = re.compile(r"^\${1,2}|\${1,2}$")
_EXTRA_SPACE = re.compile(r"\s+")
_SPACE_BEFORE_BRACE = re.compile(r"\s+([\{\}\^_])")


def normalize_latex(latex: str) -> str:
    """Best-effort cleanup of generated LaTeX: whitespace, stray `$` fences,
    space-before-brace. (The paper uses a KaTeX AST rebuild; this is the
    cheap pure-python approximation from the reference CLI.)"""
    text = latex.strip()
    text = _MATH_DELIMS.sub("", text).strip()
    text = _EXTRA_SPACE.sub(" ", text)
    text = _SPACE_BEFORE_BRACE.sub(r"\1", text)
    return text.strip()


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            current[j] = min(
                previous[j] + 1,        # deletion
                current[j - 1] + 1,     # insertion
                previous[j - 1] + cost, # substitution
            )
        previous = current
    return previous[-1]


def cer(prediction: str, reference: str, lowercase: bool = True) -> float:
    """Character error rate of `prediction` against `reference`, in [0, inf)."""
    if lowercase:
        prediction, reference = prediction.lower(), reference.lower()
    if not reference:
        return 0.0 if not prediction else 1.0
    return levenshtein(prediction, reference) / len(reference)
