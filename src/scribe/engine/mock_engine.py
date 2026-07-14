"""Zero-model engine for Phase 2: toy spoken->LaTeX + the regex router.
Mirrors the converter in frontend/index.html so behavior matches local mode."""
from __future__ import annotations
import re, time
from .base import EngineResult, TextReply
from ..router import route

_PHRASES = [
    (re.compile(r"^e equals m c squared$", re.I), r"E = mc^2"),
    (re.compile(r"^one half m v squared$", re.I), r"\frac{1}{2}mv^2"),
    (re.compile(r"^the integral from zero to infinity of e to the minus x d x$", re.I),
     r"\int_0^{\infty} e^{-x}\,dx"),
]

def toy_to_latex(s: str) -> str:
    s = s.strip()
    if re.search(r"[\\^_{}]", s):
        return s
    for pat, tex in _PHRASES:
        if pat.match(s):
            return tex
    t = f" {s.lower()} "
    t = re.sub(r"([a-z0-9]) squared", r"\1^2", t)
    t = re.sub(r"([a-z0-9]) cubed", r"\1^3", t)
    t = re.sub(r"\b(alpha|beta|gamma|delta|theta|lambda|mu|pi|sigma|phi|omega)\b", r"\\\1", t)
    t = t.replace(" plus ", " + ").replace(" minus ", " - ")
    t = t.replace(" times ", r" \cdot ").replace(" equals ", " = ")
    parts = t.split(" over ")
    if len(parts) == 2:
        t = rf"\frac{{{parts[0].strip()}}}{{{parts[1].strip()}}}"
    return re.sub(r"\s+", " ", t).strip()

class MockEngine:
    name = "mock"

    def process(self, audio, doc_context: str) -> EngineResult:
        raise NotImplementedError("mock engine is text-only; use process_text")

    def process_text(self, transcript: str, doc_context: str) -> EngineResult:
        t0 = time.perf_counter()
        a = route(transcript, toy_to_latex)
        ms = (time.perf_counter() - t0) * 1000
        if a is None or isinstance(a, str):
            a = TextReply(text=f"app intent: {a}")
        return EngineResult(action=a, transcript=transcript, engine=self.name,
                            latency_ms={"total": ms})
