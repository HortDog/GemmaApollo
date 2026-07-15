"""GemmaEngine — fine-tuned Gemma 4 12B (GemmaApollo), end-to-end
audio -> Action function call. BLOCKED on training (PLAN.md Phase 8).
Prompt build: system prompt (thinking-mode suppressed) + doc_context with
stable IDs + audio; parse Gemma's native function-call output into the frozen
Action schema (engine/base.py). Must behave identically to S2LEngine from the
app's point of view."""
from __future__ import annotations
from .base import EngineResult

class GemmaEngine:
    name = "gemma"

    def __init__(self, checkpoint: str = "TBD/gemmapollo-12b", device: str = "auto"):
        self.checkpoint = checkpoint

    def process(self, audio, doc_context: str) -> EngineResult:
        raise NotImplementedError("awaiting GemmaApollo training")

    def process_text(self, transcript: str, doc_context: str) -> EngineResult:
        raise NotImplementedError
