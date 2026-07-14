"""S2LEngine — Whisper-large-v3 (faster-whisper) ASR + marsianin500 Qwen2.5
post-correction (Speech-to-LaTeX, arXiv:2508.03542). Vendors/depends on
github.com/DingoOz/speech-to-LateX. Lazy model loading; keep total VRAM
< 10 GB (see CLAUDE.md). Implement in PLAN.md Phase 3."""
from __future__ import annotations
from .base import EngineResult

class S2LEngine:
    name = "s2l"

    def __init__(self, asr_model: str = "large-v3", device: str = "cuda"):
        self.asr_model = asr_model
        self.device = device
        self._asr = None       # faster_whisper.WhisperModel, int8
        self._corrector = None  # marsianin500 post-correction checkpoint, 4-bit

    def _lazy_load(self):
        # TODO(Phase 3): load faster-whisper (int8) and the post-correction
        # model; expose a config switch for large-v3-turbo / distil-large-v3
        # and benchmark all three with `scribe bench`.
        raise NotImplementedError

    def process(self, audio, doc_context: str) -> EngineResult:
        # TODO(Phase 3):
        # 1. transcript = ASR(audio)                                (stage: asr)
        # 2. action|intent = router.route(transcript, self._correct) (stage: route)
        #    - dictation fragments go through _correct (post-correction model)
        #    - grammar-missed commands -> LLM fallback (strict-JSON one Action)
        # 3. return EngineResult with per-stage latency_ms
        raise NotImplementedError

    def process_text(self, transcript: str, doc_context: str) -> EngineResult:
        # Same as process() minus stage 1 — used by the UI text box.
        raise NotImplementedError
