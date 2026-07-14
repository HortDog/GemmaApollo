"""S2LEngine — faster-whisper ASR + marsianin500 Qwen2.5 post-correction
(Speech-to-LaTeX, arXiv:2508.03542). Model-loading code vendored/adapted from
github.com/DingoOz/speech-to-LateX. Lazy model loading; keep total VRAM
< 10 GB (see CLAUDE.md).

Module import must stay torch-free (server imports this to build the engine);
everything heavy happens inside _lazy_load()."""
from __future__ import annotations

import json
import re
import time

from pydantic import TypeAdapter, ValidationError

from ..metrics import normalize_latex
from ..router import route, resolver_from_context
from .base import Action, Clarify, EngineResult

# Best released equation checkpoint per the paper's Table 2/5 results.
CORRECTOR_MODEL = "marsianin500/Qwen2.5-0.5B-instruct-equations_multilingual_mix_full"
CORRECTOR_PROMPT = (
    "Recognize the speech and convert the content into text. "
    "Any mathematical expressions should be transcribed in LaTeX format."
)

# Transcripts that start like a command but miss the regex grammar go to the
# LLM fallback with this strict-JSON prompt. One Action object, nothing else.
_COMMAND_VERBS = re.compile(
    r"^(change|replace|delete|remove|insert|label|set|make|swap|move)\b", re.I)
COMMAND_PROMPT = """You convert a spoken editing command over a LaTeX document into ONE JSON action.
Document lines: {doc_context}
The JSON must be exactly one of:
{{"action":"replace","target_id":"e<N>","latex":"..."}}
{{"action":"insert","after_id":"e<N>","latex":"..."}}
{{"action":"delete","target_id":"e<N>"}}
{{"action":"set_label","target_id":"e<N>","text":"..."}}
Reply with the JSON object only, no prose, no code fences.
Command: {transcript}"""

_ACTION_ADAPTER = TypeAdapter(Action)


class S2LEngine:
    name = "s2l"

    def __init__(self, asr_model: str = "large-v3", device: str = "cuda",
                 compute_type: str = "int8"):
        # asr_model: "large-v3" | "large-v3-turbo" | "distil-large-v3"
        # (config switch per CLAUDE.md; benchmark via `scribe bench`).
        self.asr_model = asr_model
        self.device = device
        self.compute_type = compute_type
        self._asr = None        # faster_whisper.WhisperModel
        self._tokenizer = None
        self._corrector = None  # marsianin500 post-correction checkpoint

    # ------------------------------------------------------------------ load
    def _lazy_load(self):
        if self._asr is not None:
            return
        # Import torch FIRST: it registers its bundled cuBLAS/cuDNN DLLs in
        # the process, which CTranslate2 (faster-whisper) then finds — the
        # standard fix for GPU faster-whisper on Windows without a system CUDA.
        import torch
        from faster_whisper import WhisperModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._asr = WhisperModel(self.asr_model, device=self.device,
                                 compute_type=self.compute_type)
        # Warm up CTranslate2 BEFORE torch touches cuBLAS: ct2 must create its
        # CUDA/cuBLAS handles first or torch matmuls later fail with
        # CUBLAS_STATUS_EXECUTION_FAILED (observed on Windows, ct2 4.8 +
        # torch 2.11 sharing one process). One short silent clip suffices.
        import numpy as np
        list(self._asr.transcribe(np.zeros(8000, dtype=np.float32),
                                  language="en", beam_size=1)[0])
        self._tokenizer = AutoTokenizer.from_pretrained(CORRECTOR_MODEL)
        self._corrector = AutoModelForCausalLM.from_pretrained(
            CORRECTOR_MODEL,
            dtype=torch.float16 if self.device.startswith("cuda") else torch.float32,
        ).to(self.device)
        self._corrector.eval()

    # ------------------------------------------------------------------ stages
    def _transcribe(self, audio) -> str:
        """audio: path to a wav, or 16 kHz mono float32 numpy array."""
        source = str(audio) if not hasattr(audio, "dtype") else audio
        segments, _info = self._asr.transcribe(source, language="en", beam_size=5)
        return " ".join(s.text.strip() for s in segments).strip()

    def _generate(self, messages: list[dict], max_new_tokens: int = 256) -> str:
        import torch
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._corrector.device)
        with torch.no_grad():
            out = self._corrector.generate(**inputs, max_new_tokens=max_new_tokens,
                                           do_sample=False)
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _to_latex(self, spoken: str) -> str:
        """Spoken-math fragment -> LaTeX via the post-correction checkpoint.
        Fragments that already look like LaTeX pass through (UI text box)."""
        s = spoken.strip()
        if re.search(r"[\\^_{}]", s):
            return s
        out = self._generate([
            {"role": "system", "content": CORRECTOR_PROMPT},
            {"role": "user", "content": s},
        ])
        return normalize_latex(out)

    def _llm_command(self, transcript: str, doc_context: str) -> Action | None:
        """Strict-JSON LLM fallback for command-shaped transcripts the regex
        grammar missed. Returns a validated Action or None (caller falls back
        to dictation). Never trusts the model: pydantic-validate before use."""
        raw = self._generate([{
            "role": "user",
            "content": COMMAND_PROMPT.format(doc_context=doc_context or "(empty)",
                                             transcript=transcript),
        }], max_new_tokens=128)
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return None
        try:
            return _ACTION_ADAPTER.validate_python(json.loads(m.group(0)))
        except (json.JSONDecodeError, ValidationError):
            return None

    # ------------------------------------------------------------------ engine
    def _route(self, transcript: str, doc_context: str, latency: dict) -> Action:
        t0 = time.perf_counter()
        try:
            a = route(transcript, self._to_latex,
                      resolve=resolver_from_context(doc_context))
        except KeyError as e:
            a = Clarify(question=f"Which line did you mean? ({e.args[0]})")
        latency["route"] = (time.perf_counter() - t0) * 1000

        # Regex grammar missed but it smells like a command -> LLM fallback.
        if (a is not None and not isinstance(a, (str, Clarify))
                and a.action == "append_math" and _COMMAND_VERBS.match(transcript)):
            t0 = time.perf_counter()
            fb = self._llm_command(transcript, doc_context)
            latency["llm_fallback"] = (time.perf_counter() - t0) * 1000
            if fb is not None:
                a = fb
        if a is None or isinstance(a, str):
            # Empty utterance or app intent leaking through (spotter missed):
            # the engine never emits app intents — surface as a clarify.
            a = Clarify(question=f"App intent or empty utterance: {transcript!r}")
        return a

    def process(self, audio, doc_context: str) -> EngineResult:
        self._lazy_load()
        latency: dict[str, float] = {}
        t0 = time.perf_counter()
        transcript = self._transcribe(audio)
        latency["asr"] = (time.perf_counter() - t0) * 1000
        a = self._route(transcript, doc_context, latency)
        latency["total"] = sum(latency.values())
        return EngineResult(action=a, transcript=transcript, engine=self.name,
                            latency_ms=latency)

    def process_text(self, transcript: str, doc_context: str) -> EngineResult:
        self._lazy_load()
        latency: dict[str, float] = {}
        a = self._route(transcript, doc_context, latency)
        latency["total"] = sum(latency.values())
        return EngineResult(action=a, transcript=transcript, engine=self.name,
                            latency_ms=latency)
