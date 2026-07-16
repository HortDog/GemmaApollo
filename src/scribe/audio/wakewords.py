"""openWakeWord spotters for commit / undo / scratch-that, running
continuously on the raw mic stream in parallel with VAD (PLAN.md Phase 5).
Spotter hits bypass the engine entirely -> app-layer intent.

Layering (same pattern as vad.py):
- IntentSpotter: pure logic — buffers 512-sample float32 mic frames into the
  80 ms/1280-sample int16 chunks openWakeWord expects, applies per-intent
  thresholds and a refractory period. Scorer is injected; testable fake.
- OWWScorer: lazy openwakeword.Model wrapper (onnx runtime on Windows).

Custom "commit"/"undo"/"scratch that" models are trained with
tools/wakewords/ (piper-sample-generator + openwakeword training); until
those exist you can smoke-test the plumbing with any pretrained oww model
via --wakeword-model name=intent mappings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

CHUNK_SAMPLES = 1280        # 80 ms @ 16 kHz — openWakeWord's expected hop
SAMPLE_RATE = 16_000

# intent -> default custom model path (produced by tools/wakewords/train.py).
# `wake` uses openWakeWord's pretrained hey-jarvis model (bare name, no path):
# it unmutes dictation — the mic starts muted and "hey Jarvis" wakes it.
DEFAULT_MODELS = {
    "commit": "models/wakewords/commit.onnx",
    "undo": "models/wakewords/undo.onnx",
    "scratch": "models/wakewords/scratch_that.onnx",
    "wake": "hey_jarvis_v0.1",
}


@dataclass
class IntentSpotter:
    """Feed 512-sample float32 frames; returns an intent name when a wake
    word fires. `score` maps an int16 1280-sample chunk to {intent: prob}.

    - threshold: min score to fire
    - refractory_s: after any fire, all intents are suppressed this long
      (a single spoken "commit" spans several overlapping chunks — without
      this one word would fire repeatedly)
    """
    score: Callable[[np.ndarray], dict[str, float]]
    threshold: float = 0.5
    refractory_s: float = 2.0

    _buf: np.ndarray = field(init=False)
    _elapsed_s: float = field(init=False, default=0.0)
    _quiet_until: float = field(init=False, default=0.0)

    def __post_init__(self):
        self._buf = np.empty(0, dtype=np.int16)

    def feed(self, frame: np.ndarray) -> Optional[str]:
        i16 = (np.clip(frame, -1.0, 1.0) * 32767.0).astype(np.int16)
        self._buf = np.concatenate([self._buf, i16])
        fired: Optional[str] = None
        while len(self._buf) >= CHUNK_SAMPLES:
            chunk, self._buf = self._buf[:CHUNK_SAMPLES], self._buf[CHUNK_SAMPLES:]
            self._elapsed_s += CHUNK_SAMPLES / SAMPLE_RATE
            scores = self.score(chunk)
            if self._elapsed_s < self._quiet_until:
                continue
            best = max(scores, key=scores.get, default=None)
            if best is not None and scores[best] >= self.threshold:
                self._quiet_until = self._elapsed_s + self.refractory_s
                fired = best
        return fired


class OWWScorer:
    """Lazy openwakeword.Model as a `chunk -> {intent: prob}` callable.

    model_map: {model_name_or_path: intent}. Paths load custom models;
    bare names load openWakeWord pretrained models (plumbing smoke tests).
    """

    def __init__(self, model_map: dict[str, str]):
        self.model_map = model_map
        self._model = None
        self._key_to_intent: dict[str, str] = {}

    def _load(self):
        if self._model is not None:
            return
        from pathlib import Path

        from openwakeword.model import Model
        paths = list(self.model_map)
        # Bare pretrained names (e.g. hey_jarvis_v0.1) resolve from the
        # openwakeword package cache; fetch once on first use. Existing
        # files are skipped, and offline-with-cache just proceeds to Model().
        pretrained = [p for p in paths if not Path(p).exists()]
        if pretrained:
            try:
                from openwakeword.utils import download_models
                download_models(model_names=pretrained)
            except Exception as e:
                print(f"wakewords: pretrained download skipped ({e})", flush=True)
        self._model = Model(wakeword_models=paths, inference_framework="onnx")
        # oww keys predictions by model basename, not the given path
        for path, intent in self.model_map.items():
            for key in self._model.models:
                if key in path or path in key:
                    self._key_to_intent[key] = intent
        if not self._key_to_intent:  # name mismatch fallback: positional
            self._key_to_intent = dict(zip(self._model.models, self.model_map.values()))

    def __call__(self, chunk: np.ndarray) -> dict[str, float]:
        self._load()
        pred = self._model.predict(chunk)
        return {self._key_to_intent[k]: float(v) for k, v in pred.items()
                if k in self._key_to_intent}
