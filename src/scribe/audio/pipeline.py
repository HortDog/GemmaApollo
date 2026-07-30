"""Per-frame audio fan-out shared by every capture source.

Extracted from the sounddevice thread in server.py so the same
VAD -> spotter -> chunker -> levels path can be fed by either the backend
mic (`--mic`) or binary ws `audio` frames streamed from a client
(PROTOCOL.md browser-mic mode). Pure python/numpy — the VAD and wake-word
scorer are injected callables (CLAUDE.md: pure logic stays torch-free).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .vad import FRAME_SAMPLES

_FRAME_BYTES = FRAME_SAMPLES * 2       # s16le


class PCMReframer:
    """s16le 16 kHz mono byte stream -> FRAME_SAMPLES float32 [-1,1] frames.

    Network chunks may split mid-sample and mid-frame; leftover bytes carry
    over to the next feed().
    """

    def __init__(self):
        self._buf = b""

    def feed(self, data: bytes) -> list[np.ndarray]:
        self._buf += data
        n = len(self._buf) // _FRAME_BYTES
        if not n:
            return []
        whole, self._buf = self._buf[:n * _FRAME_BYTES], self._buf[n * _FRAME_BYTES:]
        i16 = np.frombuffer(whole, dtype="<i2")
        f32 = i16.astype(np.float32) / 32768.0
        return [f32[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES] for i in range(n)]

    def reset(self) -> None:
        self._buf = b""


@dataclass
class FramePipeline:
    """Fan one 512-sample float32 frame out to VAD, wake-word spotter,
    utterance chunker, and the mic-test level meter. Call from exactly one
    thread at a time.

    Semantics (must match the historical server.capture() behavior):
    - the spotter sees every frame, even while muted — "hey Jarvis" must
      wake, and commit/undo/scratch stay active by design
    - mute starves the chunker only; the unmuted->muted edge drops any
      half-captured utterance so nothing recorded before muting leaks out
    - level dicts are emitted every 3rd frame (~10 Hz) while testing
    """
    vad: Callable[[np.ndarray], float]
    chunker: object                            # UtteranceChunker-compatible
    spotter: Optional[object] = None           # IntentSpotter-compatible
    is_muted: Callable[[], bool] = lambda: False
    is_testing: Callable[[], bool] = lambda: False
    on_utterance: Callable[[np.ndarray], None] = lambda u: None
    on_intent: Callable[[str], None] = lambda name: None
    on_level: Callable[[dict], None] = lambda lvl: None

    _i: int = field(init=False, default=0)
    _was_muted: bool = field(init=False)

    def __post_init__(self):
        self._was_muted = self.is_muted()

    def feed(self, frame: np.ndarray) -> None:
        prob = self.vad(frame)
        if self.spotter is not None:
            hit = self.spotter.feed(frame)
            if hit:
                self.on_intent(hit)
        muted = self.is_muted()
        if muted != self._was_muted:
            self._was_muted = muted
            if muted:
                self.chunker.reset()
        if not muted:
            u = self.chunker.feed(frame, prob=prob)
            if u is not None:
                self.on_utterance(u)
        if self.is_testing() and self._i % 3 == 0:
            rms = float(np.sqrt(float((frame ** 2).mean())))
            self.on_level({"rms": rms, "prob": prob})
        self._i += 1

    def reset(self) -> None:
        """Drop any half-captured utterance (stream release, device switch)."""
        self.chunker.reset()
        self._i = 0
