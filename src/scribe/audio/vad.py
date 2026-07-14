"""VAD chunker over a 16 kHz mono mic stream (PLAN.md Phase 4).

Layering (CLAUDE.md: pure logic stays torch-free):
- UtteranceChunker: pure segmentation logic driven by an injected
  `is_speech(frame) -> prob` callable. Testable with a fake VAD.
- SileroVAD: lazy torch wrapper providing that callable (32 ms / 512-sample
  frames at 16 kHz — silero v5's required window).
- MicStream: lazy sounddevice wrapper yielding 512-sample float32 frames.
- wav_bytes: float32 pcm -> 16-bit wav file bytes (for the datalogger).
"""
from __future__ import annotations

import io
import wave
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

import numpy as np

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 512                    # silero v5 window @16 kHz (32 ms)
FRAME_MS = FRAME_SAMPLES * 1000 / SAMPLE_RATE


def wav_bytes(pcm: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """float32 [-1,1] mono -> complete 16-bit PCM wav file bytes."""
    i16 = np.clip(pcm, -1.0, 1.0)
    i16 = (i16 * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(i16.tobytes())
    return buf.getvalue()


@dataclass
class UtteranceChunker:
    """Turns a stream of fixed-size frames + speech probabilities into
    utterance arrays. Pure python/numpy — inject the VAD.

    Rules (defaults per PLAN.md):
    - utterance opens when prob >= threshold
    - closes after `end_silence_ms` of continuous non-speech
    - shorter than `min_s` -> dropped (breath/click)
    - longer than `max_s` -> force-cut and emitted
    - `pad_ms` of pre-roll audio is prepended (attack consonants)
    """
    is_speech: Callable[[np.ndarray], float]
    threshold: float = 0.5
    min_s: float = 0.5
    max_s: float = 10.0
    pad_ms: float = 200.0
    end_silence_ms: float = 500.0

    _preroll: deque = field(init=False)
    _frames: list = field(init=False, default_factory=list)
    _silence_ms: float = field(init=False, default=0.0)
    _speech_ms: float = field(init=False, default=0.0)
    _active: bool = field(init=False, default=False)

    def __post_init__(self):
        self._preroll = deque(maxlen=max(1, int(self.pad_ms / FRAME_MS)))

    def _close(self) -> Optional[np.ndarray]:
        pcm = np.concatenate(self._frames) if self._frames else None
        # min_s gates on actual SPEECH duration — pre-roll and trailing
        # silence must not promote a click/breath into an utterance.
        too_short = self._speech_ms < self.min_s * 1000
        self._frames = []
        self._active = False
        self._silence_ms = 0.0
        self._speech_ms = 0.0
        if pcm is None or too_short:
            return None
        return pcm

    def feed(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Feed one FRAME_SAMPLES float32 frame; returns a finished utterance
        (float32, 16 kHz, includes pre-roll) or None."""
        speech = self.is_speech(frame) >= self.threshold
        if not self._active:
            if speech:
                self._active = True
                self._frames = list(self._preroll) + [frame]
                self._silence_ms = 0.0
                self._speech_ms = FRAME_MS
            else:
                self._preroll.append(frame)
            return None

        self._frames.append(frame)
        if speech:
            self._speech_ms += FRAME_MS
            self._silence_ms = 0.0
        else:
            self._silence_ms += FRAME_MS
        if self._silence_ms >= self.end_silence_ms:
            return self._close()
        if len(self._frames) * FRAME_MS >= self.max_s * 1000:
            return self._close()             # force-cut runaway utterance
        return None


class SileroVAD:
    """Lazy Silero VAD (torch) as an `is_speech(frame) -> prob` callable."""

    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is None:
            from silero_vad import load_silero_vad
            self._model = load_silero_vad()   # small CPU model

    def __call__(self, frame: np.ndarray) -> float:
        self._load()
        import torch
        with torch.no_grad():
            return float(self._model(torch.from_numpy(frame), SAMPLE_RATE).item())

    def reset(self):
        if self._model is not None:
            self._model.reset_states()


def mic_frames(device: int | None = None) -> Iterator[np.ndarray]:
    """Blocking generator of 512-sample float32 mono frames from the default
    (or given) input device. Lazy sounddevice import; run in a worker thread."""
    import queue

    import sounddevice as sd

    q: queue.Queue[np.ndarray] = queue.Queue(maxsize=256)

    def cb(indata, _frames, _time, status):
        if status:
            pass                             # over/underruns: drop, keep going
        try:
            q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass                             # consumer stalled: shed frames

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=FRAME_SAMPLES, device=device, callback=cb):
        while True:
            yield q.get()
