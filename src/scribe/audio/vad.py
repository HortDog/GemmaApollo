"""VAD chunker over a 16 kHz mono mic stream (PLAN.md Phase 4).

Layering (CLAUDE.md: pure logic stays torch-free):
- UtteranceChunker: pure segmentation logic driven by an injected
  `is_speech(frame) -> prob` callable. Testable with a fake VAD.
- SileroVAD: lazy onnxruntime wrapper providing that callable (32 ms /
  512-sample frames at 16 kHz — silero v5's required window).
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


def wav_to_float32(data: bytes) -> np.ndarray:
    """Complete wav file bytes -> float32 [-1,1] mono. Strict: the 16 kHz /
    mono / 16-bit invariant is VALIDATED, never resampled — a mismatched wav
    on the model-server hop is a caller bug, not something to paper over."""
    with wave.open(io.BytesIO(data)) as w:
        got = (w.getframerate(), w.getnchannels(), w.getsampwidth())
        if got != (SAMPLE_RATE, 1, 2):
            raise ValueError(f"expected {SAMPLE_RATE} Hz mono 16-bit wav, got "
                             f"{got[0]} Hz {got[1]}ch {got[2] * 8}-bit")
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return pcm.astype(np.float32) / 32768.0


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

    def reset(self):
        """Drop any partially-captured utterance and pre-roll (used when the
        mic is muted mid-utterance — nothing captured so far may leak out)."""
        self._frames = []
        self._active = False
        self._silence_ms = 0.0
        self._speech_ms = 0.0
        self._preroll.clear()

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

    def feed(self, frame: np.ndarray, prob: float | None = None) -> Optional[np.ndarray]:
        """Feed one FRAME_SAMPLES float32 frame; returns a finished utterance
        (float32, 16 kHz, includes pre-roll) or None. Pass `prob` when the
        caller already ran the VAD on this frame (mic tester shares the
        per-frame Silero score) — is_speech is skipped then."""
        speech = (self.is_speech(frame) if prob is None else prob) >= self.threshold
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


VAD_MODEL = "models/vad/silero_vad.onnx"     # vendored Silero v5 (MIT)
_VAD_CONTEXT = 64                            # v5 leading context @16 kHz


def _vad_model_path() -> "Path":
    """CWD-relative first (matches wakewords convention), then the repo /
    frozen-bundle root so any launch directory works."""
    from pathlib import Path

    from ..paths import resource_root
    p = Path(VAD_MODEL)
    if p.exists():
        return p
    return resource_root() / VAD_MODEL


class SileroVAD:
    """Lazy Silero VAD v5 as an `is_speech(frame) -> prob` callable.

    Runs the vendored onnx model directly on onnxruntime (CPU) instead of
    the silero-vad pip package, which hard-requires torch — the app tier
    (and the frozen desktop sidecar) must stay torch-free. Mirrors the
    package's OnnxWrapper: 512-sample window, 64 samples of context carried
    from the previous frame, LSTM state (2,1,128) threaded through.
    """

    def __init__(self, model_path=None):
        self._path = model_path
        self._sess = None
        self._state = None
        self._context = None

    def _load(self):
        if self._sess is None:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
            self._sess = ort.InferenceSession(
                str(self._path or _vad_model_path()), sess_options=opts,
                providers=["CPUExecutionProvider"])
            self.reset()

    def reset(self):
        if self._sess is not None:
            self._state = np.zeros((2, 1, 128), dtype=np.float32)
            self._context = np.zeros((1, _VAD_CONTEXT), dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        if frame.size != FRAME_SAMPLES:
            raise ValueError(f"expected {FRAME_SAMPLES}-sample frames, "
                             f"got {frame.size}")
        self._load()
        x = np.concatenate(
            [self._context, frame.reshape(1, -1).astype(np.float32)], axis=1)
        # graph order is (output, stateN), same unpacking as silero's wrapper
        out, self._state = self._sess.run(
            None, {"input": x, "state": self._state,
                   "sr": np.array(SAMPLE_RATE, dtype=np.int64)})
        self._context = x[:, -_VAD_CONTEXT:]
        return float(out[0, 0])


def list_input_devices() -> list[dict]:
    """Input-capable audio devices: [{index, name, default}]. Lazy import."""
    import sounddevice as sd

    default_idx = None
    try:
        default_idx = sd.default.device[0]   # (input, output) pair
    except Exception:
        pass
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) > 0:
            out.append({"index": i, "name": d["name"],
                        "default": i == default_idx})
    return out


def mic_frames(device: int | None = None, stop=None) -> Iterator[np.ndarray]:
    """Blocking generator of 512-sample float32 mono frames from the default
    (or given) input device. Lazy sounddevice import; run in a worker thread.
    `stop`: optional threading.Event — generator ends (and the stream closes)
    soon after it is set, enabling live device switching."""
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
        while stop is None or not stop.is_set():
            try:
                yield q.get(timeout=0.25)    # timeout: re-check stop regularly
            except queue.Empty:
                continue
