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


def wav_to_float32(data: bytes, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Inverse of wav_bytes: 16-bit PCM mono wav file bytes -> float32 [-1,1].
    The engine contract is 16 kHz mono, so anything else is rejected rather
    than silently resampled."""
    try:
        w = wave.open(io.BytesIO(data), "rb")
    except (wave.Error, EOFError) as e:
        raise ValueError(f"not a wav file: {e}") from e
    with w:
        if w.getnchannels() != 1:
            raise ValueError(f"wav must be mono, got {w.getnchannels()} channels")
        if w.getsampwidth() != 2:
            raise ValueError(f"wav must be 16-bit PCM, got {w.getsampwidth() * 8}-bit")
        if w.getframerate() != sample_rate:
            raise ValueError(f"wav must be {sample_rate} Hz, got {w.getframerate()}")
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0


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


def import_sounddevice():
    """Lazy `import sounddevice`, working around Windows-on-ARM under an x64
    (emulated) Python: the win_amd64 wheel ships only the x64 PortAudio DLL,
    but sounddevice picks its DLL by physical machine (ARM64) and fails with
    error 0x7e. The x64 DLL is the right one for this process — steer the
    one-time import to it. Native ARM64 Python needs no workaround (its
    win_arm64 wheel bundles libportaudioarm64.dll)."""
    import sys
    if "sounddevice" in sys.modules:
        return sys.modules["sounddevice"]
    import platform
    import sysconfig
    if (sys.platform == "win32" and platform.machine() == "ARM64"
            and sysconfig.get_platform() == "win-amd64"):
        orig = platform.machine
        platform.machine = lambda: "AMD64"
        try:
            import sounddevice as sd
        finally:
            platform.machine = orig
        return sd
    import sounddevice as sd
    return sd


def list_input_devices() -> list[dict]:
    """Input-capable audio devices: [{index, name, default}]. Lazy import."""
    sd = import_sounddevice()

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

    sd = import_sounddevice()

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
