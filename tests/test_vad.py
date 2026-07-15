"""UtteranceChunker segmentation logic — pure python, fake VAD, no models."""
import numpy as np

from scribe.audio.vad import (FRAME_MS, FRAME_SAMPLES, SAMPLE_RATE,
                              UtteranceChunker, wav_bytes, wav_to_float32)


def frames(n):
    return [np.full(FRAME_SAMPLES, 0.1, dtype=np.float32) for _ in range(n)]


def run(chunker, probs):
    """Feed one frame per prob; collect emitted utterances."""
    out = []
    for p, f in zip(probs, frames(len(probs))):
        chunker.is_speech = lambda _f, _p=p: _p
        u = chunker.feed(f)
        if u is not None:
            out.append(u)
    return out


def n_frames(ms):
    return int(ms / FRAME_MS)


def test_silence_produces_nothing():
    c = UtteranceChunker(is_speech=lambda f: 0.0)
    # 60 s of silence — the Phase 4 idle acceptance criterion
    out = run(c, [0.0] * n_frames(60_000))
    assert out == []


def test_basic_utterance_with_padding():
    c = UtteranceChunker(is_speech=lambda f: 0.0, pad_ms=200, end_silence_ms=500)
    probs = [0.0] * 20 + [0.9] * n_frames(1000) + [0.0] * n_frames(600)
    out = run(c, probs)
    assert len(out) == 1
    # ~1 s speech + 200 ms preroll + 500 ms trailing silence before close
    dur_s = len(out[0]) / SAMPLE_RATE
    assert 1.5 <= dur_s <= 2.0


def test_short_blip_dropped():
    c = UtteranceChunker(is_speech=lambda f: 0.0, min_s=0.5)
    probs = [0.9] * n_frames(200) + [0.0] * n_frames(600)   # 200 ms click
    assert run(c, probs) == []


def test_max_length_force_cut():
    c = UtteranceChunker(is_speech=lambda f: 0.0, max_s=2.0)
    probs = [0.9] * n_frames(5000)          # 5 s of continuous speech
    out = run(c, probs)
    assert len(out) >= 2                     # cut into >= 2 utterances
    assert all(len(u) <= 2.0 * SAMPLE_RATE + FRAME_SAMPLES for u in out)


def test_brief_pause_does_not_split():
    c = UtteranceChunker(is_speech=lambda f: 0.0, end_silence_ms=500)
    probs = ([0.9] * n_frames(800) + [0.0] * n_frames(300)   # 300 ms pause
             + [0.9] * n_frames(800) + [0.0] * n_frames(600))
    out = run(c, probs)
    assert len(out) == 1                     # pause < end_silence_ms


def test_feed_with_precomputed_prob_skips_is_speech():
    def boom(_):
        raise AssertionError("is_speech must not run when prob is given")
    c = UtteranceChunker(is_speech=boom, end_silence_ms=500)
    probs = [0.9] * n_frames(1000) + [0.0] * n_frames(600)
    out = []
    for p, f in zip(probs, frames(len(probs))):
        u = c.feed(f, prob=p)
        if u is not None:
            out.append(u)
    assert len(out) == 1


def test_wav_bytes_roundtrip():
    import io
    import wave
    pcm = np.sin(np.linspace(0, 100, SAMPLE_RATE)).astype(np.float32)
    data = wav_bytes(pcm)
    with wave.open(io.BytesIO(data)) as w:
        assert w.getframerate() == SAMPLE_RATE
        assert w.getnchannels() == 1
        assert w.getnframes() == len(pcm)


def test_wav_to_float32_inverts_wav_bytes():
    pcm = np.sin(np.linspace(0, 100, SAMPLE_RATE)).astype(np.float32) * 0.8
    out = wav_to_float32(wav_bytes(pcm))
    assert out.dtype == np.float32 and len(out) == len(pcm)
    assert np.allclose(out, pcm, atol=1.0 / 32767)   # int16 quantization


def test_wav_to_float32_rejects_wrong_formats():
    import io
    import wave
    import pytest

    def make_wav(channels=1, width=2, rate=SAMPLE_RATE):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(channels)
            w.setsampwidth(width)
            w.setframerate(rate)
            w.writeframes(b"\x00" * (width * channels * 100))
        return buf.getvalue()

    with pytest.raises(ValueError, match="mono"):
        wav_to_float32(make_wav(channels=2))
    with pytest.raises(ValueError, match="16-bit"):
        wav_to_float32(make_wav(width=1))
    with pytest.raises(ValueError, match="Hz"):
        wav_to_float32(make_wav(rate=44100))
    with pytest.raises(ValueError, match="not a wav"):
        wav_to_float32(b"definitely not RIFF")
