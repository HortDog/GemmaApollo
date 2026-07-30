"""Stage-1 extraction: PCMReframer byte->frame math and FramePipeline fan-out
semantics (must mirror the historical server.capture() behavior). Pure
python/numpy — no torch, no models, no sockets."""
import numpy as np
import pytest

from scribe.audio.pipeline import FramePipeline, PCMReframer
from scribe.audio.vad import FRAME_SAMPLES, UtteranceChunker

FRAME_BYTES = FRAME_SAMPLES * 2


def s16le(samples) -> bytes:
    return np.asarray(samples, dtype="<i2").tobytes()


# ---------------------------------------------------------------- PCMReframer
def test_reframer_exact_frame():
    frames = PCMReframer().feed(s16le(np.zeros(FRAME_SAMPLES)))
    assert len(frames) == 1
    assert frames[0].shape == (FRAME_SAMPLES,)
    assert frames[0].dtype == np.float32


def test_reframer_accumulates_subframe_chunks():
    r = PCMReframer()
    data = s16le(np.arange(FRAME_SAMPLES) % 1000)
    assert r.feed(data[:100]) == []
    frames = r.feed(data[100:])
    assert len(frames) == 1
    np.testing.assert_array_equal(
        (frames[0] * 32768.0).astype("<i2"), np.arange(FRAME_SAMPLES) % 1000)


def test_reframer_split_mid_sample():
    r = PCMReframer()
    data = s16le(np.full(FRAME_SAMPLES, 12345))
    assert r.feed(data[:3]) == []          # odd byte count: splits a sample
    frames = r.feed(data[3:])
    assert len(frames) == 1
    assert np.allclose(frames[0], 12345 / 32768.0)


def test_reframer_big_chunk_yields_frames_and_keeps_leftover():
    r = PCMReframer()
    total = FRAME_SAMPLES * 2 + FRAME_SAMPLES // 2
    frames = r.feed(s16le(np.zeros(total)))
    assert len(frames) == 2
    frames = r.feed(s16le(np.zeros(FRAME_SAMPLES // 2)))   # completes frame 3
    assert len(frames) == 1


def test_reframer_scaling_extremes():
    r = PCMReframer()
    samples = np.zeros(FRAME_SAMPLES, dtype="<i2")
    samples[0], samples[1] = 32767, -32768
    f = r.feed(samples.tobytes())[0]
    assert f[0] == pytest.approx(32767 / 32768.0)
    assert f[1] == -1.0
    assert f[2] == 0.0


def test_reframer_reset_drops_leftover():
    r = PCMReframer()
    r.feed(b"\x00" * 10)
    r.reset()
    assert r.feed(b"\x00" * (FRAME_BYTES - 10)) == []      # leftover was dropped


# --------------------------------------------------------------- FramePipeline
class SpyChunker:
    def __init__(self, ret=None):
        self.fed, self.resets, self.ret = [], 0, ret

    def feed(self, frame, prob=None):
        self.fed.append(prob)
        return self.ret

    def reset(self):
        self.resets += 1


class SpySpotter:
    def __init__(self, hits=()):
        self.hits, self.n = list(hits), 0

    def feed(self, frame):
        self.n += 1
        return self.hits.pop(0) if self.hits else None


FRAME = np.zeros(FRAME_SAMPLES, dtype=np.float32)


def test_spotter_fires_while_muted_and_chunker_starved():
    intents, chunker = [], SpyChunker()
    p = FramePipeline(vad=lambda f: 0.9, chunker=chunker,
                      spotter=SpySpotter(hits=["commit"]),
                      is_muted=lambda: True, on_intent=intents.append)
    p.feed(FRAME)
    assert intents == ["commit"]
    assert chunker.fed == []                    # mute starves dictation only


def test_mute_edge_resets_chunker_once():
    muted = {"on": False}
    chunker = SpyChunker()
    p = FramePipeline(vad=lambda f: 0.9, chunker=chunker,
                      is_muted=lambda: muted["on"])
    p.feed(FRAME)
    assert len(chunker.fed) == 1 and chunker.resets == 0
    muted["on"] = True
    p.feed(FRAME)
    p.feed(FRAME)
    assert len(chunker.fed) == 1 and chunker.resets == 1   # reset on edge only
    muted["on"] = False
    p.feed(FRAME)
    assert len(chunker.fed) == 2 and chunker.resets == 1   # unmute: no reset


def test_utterance_and_prob_passthrough():
    utt = np.ones(1234, dtype=np.float32)
    got = []
    p = FramePipeline(vad=lambda f: 0.7, chunker=SpyChunker(ret=utt),
                      on_utterance=got.append)
    p.feed(FRAME)
    assert got and got[0] is utt
    # the chunker received the caller's VAD prob (is_speech skipped)
    assert p.chunker.fed == [0.7]


def test_levels_every_third_frame_only_while_testing():
    levels = []
    testing = {"on": True}
    p = FramePipeline(vad=lambda f: 0.25, chunker=SpyChunker(),
                      is_testing=lambda: testing["on"], on_level=levels.append)
    for _ in range(7):
        p.feed(FRAME)
    assert len(levels) == 3                     # frames 0, 3, 6
    assert levels[0] == {"rms": 0.0, "prob": 0.25}
    testing["on"] = False
    for _ in range(6):
        p.feed(FRAME)
    assert len(levels) == 3


def test_pipeline_reset_resets_chunker_and_cadence():
    levels = []
    p = FramePipeline(vad=lambda f: 0.0, chunker=SpyChunker(),
                      is_testing=lambda: True, on_level=levels.append)
    p.feed(FRAME)                               # level at frame 0
    p.reset()
    assert p.chunker.resets == 1
    p.feed(FRAME)                               # cadence restarts at frame 0
    assert len(levels) == 2


def test_integration_real_chunker_emits_utterance():
    """Speech frames then silence through a real UtteranceChunker: one
    utterance comes out, sized speech+pre-roll+trailing silence."""
    got = []
    probs = iter([1.0] * 30 + [0.0] * 30)       # ~0.96 s speech, then silence
    p = FramePipeline(vad=lambda f: next(probs),
                      chunker=UtteranceChunker(is_speech=lambda f: 0.0),
                      on_utterance=got.append)
    for _ in range(60):
        p.feed(FRAME)
    assert len(got) == 1
    # 30 speech frames + 500 ms end-silence (~16 frames @32 ms) concatenated
    assert got[0].size >= 30 * FRAME_SAMPLES
