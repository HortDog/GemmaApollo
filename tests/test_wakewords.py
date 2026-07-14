"""IntentSpotter logic — pure python, fake scorer, no openWakeWord models."""
import numpy as np

from scribe.audio.wakewords import CHUNK_SAMPLES, IntentSpotter

FRAME = np.zeros(512, dtype=np.float32)


def feed_n(spotter, n_frames):
    """Feed n mic frames; return list of fired intents."""
    fires = []
    for _ in range(n_frames):
        hit = spotter.feed(FRAME)
        if hit:
            fires.append(hit)
    return fires


def frames_per_chunk():
    # 1280-sample chunks out of 512-sample frames -> fires possible every
    # 2.5 frames; use 3 to guarantee at least one chunk per group
    return 3


def test_quiet_never_fires():
    s = IntentSpotter(score=lambda c: {"commit": 0.01, "undo": 0.02})
    assert feed_n(s, 2000) == []          # ~64 s of quiet


def test_fires_on_threshold():
    scores = iter([{"commit": 0.9}] + [{"commit": 0.0}] * 100)
    s = IntentSpotter(score=lambda c: next(scores))
    fires = feed_n(s, 30)
    assert fires == ["commit"]


def test_refractory_suppresses_repeat_fires():
    # constant high score — a single spoken word spans many chunks
    s = IntentSpotter(score=lambda c: {"undo": 0.99}, refractory_s=2.0)
    # 100 frames = 100*512/16000 = 3.2 s of audio -> exactly 2 fires
    fires = feed_n(s, 100)
    assert fires == ["undo", "undo"]


def test_highest_intent_wins():
    s = IntentSpotter(score=lambda c: {"commit": 0.6, "scratch": 0.8})
    fires = feed_n(s, frames_per_chunk())
    assert fires == ["scratch"]


def test_buffering_across_frames():
    seen_lens = []
    def scorer(chunk):
        seen_lens.append(len(chunk))
        return {"commit": 0.0}
    s = IntentSpotter(score=scorer)
    feed_n(s, 10)  # 5120 samples -> 4 chunks
    assert seen_lens == [CHUNK_SAMPLES] * 4
    assert all(l == CHUNK_SAMPLES for l in seen_lens)
