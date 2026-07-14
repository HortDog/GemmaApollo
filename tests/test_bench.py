"""Bench harness stays runnable with zero models (mock engine, text path)."""
from scribe.bench import FIXTURES, load_labels, run_bench


def test_labels_match_wavs():
    labels = load_labels()
    assert len(labels) >= 4
    for lab in labels:
        assert (FIXTURES / lab["wav"]).exists(), lab["wav"]
        assert lab["spoken"] and lab["latex"]


def test_mock_bench_runs():
    s = run_bench("mock")
    assert s["n"] == len(load_labels())
    assert s["mean_cer"] is not None
    # the toy converter nails at least the simplest clip
    by_wav = {r["wav"]: r for r in s["rows"]}
    assert by_wav["emc2_sapi.wav"]["cer"] == 0.0
