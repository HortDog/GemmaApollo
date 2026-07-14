# Wake-word spotter training — runbook & results

How the `commit` / `undo` / `scratch that` openWakeWord spotters were trained
(2026-07-15), how to reproduce or retrain them, and every landmine on the way.
Scripts live in [`tools/wakewords/wsl/`](../tools/wakewords/wsl/); runtime is
[`src/scribe/audio/wakewords.py`](../src/scribe/audio/wakewords.py).

## Results (deployed in `models/wakewords/`)

| model | validation FP/hour (target 0.2) | verification |
|---|---|---|
| `commit.onnx` | **0.18** ✅ | 2/2 SAPI voices → exactly 1 fire, correct intent |
| `scratch_that.onnx` | **0.0** ✅ | 2/2 SAPI voices → exactly 1 fire, correct intent |
| `undo.onnx` | **3.0** ⚠ | 2/2 SAPI voices → exactly 1 fire, correct intent |

All three: **0 fires** over 60 s of silence and over all four math-dictation
fixtures (the never-fire-mid-equation rule). Each `.onnx` has a mandatory
`.onnx.data` sidecar (torch ≥2.9's exporter externalizes weights — deploy both).

⚠ `undo` is a short word and missed the FP target on the synthetic validation
set. Every live fire is logged by the server as `verdict: app_intent` in
`data/sessions/` — audit after real dictation sessions. If it misfires:
raise its threshold (per-run in `IntentSpotter`, default 0.5) or retrain with
`max_negative_weight` increased (e.g. 1500 → 5000) in the config section of
`train.sh`, plus your own dictation audio as extra negatives
(`tools/wakewords/make_negatives.py`).

## Training recipe

Per phrase: 4000 train + 1000 val positives synthesized with piper TTS
(libritts-high, 904 speakers), augmented with MIT room impulse responses +
AudioSet + FMA music; negatives are openWakeWord's precomputed ACAV100M
features (~2000 h, 17.3 GB); 25k steps of the standard 32-unit DNN on the
RTX 4060 Ti. Wall time ≈ 15 min per model once data is downloaded.

## Reproducing (WSL2 Ubuntu, no sudo needed)

```bash
wsl -d Ubuntu -- bash tools/wakewords/wsl/setup.sh    # env + all patches (~10 min)
wsl -d Ubuntu -- bash tools/wakewords/wsl/data.sh     # ~20 GB of downloads
wsl -d Ubuntu -- bash tools/wakewords/wsl/train.sh "commit"
wsl -d Ubuntu -- bash tools/wakewords/wsl/train.sh "undo"
wsl -d Ubuntu -- bash tools/wakewords/wsl/train.sh "scratch that"
```

Then copy `~/gemmapollo-ww/trained/<name>/<name>.onnx` **and**
`<name>.onnx.data` (WSL path `\\wsl$\Ubuntu\home\<user>\...`) into
`models/wakewords/`. `scribe serve --mic` auto-loads whatever is there
(mapping in `DEFAULT_MODELS`, `src/scribe/audio/wakewords.py`).

Long runs: launch via `Start-Process -WindowStyle Hidden wsl ...` from
PowerShell and log to a file — the WSL VM shuts down when the last `wsl.exe`
exits, so `nohup` inside WSL does NOT survive, and harness background tasks
get killed at their timeout.

## Landmines (all pre-patched by setup.sh)

The 2023-era training stack meets 2026 dependencies:

| symptom | cause | fix |
|---|---|---|
| `webrtcvad` build: `cc not found` | no compiler in stock WSL | `webrtcvad-wheels` (same module, prebuilt) |
| `pyarrow has no PyExtensionType` | datasets 2.14.6 vs new pyarrow | pin `pyarrow==13` |
| `numpy.core.multiarray failed to import` | pyarrow 13 vs numpy 2 | pin `numpy<2` |
| `No module named pkg_resources` | setuptools ≥81 removed it | pin `setuptools<81` |
| `_torchaudio.abi3.so` won't load | torch/torchaudio from different builds | install both pinned from the same cu126 index |
| `torchaudio has no set_audio_backend` | torch-audiomentations 0.11 | upgrade torch-audiomentations |
| `TorchCodec is required` on `.load` / `no attribute 'info'` | torchaudio ≥2.10 delegates I/O to torchcodec (needs system ffmpeg) | soundfile shims patched into oww `data.py` + torch-audiomentations `io.py` |
| `espeak_phonemizer`: `libespeak-ng.so` missing | no sudo for apt; deb extraction pulls an endless audio-lib chain | `espeakng-loader` wheel + symlink as `libespeak-ng.so.1` on `LD_LIBRARY_PATH` |
| `en-us-libritts-high.pt` not found | dscripka fork defaults to the **v1** checkpoint | download from piper-sample-generator release v1.0.0 |
| `UnpicklingError: weights_only` | torch ≥2.6 default flip | `torch.load(..., weights_only=False)` (trusted rhasspy release) |
| `FileNotFoundError: trained/<name>` | train.py uses `os.mkdir` | pre-create the parent dir |
| augment stage "skips" then train crashes on missing `positive_features_test.npy` | a crashed run leaves partial feature npys and the skip-check passes | delete `trained/<name>/<name>/*features*.npy`, rerun |
| `No module named onnxscript` at export | torch 2.9 ONNX exporter | `pip install onnx onnxscript` |
| `No module named onnx_tf` at the very end | tflite conversion (unneeded — Windows runs ONNX) | ignore; the `.onnx` is already saved |
| AudioSet `bal_train09.tar` 404 | HF repo moved to parquet shards | fetch `data/bal_train/*.parquet`, decode with pyarrow+soundfile |
| features `.npy` larger than `Content-Length` | two concurrent `wget -c` on one file interleave garbage | single wget; verify size == 17280000128 |

## Threshold tuning & audit loop

1. Dictate for ~10 min with `uv run scribe serve --engine s2l --mic`.
2. Every spotter fire is a `verdict: app_intent` record in
   `data/sessions/<ts>/*.json` — count false ones.
3. Adjust per-intent behavior via `IntentSpotter(threshold=...)` (default 0.5,
   2 s refractory) in `server.py`'s `make_spotter`, or retrain (see above).
