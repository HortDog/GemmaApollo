# Wake-word spotter training (PLAN.md Phase 5)

Targets: three openWakeWord models — `commit`, `undo`, `scratch_that` — that
run continuously on the mic stream and must **never fire mid-equation**.

Runtime lives in `src/scribe/audio/wakewords.py`; the server auto-loads
`models/wakewords/{commit,undo,scratch_that}.onnx` when `--mic` is on.

## Procedure

1. **Positives — synthetic speech variations** (thousands per phrase) with
   [piper-sample-generator](https://github.com/rhasspy/piper-sample-generator):

   ```bash
   python tools/wakewords/generate_samples.py --phrase "commit" --n 4000
   python tools/wakewords/generate_samples.py --phrase "undo" --n 4000
   python tools/wakewords/generate_samples.py --phrase "scratch that" --n 4000
   ```

   piper-sample-generator is Linux-oriented (espeak-ng phonemization); if it
   fights you on Windows, run this step in WSL or Colab.

   ### WSL2 setup gotchas (learned the hard way, 2026-07; env lives in
   `~/gemmapollo-ww` on the Ubuntu distro)

   - Use a **python 3.10** venv (the notebook's pins predate 3.12).
   - `webrtcvad` needs a C compiler → install **`webrtcvad-wheels`** instead
     (same module, prebuilt).
   - `datasets==2.14.6` needs **`pyarrow==13`** and **`numpy<2`**.
   - `pronouncing` needs `pkg_resources` → **`setuptools<81`**.
   - torch/torchaudio must be the **same release pair** from the same index
     (mismatched wheels fail loading `_torchaudio.abi3.so`); latest
     `torch-audiomentations` (0.11 uses a removed torchaudio API).
   - No sudo needed for espeak-ng: `pip install espeak-phonemizer
     espeakng-loader`, then symlink `espeakng_loader`'s `libespeak-ng.so` as
     `libespeak-ng.so.1` on `LD_LIBRARY_PATH`.
   - train.py wants **dscripka's fork** of piper-sample-generator (top-level
     `generate_samples.py`) and its **v1** checkpoint
     `en-us-libritts-high.pt` (release v1.0.0), NOT rhasspy v2's
     `en_US-libritts_r-medium.pt`; patch its `torch.load(...)` to
     `weights_only=False` for torch ≥2.6.
   - `os.mkdir(output_dir)` in train.py: create the parent dir first.
   - AudioSet on HF (`agkphysics/AudioSet`) is parquet shards now — decode
     with pyarrow+soundfile; old `datasets` can't parse the repo metadata.
   - The ACAV features file is 17.28 GB (`Content-Length` check it — a
     collided/resumed download can silently corrupt it).

2. **Negatives — recorded math dictation.** The spotters listen while you
   dictate equations, so dictation audio is the adversarial negative set:

   ```bash
   python tools/wakewords/make_negatives.py   # exports data/sessions/**/*.wav
   ```

   plus the standard openWakeWord negative corpus (ACAV100M feature set,
   downloaded by their training notebook).

3. **Train** with openWakeWord's pipeline — easiest is their
   [automatic model training notebook](https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb)
   (Colab, ~1 h/model on a T4): point it at the positive/negative dirs,
   export ONNX, drop the file into `models/wakewords/`.

4. **Threshold tuning.** Serve with `--mic`, dictate for 10 minutes, then
   audit `data/sessions/<ts>/*.json` for `verdict: app_intent` records —
   every spotter fire is logged. Zero false fires is the acceptance bar;
   raise the per-run threshold (IntentSpotter, default 0.5) if needed.

## Smoke test without trained models

Any pretrained oww model can stand in to test the plumbing end to end:

```bash
uv run scribe serve --engine mock --mic --wakeword-model hey_jarvis_v0.1=commit
```

then say "hey jarvis" → the pending line commits.
