#!/usr/bin/env bash
# One-shot WSL2 (Ubuntu) environment for openWakeWord spotter training.
# Consolidates every fix from the 2026-07-15 run — see docs/wakeword-training.md.
# Run:  wsl -d Ubuntu -- bash tools/wakewords/wsl/setup.sh
set -e
export PATH="$HOME/.local/bin:$PATH"

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
mkdir -p ~/gemmapollo-ww && cd ~/gemmapollo-ww

[ -d openWakeWord ] || git clone -q --depth 1 https://github.com/dscripka/openWakeWord
# dscripka's FORK of piper-sample-generator: train.py imports its top-level
# generate_samples.py (rhasspy v2 restructured it away)
[ -d piper-fork ] || git clone -q --depth 1 https://github.com/dscripka/piper-sample-generator piper-fork

# python 3.10: the training-stack pins predate 3.12
uv python install 3.10 -q
[ -d venv ] || uv venv --python 3.10 venv
source venv/bin/activate

# torch + torchaudio MUST be the same release pair from the same index
uv pip install -q 'torch==2.9.1' 'torchaudio==2.9.1' --index-url https://download.pytorch.org/whl/cu126
uv pip install -q piper-phonemize webrtcvad-wheels   # webrtcvad needs cc; -wheels fork is prebuilt
uv pip install -q -e ./openWakeWord
uv pip install -q mutagen==1.47.0 torchinfo==1.8.0 torchmetrics==1.2.0 \
  speechbrain==0.5.14 audiomentations==0.33.0 torch-audiomentations \
  acoustics==0.2.6 pronouncing==0.2.0 datasets==2.14.6 deep-phonemizer==0.0.19 \
  soundfile pyyaml tqdm \
  'pyarrow==13.0.0' 'numpy<2' 'setuptools<81' \
  onnx onnxscript espeak-phonemizer espeakng-loader

# piper voice checkpoint (the fork's default is the v1 libritts-high model)
mkdir -p piper-fork/models
[ -s piper-fork/models/en-us-libritts-high.pt ] || wget -cq -P piper-fork/models \
  https://github.com/rhasspy/piper-sample-generator/releases/download/v1.0.0/en-us-libritts-high.pt

# openwakeword feature-extraction models
mkdir -p openWakeWord/openwakeword/resources/models
( cd openWakeWord/openwakeword/resources/models
  for f in embedding_model.onnx embedding_model.tflite melspectrogram.onnx melspectrogram.tflite; do
    [ -s "$f" ] || wget -q "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/$f"
  done )

# espeak-ng without sudo: espeakng-loader ships a self-contained lib; the
# rhasspy phonemizer dlopens "libespeak-ng.so.1", so symlink + env file
mkdir -p ~/espeak/lib
python - <<'EOF'
import os

import espeakng_loader
lib = espeakng_loader.get_library_path()
data = espeakng_loader.get_data_path()
libdir = os.path.expanduser("~/espeak/lib")
for name in ("libespeak-ng.so", "libespeak-ng.so.1"):
    dst = os.path.join(libdir, name)
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(lib, dst)
with open(os.path.expanduser("~/espeak/env"), "w") as f:
    f.write("export LD_LIBRARY_PATH=$HOME/espeak/lib:$LD_LIBRARY_PATH\n")
    f.write(f"export ESPEAK_DATA_PATH={data}\n")
EOF

# --- source patches (torch/torchaudio 2026 APIs vs 2023 training code) ---

# fork loads a fully-pickled VITS checkpoint; torch>=2.6 defaults weights_only=True
sed -i 's/torch.load(model_path)$/torch.load(model_path, weights_only=False)/' piper-fork/generate_samples.py

# torchaudio removed load()/info() backends -> soundfile shims
python - <<'EOF'
import os

def patch(path, marker, shim, repls):
    src = open(path).read()
    if marker not in src:
        lines = src.split("\n")
        last = max(i for i, l in enumerate(lines)
                   if l.startswith(("import ", "from ")) and "(" not in l)
        lines.insert(last + 1, shim)
        src = "\n".join(lines)
    for old, new in repls:
        src = src.replace(old, new)
    open(path, "w").write(src)
    print("patched", path)

oww_shim = '''
def _sf_load(path):
    """soundfile stand-in for torchaudio.load (torchaudio>=2.10 needs
    torchcodec/ffmpeg; the training clips are plain wavs)."""
    import soundfile as _sf
    import torch as _torch
    d, sr = _sf.read(path, dtype="float32", always_2d=True)
    return _torch.from_numpy(d.T), sr

'''
patch("openWakeWord/openwakeword/data.py", "_sf_load", oww_shim,
      [("torchaudio.load(", "_sf_load(")])

tam_shim = '''
class _SFInfo:
    def __init__(self, path):
        import soundfile as _sf
        i = _sf.info(str(path))
        self.num_frames = i.frames
        self.sample_rate = i.samplerate


def _sf_info(path):
    return _SFInfo(path)


def _sf_load(path, frame_offset=0, num_frames=-1):
    import soundfile as _sf
    import torch as _torch
    stop = None if num_frames in (-1, None) else frame_offset + num_frames
    d, sr = _sf.read(str(path), start=frame_offset, stop=stop,
                     dtype="float32", always_2d=True)
    return _torch.from_numpy(d.T), sr

'''
import glob
tam = glob.glob(os.path.expanduser(
    "~/gemmapollo-ww/venv/lib/python3.10/site-packages/torch_audiomentations/utils/io.py"))[0]
patch(tam, "_sf_info", tam_shim,
      [("torchaudio.info(", "_sf_info("), ("torchaudio.load(", "_sf_load(")])
EOF

python -c "import torch; print('SETUP DONE — torch', torch.__version__, 'cuda', torch.cuda.is_available())"
