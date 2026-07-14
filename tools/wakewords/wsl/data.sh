#!/usr/bin/env bash
# Training data for the spotters: ~17.3 GB negative features + augmentation
# audio. Run after setup.sh:  wsl -d Ubuntu -- bash tools/wakewords/wsl/data.sh
set -e
export PATH="$HOME/.local/bin:$PATH"
cd ~/gemmapollo-ww
source venv/bin/activate

# Precomputed openWakeWord negative features (single wget — TWO CONCURRENT
# wget -c ON THE SAME FILE SILENTLY CORRUPT IT; verify size when done)
wget -cq https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy
wget -cq https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy
SIZE=$(stat -c%s openwakeword_features_ACAV100M_2000_hrs_16bit.npy)
[ "$SIZE" = "17280000128" ] || { echo "features file wrong size ($SIZE != 17280000128) — delete and re-download"; exit 1; }

# AudioSet moved to parquet shards; old `datasets` can't parse the repo, so
# fetch two shards directly and decode with pyarrow+soundfile
mkdir -p audioset
for shard in 00 01; do
  [ -s audioset/$shard.parquet ] || wget -c -nv -O audioset/$shard.parquet \
    "https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/bal_train/$shard.parquet"
done

python - <<'EOF'
import io
import os
from pathlib import Path

import datasets
import numpy as np
import pyarrow.parquet as pq
import scipy.io.wavfile
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm

# MIT room impulse responses
out = "./mit_rirs"
if not (os.path.exists(out) and len(os.listdir(out)) >= 270):
    os.makedirs(out, exist_ok=True)
    ds = datasets.load_dataset("davidscripka/MIT_environmental_impulse_responses",
                               split="train", streaming=True)
    for row in tqdm(ds, desc="rirs"):
        name = row['audio']['path'].split('/')[-1]
        scipy.io.wavfile.write(os.path.join(out, name), 16000,
                               (row['audio']['array']*32767).astype(np.int16))

# AudioSet parquet -> 16 kHz wavs
out = "./audioset_16k"
if not (os.path.exists(out) and len(os.listdir(out)) > 900):
    os.makedirs(out, exist_ok=True)
    n = 0
    for shard in ("audioset/00.parquet", "audioset/01.parquet"):
        for rec in tqdm(pq.read_table(shard, columns=["audio"]).column("audio"), desc=shard):
            rec = rec.as_py()
            data, sr = sf.read(io.BytesIO(rec["bytes"]), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            if sr != 16000:
                data = resample_poly(data, 16000, sr)
            name = (rec.get("path") or f"clip{n}").split("/")[-1].rsplit(".", 1)[0] + ".wav"
            scipy.io.wavfile.write(os.path.join(out, name), 16000,
                                   (np.clip(data, -1, 1)*32767).astype(np.int16))
            n += 1

# 1 hour of FMA music
out = "./fma"
if not (os.path.exists(out) and len(os.listdir(out)) >= 100):
    os.makedirs(out, exist_ok=True)
    ds = datasets.load_dataset("rudraml/fma", name="small", split="train", streaming=True)
    ds = iter(ds.cast_column("audio", datasets.Audio(sampling_rate=16000)))
    for _ in tqdm(range(120), desc="fma"):
        row = next(ds)
        name = row['audio']['path'].split('/')[-1].replace(".mp3", ".wav")
        scipy.io.wavfile.write(os.path.join(out, name), 16000,
                               (row['audio']['array']*32767).astype(np.int16))
EOF

echo "counts: rirs=$(ls mit_rirs | wc -l) audioset=$(ls audioset_16k | wc -l) fma=$(ls fma | wc -l)"
echo "DATA DONE"
