#!/usr/bin/env bash
# Train one spotter end to end. Run after setup.sh + data.sh:
#   wsl -d Ubuntu -- bash tools/wakewords/wsl/train.sh "scratch that"
#
# NOTE: the run intentionally "fails" at the very end — the final tflite
# conversion needs onnx_tf, which we don't install because the Windows
# runtime uses ONNX. The .onnx (+ .onnx.data sidecar) is saved just before
# that. Deploy both files into models/wakewords/.
#
# If a run crashes mid-augmentation, delete trained/<name>/<name>/*features*.npy
# before retrying, or --augment_clips will "skip" off the partial files.
set -e
export PATH="$HOME/.local/bin:$PATH"
cd ~/gemmapollo-ww
source venv/bin/activate
source ~/espeak/env

PHRASE="$1"
NAME=$(echo "$PHRASE" | tr ' ' '_')
mkdir -p trained   # train.py os.mkdir()s output_dir and needs the parent

python - "$PHRASE" <<'EOF'
import sys

import yaml

phrase = sys.argv[1]
name = phrase.replace(" ", "_")
config = yaml.load(open("openWakeWord/examples/custom_model.yml").read(), yaml.Loader)
config["target_phrase"] = [phrase]
config["model_name"] = name
config["n_samples"] = 4000
config["n_samples_val"] = 1000
config["steps"] = 25000
config["target_accuracy"] = 0.7
config["target_recall"] = 0.5
config["output_dir"] = f"./trained/{name}"
config["piper_sample_generator_path"] = "./piper-fork"
config["rir_paths"] = ["./mit_rirs"]
config["background_paths"] = ["./audioset_16k", "./fma"]
config["false_positive_validation_data_path"] = "validation_set_features.npy"
config["feature_data_files"] = {"ACAV100M_sample": "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"}
with open(f"{name}.yaml", "w") as f:
    yaml.dump(config, f)
print("config written:", f"{name}.yaml")
EOF

python openWakeWord/openwakeword/train.py --training_config "$NAME.yaml" --generate_clips
python openWakeWord/openwakeword/train.py --training_config "$NAME.yaml" --augment_clips
python openWakeWord/openwakeword/train.py --training_config "$NAME.yaml" --train_model \
  || true  # expected: dies at tflite conversion AFTER saving the onnx
ls -la "trained/$NAME/"*.onnx*
echo "TRAINED $NAME"
