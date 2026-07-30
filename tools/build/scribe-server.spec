# -*- mode: python -*-
"""PyInstaller spec for the desktop sidecar: `scribe serve` app tier only.

Torch-free by construction: build from the stage-6 freeze venv (base +
audio extras — see build_sidecar.ps1). The s2l/torch imports live inside
engine branches that this bundle never resolves; they are excluded
explicitly as a belt-and-braces measure.

Bundled data lands under _internal/ == sys._MEIPASS, which is exactly what
scribe.paths.resource_root() returns when frozen — frontend/, models/vad/,
models/wakewords/ resolve without code changes.
"""
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

REPO = Path(SPECPATH).resolve().parents[1]

datas = [
    (str(REPO / "frontend"), "frontend"),
    (str(REPO / "models" / "vad"), "models/vad"),
    # openwakeword ships feature-extractor models as package data
    *collect_data_files("openwakeword"),
]
if (REPO / "models" / "wakewords").exists():
    datas.append((str(REPO / "models" / "wakewords"), "models/wakewords"))

a = Analysis(
    [str(REPO / "tools" / "build" / "scribe_server_entry.py")],
    datas=datas,
    hiddenimports=[
        # uvicorn/websockets pick protocol impls dynamically at runtime
        *collect_submodules("uvicorn"),
        *collect_submodules("websockets"),
        "sounddevice",
    ],
    excludes=["torch", "faster_whisper", "transformers", "ctranslate2",
              "tkinter", "matplotlib", "IPython"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, exclude_binaries=True, name="scribe-server",
          console=True)
coll = COLLECT(exe, a.binaries, a.datas, name="scribe-server")
