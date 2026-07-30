"""Resource and data roots that work from a repo checkout AND inside the
PyInstaller-frozen desktop sidecar (stage 6).

- resource_root(): read-only bundled assets — frontend/, models/vad/,
  models/wakewords/. Frozen: the bundle's _internal dir (sys._MEIPASS);
  checkout: the repo root.
- data_root(): writable state — data/sessions, data/client_id. Frozen:
  a per-user app-data dir (an installed app must never write into its own
  resources); checkout: the CWD, preserving the existing dev layout.
  SCRIBE_DATA_DIR overrides both.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def resource_root() -> Path:
    if frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parents[2]


def data_root() -> Path:
    env = os.environ.get("SCRIBE_DATA_DIR")
    if env:
        return Path(env)
    if frozen():
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home())))
        return base / "GemmaApolloScribe"
    return Path(".")
