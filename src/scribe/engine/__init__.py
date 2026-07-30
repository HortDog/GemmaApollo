"""Engine registry — the single place engines are constructed. Heavy
imports stay inside the branches so `--engine mock` never touches torch."""
from __future__ import annotations


def make_engine(name: str, model_server_url: str | None = None):
    if name == "mock":
        from .mock_engine import MockEngine
        return MockEngine()
    if name == "s2l":
        from .s2l_engine import S2LEngine
        return S2LEngine()
    if name == "gemma":
        from .gemma_engine import GemmaEngine
        return GemmaEngine()
    if name == "remote":
        from .remote_engine import RemoteEngine
        return RemoteEngine(model_server_url or "http://127.0.0.1:8018")
    raise ValueError(name)
