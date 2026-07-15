"""Engine factory. Imports are lazy per engine so the app host never pulls
torch when it runs mock or remote (CLAUDE.md: model code stays behind the
Engine interface)."""
from __future__ import annotations


def make_engine(name: str, **kwargs):
    if name == "mock":
        from .mock_engine import MockEngine
        return MockEngine(**kwargs)
    if name == "s2l":
        from .s2l_engine import S2LEngine
        return S2LEngine(**kwargs)
    if name == "gemma":
        from .gemma_engine import GemmaEngine
        return GemmaEngine(**kwargs)
    if name == "remote":
        from .remote_engine import RemoteEngine
        return RemoteEngine(**kwargs)
    raise ValueError(name)
