"""Engine contract. THIS SCHEMA IS FROZEN — it is GemmaApollo's training
target. Additive changes only, with explicit sign-off (see CLAUDE.md)."""
from __future__ import annotations

from typing import Literal, Optional, Protocol, Union
from pydantic import BaseModel, Field


class AppendMath(BaseModel):
    action: Literal["append_math"] = "append_math"
    latex: str


class Replace(BaseModel):
    action: Literal["replace"] = "replace"
    target_id: str
    latex: str


class Insert(BaseModel):
    action: Literal["insert"] = "insert"
    after_id: str
    latex: str


class Delete(BaseModel):
    action: Literal["delete"] = "delete"
    target_id: str


class SetLabel(BaseModel):
    action: Literal["set_label"] = "set_label"
    target_id: str
    text: str


class TextReply(BaseModel):
    """Engine answered in prose (math question, out of scope, etc.)."""
    action: Literal["text_reply"] = "text_reply"
    text: str


class Clarify(BaseModel):
    """Ambiguous reference — engine asks instead of guessing."""
    action: Literal["clarify"] = "clarify"
    question: str
    candidates: list[str] = Field(default_factory=list)  # candidate target ids


Action = Union[AppendMath, Replace, Insert, Delete, SetLabel, TextReply, Clarify]


class EngineResult(BaseModel):
    action: Action
    transcript: Optional[str] = None      # ASR transcript when available (S2L path)
    engine: str                            # "s2l" | "gemma" | "mock"
    latency_ms: dict[str, float] = Field(default_factory=dict)  # per stage


class Engine(Protocol):
    """Both engines implement exactly this. audio is 16 kHz mono PCM float32
    (or a path to a wav); doc_context is DocState.render_context() output."""

    name: str

    def process(self, audio, doc_context: str) -> EngineResult: ...

    def process_text(self, transcript: str, doc_context: str) -> EngineResult:
        """Text-only path used by the UI's simulated-utterance box and by
        tests. Engines that are audio-native may route this through their
        text front-end or raise NotImplementedError."""
        ...
