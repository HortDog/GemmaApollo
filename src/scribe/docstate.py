"""Document model. App-owned: stable IDs, labels, undo stack, transactions.
Pure python — no torch, testable everywhere. See PLAN.md Phase 1."""
from __future__ import annotations
import copy
from dataclasses import dataclass, field
from .engine.base import Action

@dataclass
class Line:
    id: str
    latex: str
    label: str | None = None

@dataclass
class DocState:
    lines: list[Line] = field(default_factory=list)
    _next: int = 1
    _undo: list[list[Line]] = field(default_factory=list)
    UNDO_DEPTH = 50

    def render_context(self) -> str:
        return "  ".join(f"[{l.id}] {l.latex}" for l in self.lines)

    def _snapshot(self):
        self._undo.append(copy.deepcopy(self.lines))
        if len(self._undo) > self.UNDO_DEPTH:
            self._undo.pop(0)

    def _find(self, lid: str) -> int:
        for i, l in enumerate(self.lines):
            if l.id == lid:
                return i
        raise KeyError(f"unknown target id: {lid}")

    def apply(self, a: Action) -> str | None:
        """Apply an engine Action. Returns the new line id for append/insert.
        Raises KeyError on unknown targets (caller turns that into a Clarify
        or an error frame — never mutate on a bad target)."""
        kind = a.action
        if kind in ("text_reply", "clarify"):
            return None  # no document mutation
        self._snapshot()
        if kind == "append_math":
            lid = f"e{self._next}"; self._next += 1
            self.lines.append(Line(lid, a.latex)); return lid
        if kind == "replace":
            self.lines[self._find(a.target_id)].latex = a.latex; return None
        if kind == "insert":
            i = self._find(a.after_id)
            lid = f"e{self._next}"; self._next += 1
            self.lines.insert(i + 1, Line(lid, a.latex)); return lid
        if kind == "delete":
            self.lines.pop(self._find(a.target_id)); return None
        if kind == "set_label":
            self.lines[self._find(a.target_id)].label = a.text; return None
        raise ValueError(f"unhandled action: {kind}")

    def undo(self) -> bool:
        if not self._undo:
            return False
        self.lines = self._undo.pop()
        return True

    # TODO(Phase 1): ordinal resolution helpers for the router —
    # resolve("line two") -> lines[1].id, resolve("the last line") -> lines[-1].id
