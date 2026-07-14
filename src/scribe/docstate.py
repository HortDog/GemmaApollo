"""Document model. App-owned: stable IDs, labels, undo stack, transactions.
Pure python — no torch, testable everywhere. See PLAN.md Phase 1."""
from __future__ import annotations
import copy
import re
from dataclasses import dataclass, field
from .engine.base import Action

# ---------------------------------------------------------------------------
# Ordinal targeting: turn a spoken line reference into a 0-based line index.
# "line two" -> 1, "the last line" -> -1, "second line" -> 1, "line 2" -> 1.
# Pure — used by DocState.resolve() and by router.resolver_from_context().
# ---------------------------------------------------------------------------
_NUM_WORDS: dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11,
    "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
    "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19,
    "twentieth": 20,
}


def _word_to_int(w: str) -> int | str | None:
    """A single number token -> 1-based position, the sentinel 'last', or None."""
    w = w.strip().lower()
    if w == "last":
        return "last"
    if w.isdigit():
        return int(w)
    m = re.fullmatch(r"(\d+)(?:st|nd|rd|th)", w)  # 1st, 2nd, 3rd, 4th
    if m:
        return int(m.group(1))
    return _NUM_WORDS.get(w)


def parse_ordinal(phrase: str) -> int | None:
    """Map an ordinal line reference to a 0-based index (negative for 'last'),
    or None if the phrase is not an ordinal target.

    Handles: "last", "last line", "the last line", "line two", "line 2",
    "second line", bare "two"/"2nd"."""
    p = phrase.strip().lower()
    p = re.sub(r"^the\s+", "", p)
    if p in ("last", "last line"):
        return -1
    m = re.fullmatch(r"line\s+(.+)", p)          # "line two", "line 2"
    if m:
        n = _word_to_int(m.group(1))
        if n == "last":
            return -1
        if isinstance(n, int):
            return n - 1
        return None
    m = re.fullmatch(r"(.+?)\s+line", p)          # "second line", "two line"
    if m:
        n = _word_to_int(m.group(1))
        if n == "last":
            return -1
        if isinstance(n, int):
            return n - 1
        return None
    n = _word_to_int(p)                           # bare "two", "2nd"
    if isinstance(n, int):
        return n - 1
    return None


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

    def resolve(self, phrase: str) -> str:
        """Ordinal reference -> line id. "line two" -> lines[1].id,
        "the last line" -> lines[-1].id. Raises KeyError if the phrase is not
        an ordinal target or resolves out of range (caller -> error/clarify)."""
        idx = parse_ordinal(phrase)
        if idx is None:
            raise KeyError(f"not an ordinal target: {phrase!r}")
        try:
            return self.lines[idx].id
        except IndexError:
            raise KeyError(f"ordinal out of range: {phrase!r}")

    def apply(self, a: Action) -> str | None:
        """Apply an engine Action. Returns the new line id for append/insert.
        Raises KeyError on unknown targets (caller turns that into a Clarify
        or an error frame — never mutate on a bad target).

        Targets are validated BEFORE snapshotting, so a bad target leaves the
        undo history untouched (see PLAN.md Phase 1 accept criteria)."""
        kind = a.action
        if kind in ("text_reply", "clarify"):
            return None  # no document mutation

        # Resolve/validate any target first — must not dirty the undo stack.
        idx = None
        if kind == "replace":
            idx = self._find(a.target_id)
        elif kind == "insert":
            idx = self._find(a.after_id)
        elif kind == "delete":
            idx = self._find(a.target_id)
        elif kind == "set_label":
            idx = self._find(a.target_id)

        self._snapshot()
        if kind == "append_math":
            lid = f"e{self._next}"; self._next += 1
            self.lines.append(Line(lid, a.latex)); return lid
        if kind == "replace":
            self.lines[idx].latex = a.latex; return None
        if kind == "insert":
            lid = f"e{self._next}"; self._next += 1
            self.lines.insert(idx + 1, Line(lid, a.latex)); return lid
        if kind == "delete":
            self.lines.pop(idx); return None
        if kind == "set_label":
            self.lines[idx].label = a.text; return None
        # Nothing mutated; drop the snapshot we just took.
        self._undo.pop()
        raise ValueError(f"unhandled action: {kind}")

    def undo(self) -> bool:
        if not self._undo:
            return False
        self.lines = self._undo.pop()
        return True
