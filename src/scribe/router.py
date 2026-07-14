"""Transcript router: classify an ASR transcript as command vs dictation and
build the corresponding Action. Regex fast path first; LLM fallback for
commands the grammar misses (Phase 3). Pure python. Port + extend the
grammar in frontend/index.html routeUtterance(); keep the two in sync."""
from __future__ import annotations
import re
from .engine.base import Action, AppendMath, Replace, Insert, Delete, SetLabel

# App-layer intents. If these arrive as transcripts (spotter missed them),
# the server maps them to app intents — they are NOT engine Actions.
APP_INTENTS = {"commit", "undo", "scratch that", "scratch"}

_ID = r"(e\d+)"
_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(rf"^(?:delete|remove)\s+(?:line\s+)?{_ID}$", re.I), "delete"),
    (re.compile(rf"^label\s+{_ID}\s+(.+)$", re.I), "set_label"),
    (re.compile(rf"^(?:change|replace)\s+{_ID}\s+to\s+(.+)$", re.I), "replace"),
    (re.compile(rf"^insert\s+after\s+{_ID}\s+(.+)$", re.I), "insert"),
]

def route(transcript: str, to_latex) -> Action | str | None:
    """Returns an Action, an app-intent string, or None (empty).
    `to_latex` converts a spoken-math fragment to LaTeX (the post-correction
    model in production; a toy converter in tests/mock)."""
    s = transcript.strip()
    if not s:
        return None
    if s.lower() in APP_INTENTS:
        return s.lower()
    for pat, kind in _RULES:
        m = pat.match(s)
        if not m:
            continue
        if kind == "delete":
            return Delete(target_id=m.group(1).lower())
        if kind == "set_label":
            return SetLabel(target_id=m.group(1).lower(), text=m.group(2).strip())
        if kind == "replace":
            return Replace(target_id=m.group(1).lower(), latex=to_latex(m.group(2)))
        if kind == "insert":
            return Insert(after_id=m.group(1).lower(), latex=to_latex(m.group(2)))
    # TODO(Phase 1): ordinal targeting ("change line two to ...", "delete the
    # last line") — needs doc state; take an optional resolver argument.
    # TODO(Phase 3): LLM fallback for natural commands the grammar misses,
    # emitting one Action as strict JSON; validate with pydantic before use.
    return AppendMath(latex=to_latex(s))
