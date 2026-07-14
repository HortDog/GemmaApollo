"""Transcript router: classify an ASR transcript as command vs dictation and
build the corresponding Action. Regex fast path first; LLM fallback for
commands the grammar misses (Phase 3). Pure python. Port + extend the
grammar in frontend/index.html routeUtterance(); keep the two in sync."""
from __future__ import annotations
import re
from typing import Callable, Optional
from .engine.base import Action, AppendMath, Replace, Insert, Delete, SetLabel
from .docstate import parse_ordinal

# App-layer intents. If these arrive as transcripts (spotter missed them),
# the server maps them to app intents — they are NOT engine Actions.
APP_INTENTS = {"commit", "undo", "scratch that", "scratch"}

# A target is either an explicit id (e12) or an ordinal phrase resolved against
# doc state ("line two", "the last line", "second line").
_ID = r"e\d+"
_ORDINAL = r"(?:the\s+)?last(?:\s+line)?|line\s+\w+|\w+\s+line"
_TARGET = rf"({_ID}|{_ORDINAL})"

# A resolver maps an ordinal phrase to a line id (DocState.resolve). It may
# raise KeyError for out-of-range / unknown targets; callers surface that as an
# error/clarify frame rather than mutating the doc.
Resolver = Callable[[str], str]

_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(rf"^(?:delete|remove)\s+(?:line\s+)?{_TARGET}$", re.I), "delete"),
    (re.compile(rf"^label\s+{_TARGET}\s+(.+)$", re.I), "set_label"),
    (re.compile(rf"^(?:change|replace)\s+{_TARGET}\s+to\s+(.+)$", re.I), "replace"),
    (re.compile(rf"^insert\s+after\s+{_TARGET}\s+(.+)$", re.I), "insert"),
]


def resolver_from_context(doc_context: str) -> Resolver:
    """Build a Resolver from a render_context() string. Lets an engine that
    only holds the context string (no DocState) still resolve ordinals. Shares
    the ordinal grammar with DocState.resolve via parse_ordinal."""
    ids = re.findall(r"\[(e\d+)\]", doc_context)

    def resolve(phrase: str) -> str:
        idx = parse_ordinal(phrase)
        if idx is None:
            raise KeyError(f"not an ordinal target: {phrase!r}")
        try:
            return ids[idx]
        except IndexError:
            raise KeyError(f"ordinal out of range: {phrase!r}")

    return resolve


def _target_id(raw: str, resolve: Optional[Resolver]) -> str:
    """Turn a matched target group into a line id. Explicit e-ids resolve
    directly; ordinal phrases need the resolver. Raises KeyError when an
    ordinal is given but no resolver is available or it is out of range."""
    r = raw.strip()
    if re.fullmatch(_ID, r, re.I):
        return r.lower()
    if resolve is None:
        raise KeyError(f"ordinal target needs doc state: {raw!r}")
    return resolve(r)


def route(transcript: str, to_latex, resolve: Optional[Resolver] = None) -> Action | str | None:
    """Returns an Action, an app-intent string, or None (empty).
    `to_latex` converts a spoken-math fragment to LaTeX (the post-correction
    model in production; a toy converter in tests/mock). `resolve` maps ordinal
    line references to ids (pass DocState.resolve or resolver_from_context)."""
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
            return Delete(target_id=_target_id(m.group(1), resolve))
        if kind == "set_label":
            return SetLabel(target_id=_target_id(m.group(1), resolve), text=m.group(2).strip())
        if kind == "replace":
            return Replace(target_id=_target_id(m.group(1), resolve), latex=to_latex(m.group(2)))
        if kind == "insert":
            return Insert(after_id=_target_id(m.group(1), resolve), latex=to_latex(m.group(2)))
    # TODO(Phase 3): LLM fallback for natural commands the grammar misses,
    # emitting one Action as strict JSON; validate with pydantic before use.
    return AppendMath(latex=to_latex(s))
