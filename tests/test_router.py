from scribe.router import route, resolver_from_context
from scribe.engine.base import AppendMath, Replace, Delete, SetLabel, Insert
import pytest

latex = lambda s: f"<{s.strip()}>"  # identity-ish converter for tests

def test_dictation_appends():
    a = route("e equals m c squared", latex)
    assert isinstance(a, AppendMath)

def test_commands():
    assert isinstance(route("change e2 to x squared", latex), Replace)
    assert isinstance(route("delete e1", latex), Delete)
    assert isinstance(route("remove line e3", latex), Delete)
    assert isinstance(route("label e1 kinetic energy", latex), SetLabel)
    assert isinstance(route("insert after e1 f equals m a", latex), Insert)

def test_app_intents_pass_through():
    assert route("commit", latex) == "commit"
    assert route("scratch that", latex) == "scratch that"

def test_empty():
    assert route("   ", latex) is None

def test_ordinal_targeting_resolves():
    ctx = "[e1] a  [e2] b  [e3] c"
    resolve = resolver_from_context(ctx)
    a = route("change line two to x squared", latex, resolve=resolve)
    assert isinstance(a, Replace) and a.target_id == "e2"
    a = route("delete the last line", latex, resolve=resolve)
    assert isinstance(a, Delete) and a.target_id == "e3"
    a = route("insert after first line f equals m a", latex, resolve=resolve)
    assert isinstance(a, Insert) and a.after_id == "e1"
    a = route("label second line kinetic energy", latex, resolve=resolve)
    assert isinstance(a, SetLabel) and a.target_id == "e2" and a.text == "kinetic energy"

def test_explicit_id_targeting_still_works_with_resolver():
    resolve = resolver_from_context("[e1] a  [e2] b")
    a = route("change e2 to y", latex, resolve=resolve)
    assert isinstance(a, Replace) and a.target_id == "e2"
    a = route("delete e1", latex, resolve=resolve)
    assert isinstance(a, Delete) and a.target_id == "e1"

def test_ordinal_out_of_range_raises():
    resolve = resolver_from_context("[e1] a")
    with pytest.raises(KeyError):
        route("delete line five", latex, resolve=resolve)

def test_ordinal_without_resolver_raises():
    with pytest.raises(KeyError):
        route("delete the last line", latex)
