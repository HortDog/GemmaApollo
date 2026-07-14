from scribe.router import route
from scribe.engine.base import AppendMath, Replace, Delete, SetLabel, Insert

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
