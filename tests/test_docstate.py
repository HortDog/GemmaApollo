from scribe.docstate import DocState, parse_ordinal
from scribe.engine.base import AppendMath, Replace, Insert, Delete, SetLabel
import pytest

def test_append_ids_and_context():
    d = DocState()
    assert d.apply(AppendMath(latex="x^2")) == "e1"
    assert d.apply(AppendMath(latex="y")) == "e2"
    assert d.render_context() == "[e1] x^2  [e2] y"

def test_replace_insert_delete_label():
    d = DocState()
    d.apply(AppendMath(latex="a")); d.apply(AppendMath(latex="b"))
    d.apply(Replace(target_id="e1", latex="a+1"))
    assert d.lines[0].latex == "a+1"
    nid = d.apply(Insert(after_id="e1", latex="c"))
    assert [l.id for l in d.lines] == ["e1", nid, "e2"]
    d.apply(Delete(target_id="e2"))
    assert len(d.lines) == 2
    d.apply(SetLabel(target_id="e1", text="kinetic"))
    assert d.lines[0].label == "kinetic"

def test_unknown_target_raises_without_mutation():
    d = DocState()
    d.apply(AppendMath(latex="x"))
    before = d.render_context()
    undo_depth = len(d._undo)
    with pytest.raises(KeyError):
        d.apply(Delete(target_id="e99"))
    assert d.render_context() == before      # no mutation
    assert len(d._undo) == undo_depth        # snapshot not dirtied by failed find
    # A failed target must not leave a stale snapshot: undo still works cleanly.
    assert d.undo() is True                   # pops the append, not a stale entry
    assert d.render_context() == ""

def test_undo_restores():
    d = DocState()
    d.apply(AppendMath(latex="x"))
    d.apply(Delete(target_id="e1"))
    assert d.undo() is True
    assert d.lines and d.lines[0].latex == "x"

def test_undo_after_delete_restores_id_ordering():
    d = DocState()
    d.apply(AppendMath(latex="a"))   # e1
    d.apply(AppendMath(latex="b"))   # e2
    d.apply(AppendMath(latex="c"))   # e3
    d.apply(Delete(target_id="e2"))
    assert [l.id for l in d.lines] == ["e1", "e3"]
    assert d.undo() is True
    assert [l.id for l in d.lines] == ["e1", "e2", "e3"]
    assert [l.latex for l in d.lines] == ["a", "b", "c"]

def test_parse_ordinal():
    assert parse_ordinal("line two") == 1
    assert parse_ordinal("line 2") == 1
    assert parse_ordinal("second line") == 1
    assert parse_ordinal("the last line") == -1
    assert parse_ordinal("last") == -1
    assert parse_ordinal("2nd") == 1
    assert parse_ordinal("not a line ref") is None

def test_resolve_ordinal_to_id():
    d = DocState()
    d.apply(AppendMath(latex="a"))   # e1
    d.apply(AppendMath(latex="b"))   # e2
    d.apply(AppendMath(latex="c"))   # e3
    assert d.resolve("line two") == "e2"
    assert d.resolve("the last line") == "e3"
    assert d.resolve("first line") == "e1"
    with pytest.raises(KeyError):
        d.resolve("line nine")        # out of range
    with pytest.raises(KeyError):
        d.resolve("gibberish")        # not an ordinal target
