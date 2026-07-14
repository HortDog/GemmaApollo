from scribe.docstate import DocState
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
    with pytest.raises(KeyError):
        d.apply(Delete(target_id="e99"))
    assert d.render_context() == before  # snapshot popped? see note
    # NOTE for implementer: current code snapshots before the failed find —
    # decide: either look up target before snapshotting, or pop the snapshot
    # on failure so undo history stays clean. Fix and keep this test green.

def test_undo_restores():
    d = DocState()
    d.apply(AppendMath(latex="x"))
    d.apply(Delete(target_id="e1"))
    assert d.undo() is True
    assert d.lines and d.lines[0].latex == "x"
