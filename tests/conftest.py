import pytest


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    """SessionLogger writes data/sessions under cwd; keep test artifacts out
    of the repo by running every test from a temp directory."""
    monkeypatch.chdir(tmp_path)
