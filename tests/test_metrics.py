from scribe.metrics import cer, levenshtein, normalize_latex


def test_levenshtein():
    assert levenshtein("", "") == 0
    assert levenshtein("abc", "abc") == 0
    assert levenshtein("abc", "axc") == 1
    assert levenshtein("abc", "") == 3
    assert levenshtein("kitten", "sitting") == 3


def test_cer():
    assert cer("E = mc^2", "E = mc^2") == 0.0
    assert cer("", "") == 0.0
    assert cer("x", "") == 1.0
    assert cer("E = mc^3", "E = mc^2") == 1 / 8
    # case-insensitive by default (paper normalizes case)
    assert cer("e = MC^2", "E = mc^2") == 0.0


def test_normalize_latex():
    assert normalize_latex("$E = mc^2$") == "E = mc^2"
    assert normalize_latex("$$\\frac{1}{2}$$") == "\\frac{1}{2}"
    # reference behavior: strips space BEFORE ^_{} only (kept identical for
    # CER comparability with the upstream eval)
    assert normalize_latex("x ^ 2  +  y") == "x^ 2 + y"
    assert normalize_latex("  \\frac {1} {2}  ") == "\\frac{1}{2}"
