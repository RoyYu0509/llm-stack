"""CPU tests for the pretokenize benchmark harness (no GPU; needs a C++ compiler for the driver tests).

What these protect:
  * the line-wise Python reference really has `std::getline` + `simple_split` semantics (hand-computed cases);
  * the C++ driver (serial AND parallel) reproduces the reference bit-for-bit -- including inputs that stress
    the 4-way chunk boundaries (fewer lines than chunks, no trailing newline, CRLF, special tokens);
  * the equality check has teeth: a reference that keeps newlines, or ignores special tokens, is rejected;
  * the LOC counter handles comments / docstrings / blank lines and finds every real target.

"""
from __future__ import annotations

import shutil
import sys
from collections import Counter
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import bench_pretokenize as bp  # noqa: E402
import pretok_loc  # noqa: E402

SP = bp.SPECIAL_B
needs_cxx = pytest.mark.skipif(shutil.which("g++") is None and shutil.which("clang++") is None
                               and shutil.which("c++") is None, reason="no C++ compiler")


def single(b: bytes):
    return tuple(bytes([x]) for x in b)


# ------------------------------------------------------------------------------ reference semantics
def test_reference_hand_computed():
    data = b"ab cd\n  e\n"
    got = bp.ref_pretokenize_linewise(data)
    # line 1: "ab cd" -> "ab", " cd"; line 2: "  e" -> " ", " e"   (newline never seen)
    assert got == Counter({single(b"ab"): 1, single(b" cd"): 1, single(b" "): 1, single(b" e"): 1})


def test_reference_special_token_is_one_unit_and_splits_the_line():
    data = b"x y<|endoftext|>z\n"
    got = bp.ref_pretokenize_linewise(data)
    assert got == Counter({single(b"x"): 1, single(b" y"): 1, (SP,): 1, single(b"z"): 1})


def test_reference_no_trailing_newline_and_trailing_empty_piece():
    assert bp.ref_pretokenize_linewise(b"a\nb") == bp.ref_pretokenize_linewise(b"a\nb\n")
    assert bp.ref_pretokenize_linewise(b"") == Counter()


def test_reference_carriage_return_is_whitespace_not_a_line_break():
    got = bp.ref_pretokenize_linewise(b"a\r\nb")
    assert got == Counter({single(b"a"): 1, single(b"\r"): 1, single(b"b"): 1})


# ------------------------------------------------------------------------------ C++ driver == reference
@pytest.fixture(scope="module")
def driver(tmp_path_factory):
    d = tmp_path_factory.mktemp("drv")
    exe, _ = bp.build_driver(d, compile_reps=1)
    return exe


CASES = {
    "plain": b"the cat sat\non the mat\nthe end\n",
    "specials": b"one two<|endoftext|>three four\n<|endoftext|>\nfive<|endoftext|><|endoftext|>six\n",
    "no_trailing_newline": b"alpha beta\ngamma delta",
    "crlf": b"alpha  beta\r\ngamma\tdelta\r\n",
    "fewer_lines_than_chunks": b"only two\nlines here\n",
    "single_line": b"just one line with several words and  double  spaces",
    "utf8": "naïve café 你好 世界\nsecond 行\n".encode(),
    "empty_lines": b"a\n\n\nb\n\n",
}


@needs_cxx
@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("mode", ["serial", "parallel"])
def test_cxx_driver_equals_reference(driver, tmp_path, name, mode):
    f = tmp_path / "c.txt"
    f.write_bytes(CASES[name])
    ref = bp.pretok_stats(bp.ref_pretokenize_linewise(CASES[name]))
    r = bp.run_cxx(driver, f, mode)
    got = dict(unique=r["unique"], total=r["total"], checksum=r["checksum"])
    assert got == ref, f"{mode} on '{name}': C++ {got} != reference {ref}"


@needs_cxx
def test_cxx_parallel_equals_serial_on_a_larger_synthetic_file(driver, tmp_path):
    import random

    rnd = random.Random(0)
    words = ["the", "of", "and", "token", "cache", "attention", "<|endoftext|>", "x", "naïve"]
    lines = [" ".join(rnd.choice(words) for _ in range(rnd.randint(0, 40))) for _ in range(5000)]
    f = tmp_path / "big.txt"
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    a, b = bp.run_cxx(driver, f, "serial"), bp.run_cxx(driver, f, "parallel")
    assert (a["unique"], a["total"], a["checksum"]) == (b["unique"], b["total"], b["checksum"])
    ref = bp.pretok_stats(bp.ref_pretokenize_linewise(f.read_bytes()))
    assert (a["unique"], a["total"], a["checksum"]) == (ref["unique"], ref["total"], ref["checksum"])


# ------------------------------------------------------------------------------ the check has teeth
@needs_cxx
def test_equality_check_rejects_a_reference_that_keeps_newlines(driver, tmp_path):
    data = CASES["plain"]
    f = tmp_path / "c.txt"
    f.write_bytes(data)
    r = bp.run_cxx(driver, f, "serial")
    wrong = bp.pretok_stats(bp.ref_pretokenize_linewise(data.replace(b"\n", b" ")))  # newline -> a visible space
    assert (r["unique"], r["total"], r["checksum"]) != (wrong["unique"], wrong["total"], wrong["checksum"])


@needs_cxx
def test_equality_check_rejects_a_reference_that_ignores_special_tokens(driver, tmp_path):
    data = CASES["specials"]
    f = tmp_path / "c.txt"
    f.write_bytes(data)
    r = bp.run_cxx(driver, f, "serial")
    wrong = bp.pretok_stats(bp.ref_pretokenize_linewise(data, specials=()))
    assert r["checksum"] != wrong["checksum"]


def test_checksum_depends_on_counts_and_bytes():
    a = Counter({single(b"ab"): 2})
    assert bp.pretok_stats(a)["checksum"] != bp.pretok_stats(Counter({single(b"ab"): 3}))["checksum"]
    assert bp.pretok_stats(a)["checksum"] != bp.pretok_stats(Counter({single(b"ba"): 2}))["checksum"]
    assert bp.pretok_stats(a) == bp.pretok_stats(Counter({single(b"ab"): 2}))


# ------------------------------------------------------------------------------ LOC counter
def test_count_py_ignores_docstring_comments_and_blank(tmp_path):
    p = tmp_path / "m.py"
    p.write_text('def f(x):\n    """doc\n    more doc"""\n    # a comment\n\n    y = x + 1  # trailing comment kept\n    return y\n')
    r = pretok_loc.count_py(p, "f")
    assert (r["physical"], r["code"]) == (7, 3)  # def, y=..., return


def test_count_cxx_ignores_comments_strings_with_braces_and_blank(tmp_path):
    p = tmp_path / "m.cpp"
    p.write_text('// header\nint other() { return 1; }\n\n/* block\n   comment { */\nvoid BPETokenizer::foo(int a) {\n'
                 '    // note }\n    const char* s = "}{";\n\n    if (a) {\n        a++;\n    }\n}\nint after() { return 2; }\n')
    r = pretok_loc.count_cxx(p, "foo", r"void\s+BPETokenizer::foo\s*\(")
    assert (r["first"], r["last"]) == (6, 13)
    assert r["code"] == 6  # signature, const char*, if, a++, }, }


def test_real_targets_are_all_found_and_sane():
    r = pretok_loc.pretok_loc()
    for lang in ("python", "cxx"):
        assert len(r[lang]["functions"]) >= 4
        for f in r[lang]["functions"]:
            assert 0 < f["code"] <= f["physical"], f
