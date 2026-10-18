"""C2 tests (CPU; needs a C++ compiler and the repo venv).

Three layers, each answering one question (see c2_merges_equal.py header):

  1. test_algorithms_agree_when_pretokenization_is_matched_and_newline_free
       The claim that CAN be true: same split rule + no newlines -> identical merges.
  2. test_newlines_make_cxx_diverge
       The C++ trainer reads line by line (std::getline) and drops '\\n'; Python keeps it.
       Documents *why* the unconditional claim "merges identical" is false.
  3. test_default_gpt2_regex_diverges
       Python's production regex differs from C++ simple_split.

Tests 2 and 3 also prove the comparison has teeth: if they ever pass trivially (identical),
the corpus is too easy and test 1 means nothing.
"""
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import c2_merges_equal as c2  # noqa: E402

pytest.importorskip("cs336_basics.bpe_tokenizer.tokenizer")
pytestmark = pytest.mark.skipif(
    not (shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")),
    reason="needs a C++ compiler",
)

VOCAB = 420  # 420 - 257 = 163 merges: past the first newline-related pairs

DOCS = [
    "Once upon a time there was a little girl named Lily. She loved to play in the garden.\n"
    "The quick brown fox jumps over the lazy dog, again and again!  Two  spaces\tand a tab.\n",
    "Lily's dog couldn't swim; they'll go home. 123 456 7890 -- (parentheses) and \"quotes\".\n"
    "- a markdown list item\n- another list item\n"
    "中文也要有:自写分词器,并行预切词。标点,全角!\n",
    "A second document with repeated words: hello hello hello world world. newline\n\n\nmore.\n",
] * 60


def _write(p: Path, newlines: str) -> Path:
    text = c2.SPECIAL.join(DOCS)
    if newlines == "space":
        text = text.replace("\r", " ").replace("\n", " ")
    p.write_text(text, encoding="utf-8")
    return p


@pytest.fixture(scope="module")
def cxx_exe(tmp_path_factory):
    return c2.build_cpp(tmp_path_factory.mktemp("cxx_bin"))


def _cxx(exe, corpus, tmp_path_factory):
    return c2.run_cpp(exe, corpus, VOCAB, tmp_path_factory.mktemp("cxx_run"))


def test_algorithms_agree_when_pretokenization_is_matched_and_newline_free(
        cxx_exe, tmp_path_factory):
    corpus = _write(tmp_path_factory.mktemp("c") / "nl_free.txt", "space")
    cxx = _cxx(cxx_exe, corpus, tmp_path_factory)
    py = c2.run_python(corpus, VOCAB, "simple")
    r = c2.compare(py, cxx)
    assert r["identical"], (
        f"merges diverge at #{r['first_diff_index']} with the SAME split rule and NO newlines: "
        f"python={r['first_diff_python']!r} c++={r['first_diff_cxx']!r}. The two merge algorithms "
        f"really disagree. Suspects: (1) tie-break among equal counts (Python max() over "
        f"(bytes,bytes) vs C++ pair comparison); (2) special-token splitting; "
        f"(3) simple_split vs c2.SIMPLE_SPLIT_PATTERN differ on some character class."
    )


def test_newlines_make_cxx_diverge(cxx_exe, tmp_path_factory):
    corpus = _write(tmp_path_factory.mktemp("c") / "with_nl.txt", "keep")
    cxx = _cxx(cxx_exe, corpus, tmp_path_factory)
    py = c2.run_python(corpus, VOCAB, "simple")
    r = c2.compare(py, cxx)
    assert not r["identical"], (
        "expected divergence: the C++ trainer reads with getline and drops '\\n', Python keeps "
        "it. If this is identical, the corpus does not exercise newlines or C++ was changed to "
        "stop discarding them (then update README K/C2)."
    )


def test_default_gpt2_regex_diverges(cxx_exe, tmp_path_factory):
    corpus = _write(tmp_path_factory.mktemp("c") / "nl_free2.txt", "space")
    cxx = _cxx(cxx_exe, corpus, tmp_path_factory)
    py = c2.run_python(corpus, VOCAB, "gpt2")
    r = c2.compare(py, cxx)
    assert not r["identical"], (
        "With Python's default GPT-2 regex the merges are expected to differ from the C++ "
        "simple_split even without newlines."
    )
