"""C2 -- are the C++ BPE trainer's merges the same as the Python trainer's?

Background (see README "K / C2"): the resume template says "merges identical to the Python
implementation". Two things make that false *by default*:

  1. The C++ trainer pre-tokenizes with `simple_split` (split at whitespace, keep one leading
     whitespace char on the next word), the Python trainer with the GPT-2 regex
     (`'(?:[sdmt]|ll|ve|re)| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+`).
     Different pre-tokens -> different pair counts -> different merges.
  2. The C++ trainer reads the corpus with `std::getline`, so every '\n' is **discarded before
     pre-tokenization**; the Python trainer sees '\n' as part of the text and produces pre-tokens
     such as "\n-" or "\nThe". Any corpus with newlines therefore gives different pair counts,
     even when both sides use the same split rule (found by this script, 2026-10-15).
  3. The existing `tests/test_bpe_cpp_vs_python.py` does not compare them (its fixtures are
     missing, and its own header says the products "will diverge").

This script separates the questions:

  --python-pat simple   Make Python use the *same* pre-tokenization rule as C++. If the merge
                        algorithms really agree (same counts, same tie-break), merges must be
                        identical. This is the claim that can become true.
  --python-pat gpt2     Python's default rule (what production uses). Reports where they diverge.
  --newlines space      Replace '\r' / '\n' with ' ' in a temp copy of the corpus, so neither side
                        sees a newline. Together with --python-pat simple this isolates "do the two
                        merge algorithms agree?" from the two differences above.

Nothing in the repo is modified: the C++ binary is compiled into a temp dir, and the Python
pre-tokenizer pattern is overridden on the instance.

Usage:
  python c2_merges_equal.py --corpus corpus.txt --vocab-size 400 --python-pat simple
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CXX_DIR = REPO / "training" / "cs336-basics" / "cs336_basics" / "bpe_tokenizer_cpp"
SPECIAL = "<|endoftext|>"

# Same rule as C++ `simple_split`: every whitespace char starts a new token; a token is one
# optional leading whitespace char + a run of non-whitespace chars; a whitespace char followed by
# another whitespace char stands alone.
SIMPLE_SPLIT_PATTERN = r"[ \t\n\r]?[^ \t\n\r]+|[ \t\n\r]"


def build_cpp(out_dir: Path) -> Path:
    exe = out_dir / "bpe_train"
    srcs = [CXX_DIR / n for n in ("bpe_tokenizer.cpp", "pretokenizer.cpp", "parallel_helper.cpp", "main.cpp")]
    cxx = shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")
    if not cxx:
        raise RuntimeError("no C++ compiler found (g++/c++/clang++)")
    cmd = [cxx, "-std=c++20", "-O2", "-pthread", *map(str, srcs), "-I", str(CXX_DIR), "-o", str(exe)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return exe


def run_cpp(exe: Path, corpus: Path, vocab_size: int, work: Path) -> list[tuple[bytes, bytes]]:
    """`bpe_train` writes vocab.txt / merges.txt (hex pairs) into its cwd."""
    subprocess.run([str(exe), str(corpus), str(vocab_size)], cwd=work, check=True,
                   capture_output=True, text=True)
    merges = []
    for line in (work / "merges.txt").read_text().splitlines():
        a, b = line.split(" ")
        merges.append((bytes.fromhex(a), bytes.fromhex(b)))
    return merges


def run_python(corpus: Path, vocab_size: int, pat: str, procs: int = 2) -> list[tuple[bytes, bytes]]:
    import regex
    from cs336_basics.bpe_tokenizer.tokenizer import Tokenizer

    tok = Tokenizer(vocab=vocab_size, special_tokens=[SPECIAL])
    if pat == "simple":
        tok.PAT = regex.compile(SIMPLE_SPLIT_PATTERN)
    elif pat != "gpt2":
        raise ValueError(f"--python-pat must be simple|gpt2, got {pat}")
    _vocab, merges = tok.train(str(corpus), vocab_size, procs, SPECIAL.encode("utf-8"))
    return [(bytes(a), bytes(b)) for a, b in merges]


def compare(py: list, cxx: list) -> dict:
    n = min(len(py), len(cxx))
    first_diff = next((i for i in range(n) if py[i] != cxx[i]), None)
    return dict(
        n_python=len(py), n_cxx=len(cxx),
        identical=(first_diff is None and len(py) == len(cxx)),
        identical_prefix=n if first_diff is None else first_diff,
        first_diff_index=first_diff,
        first_diff_python=py[first_diff] if first_diff is not None else None,
        first_diff_cxx=cxx[first_diff] if first_diff is not None else None,
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, help="UTF-8 text; documents separated by <|endoftext|>")
    ap.add_argument("--vocab-size", type=int, default=400)
    ap.add_argument("--python-pat", choices=["simple", "gpt2"], default="simple")
    ap.add_argument("--newlines", choices=["keep", "space"], default="keep",
                    help="'space': replace \\r/\\n by ' ' in a temp copy of the corpus (see header)")
    ap.add_argument("--cxx-bin", default=None, help="prebuilt bpe_train; default: compile from source")
    ap.add_argument("--csv", default=None, help="append result rows to this CSV (schema.py)")
    args = ap.parse_args(argv)

    corpus = Path(args.corpus).resolve()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        if args.newlines == "space":
            text = corpus.read_text(encoding="utf-8").replace("\r", " ").replace("\n", " ")
            corpus = td / "corpus_no_newlines.txt"
            corpus.write_text(text, encoding="utf-8")
        exe = Path(args.cxx_bin) if args.cxx_bin else build_cpp(td)
        cxx = run_cpp(exe, corpus, args.vocab_size, td)
        py = run_python(corpus, args.vocab_size, args.python_pat)

    r = compare(py, cxx)
    print(f"python-pat={args.python_pat}  newlines={args.newlines}  vocab={args.vocab_size}  "
          f"merges python={r['n_python']} cxx={r['n_cxx']}")
    if r["identical"]:
        print(f"IDENTICAL: all {r['n_python']} merges match")
    else:
        i = r["first_diff_index"]
        print(f"DIVERGE at merge #{i}: identical prefix = {r['identical_prefix']}")
        if i is not None:
            print(f"  python: {r['first_diff_python']!r}")
            print(f"  c++   : {r['first_diff_cxx']!r}")

    if args.csv:
        sys.path.insert(0, str(HERE.parent))
        from schema import ResultWriter

        w = ResultWriter(args.csv, REPO)
        w.write(module="tokenizer", experiment="C2", system=f"cxx_vs_python[{args.python_pat}]",
                variant=f"vocab={args.vocab_size};newlines={args.newlines}", metric="merges_identical_prefix",
                value=r["identical_prefix"], unit="merges", n_reps=1,
                notes=f"identical={r['identical']}; n_python={r['n_python']}; n_cxx={r['n_cxx']}")
    # exit 0 only when the claim holds under the *matched* rule; gpt2 mode is informational
    return 0 if (r["identical"] or args.python_pat == "gpt2") else 1


if __name__ == "__main__":
    raise SystemExit(main())
