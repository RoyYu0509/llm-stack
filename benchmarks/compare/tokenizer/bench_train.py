"""K1 -- BPE *training* time: Python vs C++ (serial / 4 threads) vs HuggingFace `tokenizers`.

READ THIS BEFORE QUOTING A NUMBER (details in ../README.md, "K"):

  * The three systems do not pre-tokenize the same way (C++: whitespace split, Python: GPT-2
    regex, HF: ByteLevel + its own regex), and the C++ trainer discards newlines (see C2). So the
    produced vocabularies differ and a speed ratio compares "train a BPE" jobs, not identical jobs.
    The row set `python[simple]` uses the same split rule as C++ to make that pair comparable.
  * The C++ binary is an executable, not an extension: the time includes process start-up and the
    small encode/decode self-test that `main_benchmark` runs. Report that, do not hide it.
  * One run on one corpus is a measurement, not a distribution: use --reps >= 3 and a corpus of
    at least a few hundred MB for anything that goes on a CV.

Usage:
  python bench_train.py --corpus corpus.txt --vocab-size 5000 --reps 3 --out ../results/final/tokenizer.csv
"""
from __future__ import annotations

import argparse
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from schema import ResultWriter  # noqa: E402
import c2_merges_equal as c2  # noqa: E402

REPO = c2.REPO
SPECIAL = c2.SPECIAL


def time_python(corpus: Path, vocab: int, pat: str, procs: int) -> float:
    t0 = time.perf_counter()
    c2.run_python(corpus, vocab, pat, procs=procs)
    return time.perf_counter() - t0


def build_cxx_benchmark(out_dir: Path) -> Path:
    import shutil

    exe = out_dir / "bpe_benchmark"
    cxx = shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")
    srcs = [c2.CXX_DIR / n for n in ("bpe_tokenizer.cpp", "pretokenizer.cpp",
                                     "parallel_helper.cpp", "main_benchmark.cpp")]
    subprocess.run([cxx, "-std=c++20", "-O2", "-pthread", *map(str, srcs), "-I", str(c2.CXX_DIR),
                    "-o", str(exe)], check=True, capture_output=True, text=True)
    return exe


def time_cxx(exe: Path, corpus: Path, vocab: int) -> dict[str, float]:
    """`bpe_benchmark` times serial and 4-thread training itself and prints a table."""
    out = subprocess.run([str(exe), str(corpus), str(vocab)], check=True, capture_output=True,
                         text=True).stdout
    res = {}
    for label, key in (("Serial", "cxx_serial"), ("Parallel", "cxx_4threads")):
        m = re.search(rf"║ {label}[^║]*║\s*([0-9.]+)s", out)
        if not m:
            raise RuntimeError(f"could not parse '{label}' time from bpe_benchmark output:\n{out[-600:]}")
        res[key] = float(m.group(1))
    return res


_HF_SNIPPET = r"""
import sys, time
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
corpus, vocab = sys.argv[1], int(sys.argv[2])
tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
tr = trainers.BpeTrainer(vocab_size=vocab, special_tokens=["<|endoftext|>"],
                         initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
t0 = time.perf_counter(); tok.train([corpus], tr); print(time.perf_counter() - t0)
"""


def time_hf(corpus: Path, vocab: int, threads: int) -> float:
    env = {**os.environ, "RAYON_NUM_THREADS": str(threads), "TOKENIZERS_PARALLELISM": "true"}
    out = subprocess.run([sys.executable, "-I", "-c", _HF_SNIPPET, str(corpus), str(vocab)],
                         check=True, capture_output=True, text=True, env=env).stdout.strip().splitlines()[-1]
    return float(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--vocab-size", type=int, default=2000)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--py-procs", type=int, default=4)
    ap.add_argument("--skip", nargs="*", default=[], choices=["python", "cxx", "hf"])
    ap.add_argument("--out", type=Path, default=HERE.parent / "results" / "final" / "tokenizer.csv")
    args = ap.parse_args(argv)

    w = ResultWriter(args.out, REPO)
    size_mb = args.corpus.stat().st_size / 2**20
    runs: dict[str, list[float]] = {}

    def add(k, v):
        runs.setdefault(k, []).append(v)

    with tempfile.TemporaryDirectory() as td:
        exe = None if "cxx" in args.skip else build_cxx_benchmark(Path(td))
        for r in range(args.reps):
            if "python" not in args.skip:
                add("python[simple]_1proc", time_python(args.corpus, args.vocab_size, "simple", 1))
                add(f"python[simple]_{args.py_procs}proc", time_python(args.corpus, args.vocab_size, "simple", args.py_procs))
                add(f"python[gpt2]_{args.py_procs}proc", time_python(args.corpus, args.vocab_size, "gpt2", args.py_procs))
            if exe:
                for k, v in time_cxx(exe, args.corpus, args.vocab_size).items():
                    add(k, v)
            if "hf" not in args.skip:
                add("hf_tokenizers_1thread", time_hf(args.corpus, args.vocab_size, 1))
                add("hf_tokenizers_all_threads", time_hf(args.corpus, args.vocab_size, os.cpu_count() or 4))
            print(f"rep {r + 1}/{args.reps} done")

    for system, vals in runs.items():
        m = statistics.fmean(vals)
        s = statistics.pstdev(vals) if len(vals) > 1 else ""
        w.write(module="tokenizer", experiment="K1", system=system,
                variant=f"vocab={args.vocab_size};corpus_mb={size_mb:.1f}",
                metric="train_time_s", value=round(m, 3), unit="s", std=round(s, 3) if s != "" else "",
                n_reps=len(vals), notes="pre-tokenization rules differ across systems (see header)")
        print(f"{system:34s} {m:9.2f} s  (n={len(vals)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
