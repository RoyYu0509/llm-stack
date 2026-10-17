"""K1-pretok -- the *pretokenization* stage only: Python (his implementation) vs C++ (his implementation).

Decision record: docs/sprints/week7-discussion.md D-7, "Tokenizer". The question the interviewer asked:
"Python -> C++ pretokenization: did the line count change? compile + run time vs the Python run time? any
optimisation besides 'C++ is compiled'?". This script produces the numbers; it does not time the merge loop
(that part is serial in both implementations and is annotated from older data, see the README).

What is compared (the same corpus file, the same pre-token rule = the C++ `simple_split`):

  python_serial_inproc        his `process_chunk` over the whole file, in this process (no pool)
  python_<N>proc_steady       his `process_chunk` + `find_chunk_boundaries`, N workers from a pool that was
                              spawned and warmed BEFORE timing (process start-up excluded)
  python_<N>proc_cold         his `BBPE.pretokenization()` exactly as the trainer calls it (pool created
                              inside the call -> includes worker spawn + imports; shown, not headlined)
  cxx_serial / cxx_4threads   `pretokenizeSerial` / `pretokenizeParallel` (4 threads is hard-coded there),
                              timed inside `pretok_driver` around the call only

Fairness rules enforced here:
  * the Python pattern is overridden with the C++ rule (`SIMPLE_SPLIT_PATTERN`, same switch as C2);
  * before any timing, the C++ driver (serial AND parallel) must reproduce a line-wise Python reference
    bit-for-bit on a sample (unique count, total count, order-independent checksum) -- otherwise the two
    sides are not doing the same work and no number is written;
  * one untimed warm-up round (also warms the OS page cache), then `--reps` timed rounds, configs interleaved;
  * compile time is reported separately (one-off), never added to the run time.

Known, reported difference that cannot be removed without editing his code: the C++ reader is `std::getline`
(newline dropped, each line processed alone) while the Python trainer sees the newline characters; the
sample check prints how many pre-tokens that changes.

"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import c2_merges_equal as c2  # noqa: E402
import pretok_loc  # noqa: E402
from schema import ResultWriter  # noqa: E402
import manifest  # noqa: E402

REPO = c2.REPO
CXX_DIR = c2.CXX_DIR
SPECIAL = c2.SPECIAL
SPECIAL_B = SPECIAL.encode("utf-8")
MASK = (1 << 64) - 1
WS = b" \t\n\r"


# ------------------------------------------------------------------------------ work-equivalence reference
def fnv1a64(b: bytes) -> int:
    h = 1469598103934665603
    for c in b:
        h = ((h ^ c) * 1099511628211) & MASK
    return h


def pretok_stats(counter) -> dict:
    """(unique, total, checksum); the checksum definition is identical to pretok_driver.cpp."""
    csum = 0
    total = 0
    for seq, count in counter.items():
        total += count
        csum = (csum + (fnv1a64(b"".join(seq)) ^ ((count * 0x9E3779B97F4A7C15) & MASK))) & MASK
    return dict(unique=len(counter), total=total, checksum=str(csum))


def _split_specials(line: bytes, specials: list[bytes]) -> list[bytes]:
    """Mirror of C++ split_by_special_tokens: earliest special first; ties go to the earlier list entry."""
    if not specials:
        return [line]
    parts, pos = [], 0
    while pos < len(line):
        best_pos, best_len = -1, 0
        for sp in specials:
            f = line.find(sp, pos)
            if f != -1 and (best_pos == -1 or f < best_pos):
                best_pos, best_len = f, len(sp)
        if best_pos == -1:
            parts.append(line[pos:])
            break
        if best_pos > pos:
            parts.append(line[pos:best_pos])
        parts.append(line[best_pos:best_pos + best_len])
        pos = best_pos + best_len
    return parts


def _simple_split(part: bytes) -> list[bytes]:
    """Mirror of C++ simple_split, char by char: a whitespace byte closes the current word and starts the next."""
    words, cur = [], bytearray()
    for c in part:
        if c in WS and cur:
            words.append(bytes(cur))
            cur = bytearray()
        cur.append(c)
    if cur:
        words.append(bytes(cur))
    return words


def ref_pretokenize_linewise(data: bytes, specials=(SPECIAL_B,)) -> Counter:
    """Line-wise reference with `std::getline` semantics (newline removed, final empty piece is not a line)."""
    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    specials = list(specials)
    counter: Counter = Counter()
    single = [bytes([i]) for i in range(256)]
    for line in lines:
        for part in _split_specials(line, specials):
            if not part:
                continue
            if part in specials:
                counter[(part,)] += 1
                continue
            for word in _simple_split(part):
                counter[tuple(single[b] for b in word)] += 1
    return counter


# ------------------------------------------------------------------------------ his Python side
def make_python_tokenizer():
    import regex
    from cs336_basics.bpe_tokenizer.tokenizer import Tokenizer

    tok = Tokenizer(vocab=512, special_tokens=[SPECIAL])
    tok.PAT = regex.compile(c2.SIMPLE_SPLIT_PATTERN)  # the C++ rule, same override as C2
    return tok


def _warm(_i: int) -> int:  # runs in the pool workers: import everything, hold the worker so all N spawn
    import cs336_basics.bpe_tokenizer.helpers.bpe_helper  # noqa: F401
    time.sleep(0.4)
    return os.getpid()


def py_serial_inproc(tok, path: Path):
    from cs336_basics.bpe_tokenizer.helpers.bpe_helper import process_chunk

    size = path.stat().st_size
    t0 = time.perf_counter()
    counter = process_chunk((0, size), str(path), tok.SPLIT, tok.PAT, tok.special_tokens_bytes)
    dt = time.perf_counter() - t0
    return dt, counter


def py_pool_steady(pool, tok, path: Path, nproc: int):
    from cs336_basics.bpe_tokenizer.helpers.bpe_helper import find_chunk_boundaries, process_chunk

    t0 = time.perf_counter()
    with open(path, "rb") as f:
        b = find_chunk_boundaries(f, nproc, SPECIAL_B)
    futs = [pool.submit(process_chunk, (s, e), str(path), tok.SPLIT, tok.PAT, tok.special_tokens_bytes)
            for s, e in zip(b[:-1], b[1:])]
    total: Counter = Counter()
    for fut in as_completed(futs):
        total.update(fut.result())
    return time.perf_counter() - t0, total


def py_cold(tok, path: Path, nproc: int):
    os.environ["TQDM_DISABLE"] = "1"
    tok.pretok_dict = Counter()
    t0 = time.perf_counter()
    tok.pretokenization(str(path), nproc, SPECIAL_B)
    return time.perf_counter() - t0, tok.pretok_dict


# ------------------------------------------------------------------------------ his C++ side
def find_cxx() -> str:
    cxx = shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")
    if not cxx:
        raise RuntimeError("no C++ compiler (g++/c++/clang++)")
    return cxx


FLAGS = ["-std=c++20", "-O2", "-pthread"]  # same as bpe_tokenizer_cpp/run_benchmark.sh
LIB_SOURCES = ["bpe_tokenizer.cpp", "pretokenizer.cpp", "parallel_helper.cpp"]


def build_driver(out_dir: Path, compile_reps: int = 3) -> tuple[Path, dict]:
    """Compile exactly like run_benchmark.sh (one command, all sources) with pretok_driver.cpp instead of
    main_benchmark.cpp. Returns the executable and the compile-time measurements (clean builds)."""
    cxx = find_cxx()
    exe = out_dir / "pretok_driver"
    srcs = [str(CXX_DIR / n) for n in LIB_SOURCES] + [str(HERE / "pretok_driver.cpp")]
    inc = ["-I", str(CXX_DIR)]

    def timed(cmd):
        t0 = time.perf_counter()
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return time.perf_counter() - t0

    full = [timed([cxx, *FLAGS, *srcs, *inc, "-o", str(exe)]) for _ in range(compile_reps)]
    per_tu = {}
    for s in srcs:
        obj = out_dir / (Path(s).stem + ".o")
        per_tu[Path(s).name] = statistics.median(
            timed([cxx, *FLAGS, "-c", s, *inc, "-o", str(obj)]) for _ in range(compile_reps))
    version = subprocess.run([cxx, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    return exe, dict(cxx=cxx, version=version, flags=" ".join(FLAGS), full_median_s=statistics.median(full),
                     full_all_s=full, per_tu_median_s=per_tu)


def run_cxx(exe: Path, corpus: Path, mode: str) -> dict:
    p = subprocess.run([str(exe), str(corpus), mode], capture_output=True, text=True, check=True)
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT "):])


# ------------------------------------------------------------------------------ validation
def make_sample(corpus: Path, nbytes: int, out: Path) -> int:
    with open(corpus, "rb") as f:
        data = f.read(nbytes)
    cut = data.rfind(b"\n")
    data = data[: cut + 1] if cut != -1 else data
    out.write_bytes(data)
    return len(data)


def validate_equal_work(exe: Path, tok, sample: Path) -> dict:
    """The C++ driver (serial and parallel) must equal the line-wise reference on `sample`."""
    ref = pretok_stats(ref_pretokenize_linewise(sample.read_bytes()))
    res = {}
    ok = True
    for mode in ("serial", "parallel"):
        r = run_cxx(exe, sample, mode)
        got = dict(unique=r["unique"], total=r["total"], checksum=r["checksum"])
        res[mode] = got
        if got != ref:
            ok = False
    from cs336_basics.bpe_tokenizer.helpers.bpe_helper import process_chunk

    his = pretok_stats(process_chunk((0, sample.stat().st_size), str(sample), tok.SPLIT, tok.PAT,
                                     tok.special_tokens_bytes))
    return dict(ok=ok, reference=ref, cxx=res, his_python=his,
                newline_effect=dict(d_unique=his["unique"] - ref["unique"], d_total=his["total"] - ref["total"]))


# ------------------------------------------------------------------------------ main
def env_string(compile_info: dict) -> str:
    bits = [f"python {platform.python_version()}", f"machine {platform.machine()}"]
    try:
        import regex

        bits.append(f"regex {regex.__version__}")
    except Exception:
        pass
    if sys.platform == "darwin":
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout.splitlines()
        bits += [cpu, f"ncpu {os.cpu_count()}", batt[0] if batt else "", batt[1].strip() if len(batt) > 1 else ""]
    else:
        bits += [platform.processor(), f"ncpu {os.cpu_count()}"]
    bits += [compile_info["version"], "flags " + compile_info["flags"]]
    return "; ".join(b for b in bits if b)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1, help="untimed full rounds")
    ap.add_argument("--py-procs", type=int, nargs="+", default=[4])
    ap.add_argument("--skip-cold", action="store_true", help="skip the pool-created-inside-the-call variant")
    ap.add_argument("--validate-bytes", type=int, default=8_000_000)
    ap.add_argument("--force", action="store_true", help="time even if the equal-work check fails (do not quote)")
    ap.add_argument("--out", type=Path, default=HERE.parent / "results" / "final" / "tokenizer.csv")
    args = ap.parse_args(argv)

    corpus = args.corpus.resolve()
    size_mb = corpus.stat().st_size / 1e6
    manifest.record("tokenizer_pretokenize", out_dir=args.out.parent, extra=dict(corpus=str(corpus)))
    w = ResultWriter(args.out, REPO)
    base = dict(module="tokenizer", experiment="K1-pretok")
    variant = f"corpus={corpus.name};size_mb={size_mb:.1f};rule=simple_split;special={SPECIAL}"

    loc = pretok_loc.pretok_loc()
    for lang in ("python", "cxx"):
        for kind in ("code", "physical"):
            w.write(**base, system=f"{lang}_pretokenize", variant="functions: see pretok_loc.py",
                    metric=f"loc_{kind}", value=loc[lang][kind], unit="lines", n_reps=1,
                    notes="pretokenization functions only; 'code' = non-blank, non-comment, non-docstring")
    print(f"LOC  python code={loc['python']['code']} physical={loc['python']['physical']}   "
          f"c++ code={loc['cxx']['code']} physical={loc['cxx']['physical']}")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        print("compiling (3 clean builds + 3 per-TU) ...")
        exe, cinfo = build_driver(td)
        env = env_string(cinfo)
        print(f"compiler: {cinfo['version']}  flags: {cinfo['flags']}")
        print(f"compile, one-off: full build median {cinfo['full_median_s']:.2f}s  per-TU {cinfo['per_tu_median_s']}")
        w.write(**base, system="cxx_build", variant=cinfo["flags"], metric="compile_time_s",
                value=round(cinfo["full_median_s"], 3), unit="s",
                std=round(statistics.pstdev(cinfo["full_all_s"]), 3), n_reps=len(cinfo["full_all_s"]),
                notes=f"one-off, clean build, one command like run_benchmark.sh; per-TU (-c) medians {cinfo['per_tu_median_s']}")
        w.write(**base, system="environment", variant=variant, metric="environment", value=1, unit="flag",
                n_reps=1, notes=env)

        tok = make_python_tokenizer()
        sample = td / "sample.txt"
        n = make_sample(corpus, args.validate_bytes, sample)
        print(f"equal-work check on the first {n / 1e6:.1f} MB ...")
        v = validate_equal_work(exe, tok, sample)
        print(f"  reference {v['reference']}")
        print(f"  c++ serial   {v['cxx']['serial']}\n  c++ parallel {v['cxx']['parallel']}")
        print(f"  his Python (newlines visible) differs from the reference by {v['newline_effect']}")
        w.write(**base, system="equal_work_check", variant=f"sample_bytes={n}", metric="cxx_equals_reference",
                value=int(v["ok"]), unit="bool", n_reps=1,
                notes=f"reference={v['reference']}; newline_effect={v['newline_effect']}")
        if not v["ok"] and not args.force:
            print("EQUAL-WORK CHECK FAILED: the C++ driver and the reference disagree; not timing.", file=sys.stderr)
            return 1

        configs = ["python_serial_inproc"]
        configs += [f"python_{n}proc_steady" for n in args.py_procs]
        if not args.skip_cold:
            configs += [f"python_{n}proc_cold" for n in args.py_procs]
        configs += ["cxx_serial", "cxx_4threads"]
        pools = {n: ProcessPoolExecutor(max_workers=n) for n in args.py_procs}
        for n, pool in pools.items():  # spawn and warm all workers before the clock starts
            list(pool.map(_warm, range(n)))

        times: dict[str, list[float]] = {c: [] for c in configs}
        stats: dict[str, dict] = {}
        try:
            for rnd in range(args.warmup + args.reps):
                timed = rnd >= args.warmup
                for c in configs:
                    if c == "python_serial_inproc":
                        dt, cnt = py_serial_inproc(tok, corpus)
                    elif c.endswith("_steady"):
                        n = int(c.split("_")[1].removesuffix("proc"))
                        dt, cnt = py_pool_steady(pools[n], tok, corpus, n)
                    elif c.endswith("_cold"):
                        n = int(c.split("_")[1].removesuffix("proc"))
                        dt, cnt = py_cold(tok, corpus, n)
                    else:
                        r = run_cxx(exe, corpus, "serial" if c == "cxx_serial" else "parallel")
                        dt, cnt = r["seconds"], None
                        stats[c] = dict(unique=r["unique"], total=r["total"])
                    if cnt is not None:
                        stats[c] = dict(unique=len(cnt), total=sum(cnt.values()))
                    del cnt
                    gc.collect()
                    tag = "" if timed else " (warm-up)"
                    print(f"round {rnd + 1}/{args.warmup + args.reps} {c:26s} {dt:8.2f}s{tag}", flush=True)
                    if timed:
                        times[c].append(dt)
        finally:
            for pool in pools.values():
                pool.shutdown(wait=True, cancel_futures=True)

    med = {c: statistics.median(v) for c, v in times.items()}
    for c, v in times.items():
        w.write(**base, system=c, variant=variant, metric="pretokenize_time_s", value=round(med[c], 3), unit="s",
                std=round(statistics.pstdev(v), 3), n_reps=len(v), warmup=args.warmup,
                notes=f"median; min={min(v):.3f}; all={[round(x, 3) for x in v]}; stats={stats.get(c)}")
    print("\n== pretokenize time, median of", args.reps, "(s)")
    for c in configs:
        print(f"  {c:26s} {med[c]:9.2f}   min {min(times[c]):8.2f}   {stats.get(c)}")

    def ratio(num, den, what):
        if num in med and den in med:
            x = med[num] / med[den]
            w.write(**base, system=f"{num}_over_{den}", variant=variant, metric="time_ratio", value=round(x, 3),
                    unit="x", n_reps=args.reps, notes=f"derived from medians: {what}")
            print(f"  {what}: {x:.2f}x")

    n4 = args.py_procs[0]
    ratio("python_serial_inproc", "cxx_serial", "Python serial / C++ serial")
    ratio(f"python_{n4}proc_steady", "cxx_4threads", f"Python {n4}proc (warm pool) / C++ 4 threads")
    ratio(f"python_{n4}proc_cold", "cxx_4threads", f"Python {n4}proc (pool inside call) / C++ 4 threads")
    ratio("cxx_serial", "cxx_4threads", "C++ serial / C++ 4 threads")
    ratio("python_serial_inproc", f"python_{n4}proc_steady", f"Python serial / Python {n4}proc (warm pool)")
    cmp_unique = {k: v["unique"] for k, v in stats.items()}
    print(f"  unique pre-tokens per config (C++ drops newlines, Python keeps them): {cmp_unique}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
