"""C5 -- does his inference engine produce the same greedy tokens as HuggingFace `generate`?

Decision record: docs/sprints/week7-discussion.md D-7, "Correctness C5". Why this is a gate: every engine
throughput number is only worth quoting if the engine computes the right thing. The HF side is the C3 conversion
of the same checkpoint (`convert_hf/convert_190m_to_hf.py`, logits verified to ~1e-5 in fp32).

Two engine paths are compared, because a bug can hide in only one of them:
  seq   concurrency 1, one request at a time (single-sequence KV cache)
  cb    concurrency C, continuous batching (left-padded ragged batch, per-row masks)  <- the risky one

The OPTIMISED cb path (pre-allocated KV + flash-attn decode, batched prefill, torch.compile) is compared with the same
script: --attn fa2 --prefill-last-logits --prefill-batch-tokens N --compile-decode (cb only). FlashAttention has no
fp32 kernel, so that path runs in bf16 and is REPORT-ONLY; on CPU the same plumbing can be exercised in fp32 with
`--decode-backend prealloc` (naive attention, eager cache kernel, `--compile-backend aot_eager`), which is how the tests
prove the new path computes the same function. A run aborts if the path it was asked to use never ran (compiled graph
absent, no prefill group larger than one, no pre-allocated decode step): a label must not outlive its mechanism.

Pass rule
  fp32   every prompt must match HF **token for token** (exit code 1 otherwise). Strict fp32 matmuls are forced
         (`float32_matmul_precision=highest`) so TF32 rounding cannot create a false divergence.
  other  (bf16/fp16) two correct implementations may legitimately diverge after some step because their rounding
         differs, so the script only REPORTS: exact-match count, first divergence index, and the teacher-forced
         argmax agreement (HF scores the engine's own tokens: fraction of positions where HF's argmax equals
         the engine's token). Divergence is not a bug by itself; a low teacher-forced agreement would be.

Usage
  python c5_greedy_equal.py --checkpoint ../../training/ckpt_190m_fwe/model_only.pt \
      --workload ../workloads/workload_wiki128.json --n-prompts 16 --new-tokens 32 --dtype float32 --device cpu

"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
COMPARE = HERE.parent
REPO = HERE.parents[2]
for p in (COMPARE, COMPARE / "convert_hf", REPO / "serving" / "benchmarks"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


# ------------------------------------------------------------------------------------ HF side
def hf_greedy(hf, prompt: list[int], n_new: int, device: str) -> list[int]:
    """Greedy continuation with EOS stopping disabled (the engine runs with NO_EOS, so the comparison must too)."""
    hf.generation_config.eos_token_id = None
    hf.generation_config.bos_token_id = None
    ids = torch.tensor([prompt], device=device)
    out = hf.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=n_new, do_sample=False,
                      num_beams=1, eos_token_id=None, pad_token_id=0, use_cache=True)
    return out[0, len(prompt):].tolist()


@torch.no_grad()
def teacher_forced_agreement(hf, prompt: list[int], gen: list[int], device: str) -> float:
    """Fraction of generated positions where HF's argmax, given prompt + the engine's own previous tokens,
    equals the engine's token."""
    ids = torch.tensor([prompt + gen], device=device)
    logits = hf(input_ids=ids).logits[0]
    pred = logits[len(prompt) - 1: len(prompt) - 1 + len(gen)].argmax(-1).tolist()
    return sum(int(a == b) for a, b in zip(pred, gen)) / max(1, len(gen))


def first_divergence(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


# ------------------------------------------------------------------------------------ engine side
def engine_opts(attn: str, decode_backend: str = "auto", last_logits: bool = False, prefill_batch_tokens: int | None = None,
                compile_decode: bool = False, compile_backend: str = "inductor", modes=("cb",)) -> dict:
    """The `record_trial` switches of the engine path under test, validated the way record_demo.py validates them."""
    backend = decode_backend if decode_backend != "auto" else ("prealloc" if attn == "fa2" else "ragged")
    if compile_decode and backend != "prealloc":
        raise SystemExit("--compile-decode compiles the pre-allocated-cache decode: it needs --attn fa2 or --decode-backend prealloc")
    if (compile_decode or prefill_batch_tokens) and "seq" in modes:
        raise SystemExit("--compile-decode / --prefill-batch-tokens belong to the continuous-batching path: use --modes cb")
    return dict(decode_backend=backend, last_logits=last_logits, prefill_batch_tokens=prefill_batch_tokens,
                compile_decode=compile_decode, compile_backend=compile_backend)


def check_exercised(trial: dict, opts: dict) -> None:
    """Abort when the path the run is labelled with did not actually run (same idea as record_demo.guard)."""
    kv = trial.get("kv_stats") or {}
    if opts.get("decode_backend") == "prealloc" and not kv.get("steps"):
        raise SystemExit("ABORT: asked for the pre-allocated decode but PreallocDecodeBatch never ran a step")
    if opts.get("compile_decode") and not kv.get("compiled"):
        raise SystemExit("ABORT: asked for the compiled decode but the decode batch was not compiled")
    sizes = trial.get("prefill_group_sizes") or []
    if opts.get("prefill_batch_tokens") and max(sizes, default=0) < 2:
        raise SystemExit("ABORT: asked for batched prefill but no prefill forward packed more than one prompt; "
                         "raise --concurrency or the budget, or this run says nothing about grouped prefill")


def run_engine(rd, model, adapters, prompts, n_new, mode, concurrency, opts: dict | None = None) -> dict[str, list[int]]:
    reqs = [(f"r{i:03d}", p, n_new) for i, p in enumerate(prompts)]
    trial = asyncio.run(rd.record_trial(model, adapters, mode, concurrency, reqs, **(opts or {})))
    check_exercised(trial, opts or {})
    return {rid: list(trial["outputs"][rid]) for rid, _, _ in reqs}


def compare_all(rd, adapters, engine_model, hf, prompts, n_new, modes, concurrency, device, strict,
                opts: dict | None = None) -> list[dict]:
    refs = [hf_greedy(hf, p, n_new, device) for p in prompts]
    rows = []
    for mode in modes:
        outs = run_engine(rd, engine_model, adapters, prompts, n_new, mode, 1 if mode == "seq" else concurrency, opts)
        gens = [outs[f"r{i:03d}"] for i in range(len(prompts))]
        divs = [first_divergence(g, r) for g, r in zip(gens, refs)]
        tf = [teacher_forced_agreement(hf, p, g, device) for p, g in zip(prompts, gens)]
        rows.append(dict(mode=mode, n_prompts=len(prompts), n_new=n_new,
                         exact=sum(d is None for d in divs), first_divergence=divs,
                         teacher_forced_mean=sum(tf) / len(tf), teacher_forced_min=min(tf),
                         passed=(all(d is None for d in divs) if strict else None)))
    return rows


# ------------------------------------------------------------------------------------ models from a checkpoint
def build_pair(checkpoint: str, device: str, dtype: torch.dtype, attn: str):
    import convert_190m_to_hf as c3
    import record_demo as rd

    sd = c3.load_state_dict(checkpoint)
    cfg = c3.infer_config(sd)
    hf = c3.build_hf_model(cfg, c3.convert_state_dict(sd, cfg)).to(device=device, dtype=dtype).eval()
    model_cfg = dict(vocab_size=cfg["vocab_size"], context_length=cfg["context_length"], num_layers=cfg["n_layers"],
                     d_model=cfg["d_model"], num_heads=cfg["n_heads"], d_ff=cfg["d_ff"], rope_theta=cfg["theta"])
    engine_model = rd.build_model(model_cfg, device, 0, checkpoint, rd.get_attention(attn), dtype=dtype)
    return rd, hf, engine_model


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--workload", required=True, help="JSON with token-id `prompts` (the serving workload)")
    ap.add_argument("--n-prompts", type=int, default=16)
    ap.add_argument("--min-prompt-len", type=int, default=0)
    ap.add_argument("--new-tokens", type=int, default=32)
    ap.add_argument("--dtype", default="float32", choices=list(DTYPES))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--attn", default="naive", choices=["naive", "sdpa", "fa2"],
                    help="attention inside the baseline engine (naive = the original one; fa2 = flash-attn prefill + kv-cache decode, bf16/fp16 only)")
    ap.add_argument("--decode-backend", default="auto", choices=["auto", "ragged", "prealloc"],
                    help="auto: prealloc with --attn fa2, ragged otherwise. prealloc + naive runs on CPU (tests)")
    ap.add_argument("--prefill-last-logits", action="store_true")
    ap.add_argument("--prefill-batch-tokens", type=int, default=None)
    ap.add_argument("--compile-decode", action="store_true")
    ap.add_argument("--compile-backend", default="inductor", help="aot_eager = trace only (CPU)")
    ap.add_argument("--label", default=None, help="name of the engine path in the CSV (default: --attn), e.g. fa2_compiled")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--modes", nargs="+", default=None, choices=["seq", "cb"],
                    help="default: seq cb for the original path; cb only when an optimisation switch is on")
    ap.add_argument("--out", default=str(COMPARE / "results" / "final" / "serving_correctness.csv"))
    args = ap.parse_args(argv)

    import json

    from schema import ResultWriter
    import manifest

    manifest.record("c5_greedy_equal", out_dir=Path(args.out).parent, extra=dict(dtype=args.dtype, device=args.device))
    torch.set_float32_matmul_precision("highest")  # no TF32: fp32 must mean fp32 on both sides
    dtype = DTYPES[args.dtype]
    prompts = [p for p in json.load(open(args.workload))["prompts"] if len(p) >= args.min_prompt_len][: args.n_prompts]
    optimised = bool(args.compile_decode or args.prefill_batch_tokens)
    modes = args.modes or (["cb"] if optimised else ["seq", "cb"])
    if args.attn == "fa2" and args.dtype == "float32":
        raise SystemExit("--attn fa2 needs --dtype bfloat16 or float16 (FlashAttention has no fp32 kernel)")
    opts = engine_opts(args.attn, args.decode_backend, args.prefill_last_logits, args.prefill_batch_tokens,
                       args.compile_decode, args.compile_backend, modes)
    rd, hf, engine_model = build_pair(args.checkpoint, args.device, dtype, args.attn)
    adapters = rd.load_adapters(rd.SERVING / "tests" / "m2_adapters.py")
    strict = args.dtype == "float32"
    rows = compare_all(rd, adapters, engine_model, hf, prompts, args.new_tokens, modes, args.concurrency,
                       args.device, strict, opts)
    label = args.label or args.attn
    flags = (f"decode={opts['decode_backend']};last_logits={opts['last_logits']};"
             f"prefill_batch_tokens={opts['prefill_batch_tokens']};compile={opts['compile_decode']}")

    w = ResultWriter(args.out, REPO)
    rc = 0
    for r in rows:
        verdict = {True: "PASS", False: "FAIL", None: "report-only"}[r["passed"]]
        print(f"C5 {args.dtype} engine={label} mode={r['mode']:3s}: exact {r['exact']}/{r['n_prompts']}  "
              f"first divergence per prompt {r['first_divergence']}  teacher-forced mean {r['teacher_forced_mean']:.4f} "
              f"min {r['teacher_forced_min']:.4f}  -> {verdict}")
        base = dict(module="serving", experiment="C5", system=f"engine_{label}_{r['mode']}_vs_hf_generate",
                    variant=f"n_prompts={r['n_prompts']};new_tokens={r['n_new']};device={args.device};"
                            f"concurrency={1 if r['mode'] == 'seq' else args.concurrency};{flags}",
                    dtype=args.dtype)
        w.write(**base, metric="greedy_exact_match_prompts", value=r["exact"], unit="prompts", n_reps=1,
                notes=f"of {r['n_prompts']}; first divergence: {r['first_divergence']}; {verdict}")
        w.write(**base, metric="teacher_forced_argmax_agreement_mean", value=round(r["teacher_forced_mean"], 5),
                unit="fraction", n_reps=1, notes=f"min over prompts {r['teacher_forced_min']:.4f}")
        if r["passed"] is False:
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
