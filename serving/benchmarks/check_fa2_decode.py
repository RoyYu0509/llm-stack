"""GPU check of the FlashAttention-2 decode path (needs flash_attn + CUDA). Run before reporting any fa2 number.

  1. kernel : kvcache_attention_flash vs kvcache_attention_reference on random ragged data
  2. model  : PreallocDecodeBatch(impl=flash) vs PreallocDecodeBatch(impl=reference), identical KV and inputs
  3. system : prealloc/FA-2 vs the ragged batch (naive attention) -- same prefill KV, bf16 numerics differ by kernel,
              so this reports logit deltas and greedy-token agreement rather than asserting equality

Usage (from serving/):  python benchmarks/check_fa2_decode.py --workload <closed_loop workload json> [--dtype bfloat16]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

SERVING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVING))
for _p in (SERVING.parent / "training" / "cs336-basics", SERVING.parent / "training"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.append(str(_p))

spec = importlib.util.spec_from_file_location("record_demo", Path(__file__).resolve().parent / "record_demo.py")
rd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rd)

from cs336_basics.transfromer.kv_cache_attention import (  # noqa: E402
    kvcache_attention_flash, kvcache_attention_reference)
from inferenceLM.engine.prealloc_batch import PreallocDecodeBatch  # noqa: E402
from inferenceLM.engine.ragged_batch import RaggedDecodeBatch  # noqa: E402


def kernel_check(dtype, dev="cuda", B=16, H=12, D=64, max_len=1024, slots=24):
    g = torch.Generator(device=dev).manual_seed(0)
    k0 = torch.randn(slots, max_len, H, D, device=dev, dtype=dtype, generator=g)
    v0 = torch.randn_like(k0)
    lens = torch.randint(8, max_len - 2, (B,), device=dev, generator=g).int()
    idx = torch.randperm(slots, device=dev, generator=g)[:B].int()
    q = torch.randn(B, 1, H, D, device=dev, dtype=dtype, generator=g)
    kn, vn = torch.randn_like(q), torch.randn_like(q)
    a = kvcache_attention_flash(q, k0.clone(), v0.clone(), kn, vn, lens, idx)
    b = kvcache_attention_reference(q, k0.clone(), v0.clone(), kn, vn, lens, idx)
    # the append itself must also match: run once more and compare the written cache rows
    kf, vf = k0.clone(), v0.clone()
    kr, vr = k0.clone(), v0.clone()
    kvcache_attention_flash(q, kf, vf, kn, vn, lens, idx)
    kvcache_attention_reference(q, kr, vr, kn, vn, lens, idx)
    return float((a.float() - b.float()).abs().max()), float((kf.float() - kr.float()).abs().max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True)
    ap.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--checkpoint", default=None, help="real weights (model_only.pt with a 'model' state dict)")
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)
    dev = "cuda"

    d_out, d_cache = kernel_check(dtype)
    print(f"[1 kernel ] flash vs reference: max|out diff| {d_out:.3e}   max|cache diff after append| {d_cache:.3e}")

    cfg = rd.PRESETS["190m"]
    prompts = json.load(open(args.workload))["prompts"][: args.rows]
    fa_model = rd.build_model(cfg, dev, 0, args.checkpoint, rd.get_attention("fa2"), dtype=dtype)
    naive_model = rd.build_model(cfg, dev, 0, args.checkpoint, rd.get_attention("naive"), dtype=dtype)  # same weights
    slots, max_len = args.rows + 2, max(len(p) for p in prompts) + args.steps + 2

    pasts, first = [], []
    with torch.no_grad():
        for p in prompts:
            logits, past = fa_model.forward_with_cache(torch.tensor([p], device=dev))
            pasts.append(past)
            first.append(int(logits[0, -1].argmax()))

    def mk_prealloc(impl):
        b = PreallocDecodeBatch(fa_model, slots, max_len, dtype, dev, impl=impl)
        for i, past in enumerate(pasts):
            b.add(i, past)
        return b

    bf, br = mk_prealloc("flash"), mk_prealloc("reference")
    bn = RaggedDecodeBatch(naive_model)
    for i, past in enumerate(pasts):
        bn.add(i, past)

    tok = {i: t for i, t in enumerate(first)}
    saved = {"flash": [], "reference": [], "naive": []}   # per step: {row: logits}
    worst_fr = worst_fn = 0.0
    agree_fr = agree_fn = total = 0
    free = {i: t for i, t in enumerate(first)}       # free-running ragged/naive path (feeds its own tokens)
    free_tokens = {i: [] for i in range(args.rows)}
    fa_tokens = {i: [] for i in range(args.rows)}
    with torch.no_grad():
        for _ in range(args.steps):
            lf = bf.step(tok)
            lr = br.step(tok)
            ln = bn.step(tok)                        # teacher-forced with the FA path's tokens
            saved["flash"].append({i: lf[i].float().cpu() for i in lf})
            saved["reference"].append({i: lr[i].float().cpu() for i in lr})
            saved["naive"].append({i: ln[i].float().cpu() for i in ln})
            for i in tok:
                worst_fr = max(worst_fr, float((lf[i].float() - lr[i].float()).abs().max()))
                worst_fn = max(worst_fn, float((lf[i].float() - ln[i].float()).abs().max()))
                agree_fr += int(lf[i].argmax() == lr[i].argmax())
                agree_fn += int(lf[i].argmax() == ln[i].argmax())
                total += 1
            tok = {i: int(lf[i].argmax()) for i in lf}
            for i in tok:
                fa_tokens[i].append(tok[i])
    # free-running naive/ragged path from the same prefill, feeding its own greedy tokens
    bn2 = RaggedDecodeBatch(naive_model)
    for i, past in enumerate(pasts):
        bn2.add(i, past)
    with torch.no_grad():
        for _ in range(args.steps):
            out = bn2.step(free)
            free = {i: int(out[i].argmax()) for i in out}
            for i in free:
                free_tokens[i].append(free[i])
    ident = sum(fa_tokens[i] == free_tokens[i] for i in fa_tokens)
    print(f"[2 model  ] prealloc(flash) vs prealloc(reference): max|logit diff| {worst_fr:.3e}, "
          f"argmax agreement {agree_fr}/{total}")
    print(f"[3 system ] prealloc(flash) vs ragged(naive), same KV, teacher-forced: max|logit diff| {worst_fn:.3e}, "
          f"argmax agreement {agree_fn}/{total}")
    print(f"[3 system ] free-running greedy, {args.steps} tokens: {ident}/{args.rows} rows identical to the ragged/naive path")
    print(f"stats: {bf.stats}")

    # [4] ground truth: the same weights in fp32, full forward over the exact sequence each step saw.
    # bf16/fp16 paths are judged against THIS (error scale + logit scale), not only against each other.
    truth_model = rd.build_model(cfg, dev, 0, None, rd.get_attention("naive"), dtype=torch.float32)
    truth_model.load_state_dict(naive_model.state_dict(), strict=False)   # same weights (bf16 -> fp32 is exact), fp32 math
    err = {k: [] for k in saved}
    scale = []
    with torch.no_grad():
        for step in range(args.steps):
            for i, p in enumerate(prompts):
                seq = list(p) + [first[i]] + fa_tokens[i][:step]
                truth = truth_model(torch.tensor([seq], device=dev))[0, -1].float().cpu()
                scale.append(float(truth.std()))
                for k in saved:
                    err[k].append(float((saved[k][step][i] - truth).abs().max()))
    import statistics as st

    print(f"[4 truth  ] fp32 logit std {st.mean(scale):.2f}; max|logit - fp32 truth|: "
          + ", ".join(f"{k} mean {st.mean(v):.2f} / worst {max(v):.2f}" for k, v in err.items()))


if __name__ == "__main__":
    main()
