"""Correctness gate for the optimisation ladder (stage C). Run BEFORE any throughput number is reported.

Every rung changes how work is done, not what is computed, so each one is held against the rung before it, with
thresholds fixed here (not after seeing the numbers):

  [A] prefill    LMEngine.prefill_batch (right-padded group) vs prefill one prompt at a time: first greedy token agrees
                 on >= 95% of the prompts, and each row's K/V (padded tail sliced off) is within 2% of the largest |K|.
  [B] compile    PreallocDecodeBatch(compile_decode=True) vs the same batch eager, teacher-forced with identical tokens
                 for --steps steps: max|logit diff| <= TOL[dtype] and argmax agreement >= 98%; and the K/V cache written
                 by the compiled step equals the eager one (same tolerance): the cache is written IN PLACE.
                 Also reported: whether the compiled steps allocated a cache-sized buffer (a functionalisation clone).

Exit code 0 = all pass, 1 = some gate failed. On the GPU box run with the real checkpoint:
    python benchmarks/check_stage_c.py --workload <workload.json> --checkpoint <model_only.pt> --dtype bfloat16
Self-test on a laptop (no GPU): --device cpu --preset small --dtype float32 --impl reference --compile-backend aot_eager
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

from inferenceLM.engine.lm_engine import LMEngine  # noqa: E402
from inferenceLM.engine.prealloc_batch import PreallocDecodeBatch  # noqa: E402
from inferenceLM.engine.transformer_lm_adapter import TransformerLMAdapter  # noqa: E402

TOL = {"float32": 1e-3, "bfloat16": 0.5, "float16": 0.25}      # max |logit diff| between two variants of one model
AGREE_DECODE, AGREE_PREFILL, KV_REL = 0.98, 0.95, 0.02


def prompts_for(args, cfg):
    if args.workload:
        return json.load(open(args.workload))["prompts"][: args.rows]
    g = torch.Generator().manual_seed(0)
    return [torch.randint(0, cfg["vocab_size"], (n,), generator=g).tolist() for n in (9, 23, 14, 31, 6, 18, 27, 12)[: args.rows]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--preset", default="190m")
    ap.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--impl", default="auto", help="kv-cache attention: auto | flash | reference")
    ap.add_argument("--compile-backend", default="inductor")
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--steps", type=int, default=16)
    args = ap.parse_args()
    dtype, dev, cfg = getattr(torch, args.dtype), args.device, rd.PRESETS[args.preset]
    attn = rd.get_attention("fa2") if args.impl != "reference" else rd.get_attention("naive")
    model = rd.build_model(cfg, dev, 0, args.checkpoint, attn, dtype=dtype)
    prompts = prompts_for(args, cfg)
    ok = True

    # [A] prefill: grouped (right-padded) vs one by one
    eng = LMEngine(TransformerLMAdapter(model, context_length=cfg["context_length"], eos_token_id=-1,
                                        vocab_size=cfg["vocab_size"], use_kv_cache=True))
    with torch.no_grad():
        toks, pasts = eng.prefill_batch(prompts)
        agree, worst = 0, 0.0
        for i, p in enumerate(prompts):
            t1, past1 = eng.prefill(torch.tensor([p], device=dev), max_length=len(p) + 8)
            agree += int(toks[i] == int(t1))
            for (k, v), (k1, v1) in zip(pasts[i], past1):
                scale = max(float(k1.abs().max()), float(v1.abs().max()), 1e-6)
                worst = max(worst, float((k - k1).abs().max()) / scale, float((v - v1).abs().max()) / scale)
    frac = agree / len(prompts)
    print(f"[A prefill ] grouped vs single: first token agrees {agree}/{len(prompts)}, worst relative K/V diff {worst:.3e}"
          f"  (gate: >= {AGREE_PREFILL:.0%} and <= {KV_REL:.0%})")
    ok &= frac >= AGREE_PREFILL and worst <= KV_REL

    # [B] compile: same batch, same tokens, eager vs compiled
    slots, max_len = args.rows + 2, min(cfg["context_length"], max(len(p) for p in prompts) + args.steps + 2)
    with torch.no_grad():
        pasts_fw, first = [], []
        for p in prompts:
            logits, past = model.forward_with_cache(torch.tensor([p], device=dev))
            pasts_fw.append(past)
            first.append(int(logits[0, -1].argmax()))

    def mk(compiled: bool):
        b = PreallocDecodeBatch(model, slots, max_len, dtype, dev, impl=args.impl, compile_decode=compiled,
                                compile_backend=args.compile_backend)
        for i, past in enumerate(pasts_fw):
            b.add(i, past)
        return b

    if dev == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    be, bc = mk(False), mk(True)
    if dev == "cuda":
        torch.cuda.synchronize()
        base_mem = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
    tok, worst, agree, total = dict(enumerate(first)), 0.0, 0, 0
    for _ in range(args.steps):
        le, lc = be.step(tok), bc.step(tok)
        for i in tok:
            worst = max(worst, float((le[i].float() - lc[i].float()).abs().max()))
            agree += int(le[i].argmax() == lc[i].argmax())
            total += 1
        tok = {i: int(le[i].argmax()) for i in le}
    kv_worst = max(max(float((a.float() - b.float()).abs().max()) for a, b in ((ke, kc), (ve, vc)))
                   for ke, kc, ve, vc in zip(be.k, bc.k, be.v, bc.v))
    tol = TOL[args.dtype]
    print(f"[B compile ] compiled vs eager decode, {args.steps} teacher-forced steps x {len(tok)} rows: max|logit diff| {worst:.3e}, "
          f"argmax agreement {agree}/{total}, max|K/V cache diff| {kv_worst:.3e}  (gate: <= {tol} and >= {AGREE_DECODE:.0%})")
    ok &= worst <= tol and agree / total >= AGREE_DECODE and kv_worst <= tol
    if dev == "cuda":
        cache_bytes = bc.k[0].numel() * bc.k[0].element_size()
        extra = torch.cuda.max_memory_allocated() - base_mem
        print(f"[B compile ] peak extra memory while stepping: {extra / 2**20:.1f} MiB (one layer's K cache = {cache_bytes / 2**20:.1f} MiB; "
              f"a value near or above that means the compiled graph is COPYING the cache instead of writing it in place)")
    print(f"stats compiled: {bc.stats}")
    print("STAGE_C_CHECK " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
