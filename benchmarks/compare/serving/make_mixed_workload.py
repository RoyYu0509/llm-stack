"""Build a closed-loop workload with MIXED output lengths from an existing prompt workload.

Why: with one fixed output length, a closed loop with N=all degenerates into lock-step static batching -- every
request of a wave finishes in the same round and the next C arrive together -- so scheduler results (N sweep,
TTFT, ITL) are properties of that synchronisation, not of the scheduler. Lengths are drawn once from `--lengths`
(uniform, seeded) so engine, HF and SGLang all see the same prompt AND the same output length per request.
"""
import argparse
import json
import random

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True, help="workload json with 'prompts' (e.g. workloads/workload_wiki128.json)")
ap.add_argument("--n", type=int, default=4096)
ap.add_argument("--lengths", type=int, nargs="+", default=[32, 48, 64, 80, 96])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()
base = json.load(open(a.base))
rng = random.Random(a.seed)
prompts = [list(base["prompts"][i % len(base["prompts"])]) for i in range(a.n)]
lens = [rng.choice(a.lengths) for _ in range(a.n)]
wl = dict(prompts=prompts, new_tokens=int(round(sum(lens) / len(lens))), new_tokens_list=lens,
          meta=dict(source=a.base, n=a.n, lengths=a.lengths, seed=a.seed, mean_new_tokens=sum(lens) / len(lens),
                    note="prompts cycled from the base workload; output length per request drawn uniformly from lengths"))
json.dump(wl, open(a.out, "w"))
print(f"{a.n} requests, mean new_tokens {sum(lens) / len(lens):.1f}, prompt len {min(map(len, prompts))}-{max(map(len, prompts))} -> {a.out}")
