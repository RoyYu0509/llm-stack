"""Record ONE arm of the inference demo video: per-request, per-token timestamps from the real engine.

The video shows three windows side by side. They are NOT three engines running at the same
time -- the engine is a single-threaded asyncio loop and a forward blocks it, so concurrent
engines on one GPU would time each other. Instead each arm is recorded alone (same model,
same prompts, same seed, same GPU) and the video replays the three recordings on one shared
clock. Everything on screen is therefore a measurement, not a simulation.

Arms (the video's three windows):
    W1  --attn naive --mode seq     naive attention, KV cache, one request at a time
    W2  --attn flash --mode seq     FlashAttention, KV cache, one request at a time
    W3  --attn flash --mode cb      FlashAttention, KV cache, continuous batching
(--attn naive --mode cb is the fallback for W3 when FlashAttention can't do the masked
ragged-decode path yet; the window must then be labelled "naive + CB".)

Honesty guards, each of which exists because the opposite silently produced a wrong video:
  * `--attn flash` aborts if the flash kernel was never called. `forward_with_cache` once
    hard-coded the naive kernel, so a "flash" arm would have run naive and been labelled flash.
  * `--attn flash --mode cb` additionally aborts if no call carried an attn_mask: the ragged
    decode path must really go through the kernel.
  * a warning if the Triton autotune cache has grown with the number of distinct KV lengths
    (autotune key contains N_KEYS => every decode step re-tunes => the timings are garbage).
  * `compare` mode checks that the arms generated the same tokens (greedy decode, so any
    divergence is numerics, and then the windows are no longer the same workload).

Usage (from serving/):
    # smoke, CPU, random prompts, tiny model
    uv run --no-sync python benchmarks/record_demo.py --preset small --synthetic \\
        --attn naive --mode cb --arm-name smoke_cb --out /tmp/demo_smoke_cb.json
    # real recording on the GPU box
    python benchmarks/record_demo.py --config ../training/configs_sft/lr1e5_replay.json \\
        --checkpoint <ckpt.pt> --prompts-npy <wiki_valid.npy> --device cuda \\
        --attn flash --mode cb --arm-name w3 --out demo_w3.json
    python benchmarks/record_demo.py compare demo_w1.json demo_w2.json demo_w3.json
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import torch

SERVING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVING))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # arrivals.py
for _p in (SERVING.parent / "training" / "cs336-basics", SERVING.parent / "training"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.append(str(_p))

from inferenceLM.data.request import RequestData  # noqa: E402
from inferenceLM.data.request_status import RequestStatus  # noqa: E402
from inferenceLM.data.tokenized_data import TokenizedData  # noqa: E402

NO_EOS = -1
GPT2_EOT = 50256

PRESETS = {  # same shapes as cb_benchmark.py
    "190m": dict(vocab_size=50257, context_length=1024, num_layers=16, d_model=768, num_heads=12,
                 d_ff=2048, rope_theta=10000.0),
    "small": dict(vocab_size=8192, context_length=512, num_layers=4, d_model=256, num_heads=8,
                  d_ff=688, rope_theta=10000.0),
    # larger random-init shapes for the KV-cache size sweep (run_kv_sweep.sh): head_dim 128, MHA like the 190m model, so the
    # KV cache costs 2 * layers * d_model * 2 bytes per token (bf16): 192 KiB / 320 KiB / 512 KiB. context_length is overridden per run.
    "1.4b": dict(vocab_size=50257, context_length=1024, num_layers=24, d_model=2048, num_heads=16,
                 d_ff=5632, rope_theta=10000.0),
    "2.8b": dict(vocab_size=50257, context_length=1024, num_layers=32, d_model=2560, num_heads=20,
                 d_ff=6912, rope_theta=10000.0),
    "6.9b": dict(vocab_size=50257, context_length=1024, num_layers=32, d_model=4096, num_heads=32,
                 d_ff=11008, rope_theta=10000.0),
}


# ----------------------------------------------------------------------------------------
# Model and attention kernel
# ----------------------------------------------------------------------------------------


class AttnCounter:
    """Wraps an attention kernel so the run can prove which kernel actually executed."""

    def __init__(self, fn):
        self.fn, self.calls, self.mask_calls = fn, 0, 0
        self.__name__ = getattr(fn, "__name__", "attention")

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if kwargs.get("attn_mask") is not None:
            self.mask_calls += 1
        return self.fn(*args, **kwargs)


def _vectorised_posencod_init(self, theta, d_k, max_seq_len, device=None):
    """Same `Rs` table as PosEncod.__init__ (a (max_seq_len, d_k/2, 2, 2) float32 buffer of 2x2 rotations), built in a few tensor ops
    instead of max_seq_len * d_k/2 separate torch.stack calls. The original loop costs ~190 us per entry on the GPU box: 2.5 minutes of
    pure Python for a 12k-token context, once per recording process."""
    torch.nn.Module.__init__(self)
    self.d_k = d_k
    self.rot_dim = d_k if d_k % 2 == 0 else d_k + 1
    half = self.rot_dim // 2
    pos = torch.arange(max_seq_len, dtype=torch.float32, device=device)[:, None]
    exps = torch.tensor([(2 * k) / self.rot_dim for k in range(half)], dtype=torch.float32, device=device)
    ang = pos / (torch.tensor(theta, dtype=torch.float32, device=device) ** exps)[None, :]
    c, s = torch.cos(ang), torch.sin(ang)
    self.register_buffer("Rs", torch.stack([torch.stack([c, -s], dim=-1), torch.stack([s, c], dim=-1)], dim=-2), persistent=False)


def build_model(model_cfg: Dict, device: str, seed: int, checkpoint: Optional[str], attn_fn,
                dtype: torch.dtype = torch.float32, fast_rope_init: bool = False):
    from cs336_basics.lm import TransformerLM
    from cs336_basics.transfromer import positionalNencoding as pe

    torch.manual_seed(seed)
    orig_init = pe.PosEncod.__init__
    if fast_rope_init:
        pe.PosEncod.__init__ = _vectorised_posencod_init
    try:
        model = TransformerLM(
            vocab_size=model_cfg["vocab_size"], context_length=model_cfg["context_length"],
            num_layers=model_cfg["num_layers"], d_model=model_cfg["d_model"], heads_num=model_cfg["num_heads"],
            d_ff=model_cfg["d_ff"], theta=model_cfg.get("rope_theta", 10000.0), device=device,
            dtype=dtype, attention_fn=attn_fn,
        )
    finally:
        pe.PosEncod.__init__ = orig_init
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device)
        model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()})
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def sdpa_kernel_fn(q, k, v, is_causal=False, attn_mask=None):
    """PyTorch SDPA with the packed (B*H, S, D) layout used by the engine. bool mask: True = attend
    (same convention as the naive kernel). Unmasked causal prefill -> FlashAttention-2 kernel;
    masked ragged decode -> memory-efficient kernel (flash does not take an arbitrary mask)."""
    import torch.nn.functional as F

    if is_causal and attn_mask is None and not (q.stride(-1) == k.stride(-1) == v.stride(-1) == 1):
        # The model's einsum projections leave the head dim with a stride > 1 (q/k 32, v 16384 at S = 16384). Every fused SDPA kernel
        # needs stride 1 there, so without this copy SDPA silently falls back to the math kernel and materialises the whole
        # (heads, S, S) fp32 score matrix: 32 GiB at S = 16384. Only the full-sequence causal call needs it; decode scores are tiny.
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    m = attn_mask.unsqueeze(0) if attn_mask is not None else None
    o = F.scaled_dot_product_attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), attn_mask=m,
                                       is_causal=bool(is_causal) and m is None)
    return o.squeeze(0)


def fa2_prefill_fn(q, k, v, is_causal=False, attn_mask=None):
    """FlashAttention-2 (flash_attn.flash_attn_func) for the unmasked causal prefill; packed (B*H, S, D) is
    presented as batch=B*H, 1 head. There is deliberately NO masked path: fa2 decode never builds a mask, it
    goes through flash_attn_with_kvcache against the pre-allocated cache (see PreallocDecodeBatch)."""
    from flash_attn import flash_attn_func

    if attn_mask is not None:
        raise RuntimeError("fa2 arm got a masked attention call: the ragged decode path must not run here")
    o = flash_attn_func(q.unsqueeze(2), k.unsqueeze(2), v.unsqueeze(2),
                        causal=bool(is_causal) and q.shape[-2] == k.shape[-2])
    return o.squeeze(2)


def get_attention(name: str):
    from cs336_basics.transfromer.scaled_dot_prod_attention import ATTENTION_KERNEL_REGISTRY

    if name == "sdpa":
        return AttnCounter(sdpa_kernel_fn)
    if name == "fa2":
        return AttnCounter(fa2_prefill_fn)

    key = {"naive": "scaled_dot_prod_attention", "flash": "flash_attention_triton"}[name]
    if key not in ATTENTION_KERNEL_REGISTRY:
        raise SystemExit(f"attention kernel {key!r} is not registered (flash needs CUDA + Triton)")
    return AttnCounter(ATTENTION_KERNEL_REGISTRY[key])


def autotune_cache_size() -> Optional[int]:
    """Number of (N_QUERIES, N_KEYS, D) configurations the Triton kernel has tuned, if observable."""
    try:
        from cs336_systems.FlashAttention import flash_attention_triton as m

        return len(m.flash_fwd_kernel.cache)
    except Exception:
        return None


# ----------------------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------------------


def load_prompts(npy_path: Optional[str], n: int, lo: int, hi: int, seed: int, vocab: int) -> List[List[int]]:
    """n prompts, lengths uniform in [lo, hi]. Real text: prefixes of whole documents from a
    GPT-2-tokenised stream (documents separated by <|endoftext|>); --synthetic: random ids."""
    rng = random.Random(seed)
    lengths = [rng.randint(lo, hi) for _ in range(n)]
    if npy_path is None:
        return [[rng.randrange(vocab) for _ in range(L)] for L in lengths]
    import numpy as np

    arr = np.load(npy_path, mmap_mode="r")
    ends = np.flatnonzero(np.asarray(arr) == GPT2_EOT)
    starts = np.concatenate([[0], ends[:-1] + 1])
    docs = [(int(s), int(e)) for s, e in zip(starts, ends) if e - s >= hi]
    if len(docs) < n:
        raise SystemExit(f"only {len(docs)} documents >= {hi} tokens in {npy_path}; need {n}")
    chosen = rng.sample(docs, n)
    return [[int(t) for t in arr[s:s + L]] for (s, _), L in zip(chosen, lengths)]


# ----------------------------------------------------------------------------------------
# One recorded trial (the closed-loop client and the polling are the same as cb_benchmark.py;
# the difference is that every token's timestamp is kept instead of being reduced to metrics)
# ----------------------------------------------------------------------------------------


async def record_trial(model, adapters, mode: str, concurrency: int, reqs,
                       max_prefills: Optional[int] = None, decode_backend: str = "ragged",
                       batched_sampling: bool = False, last_logits: bool = False,
                       prefill_batch_tokens: Optional[int] = None, compile_decode: bool = False,
                       compile_backend: str = "inductor", arrivals: Optional[List[float]] = None,
                       no_kv_cache: bool = False) -> Dict:
    """`arrivals=None`: closed loop at `concurrency`. `arrivals=[t0, t1, ...]` (seconds from the start): OPEN loop --
    request i arrives at arrivals[i] whether or not the engine keeps up; `concurrency` is then the cap on requests
    handed to the engine at once (its slot count, like max_running_requests); arrived requests above the cap wait in
    FIFO order and that wait counts as latency, because a request's arrival time is its scheduled time."""
    queue: asyncio.Queue = asyncio.Queue()
    store: Dict[str, RequestData] = {}
    engine = adapters.build_engine(model, queue, store, NO_EOS)
    engine.max_prefills_per_round = max_prefills  # N prefills per round (None = all admitted)
    adapter = engine.lm_engine.model
    adapter.use_kv_cache = not no_kv_cache   # build_engine hard-codes True; False = recompute the whole prefix per token
    adapter.decode_backend = decode_backend
    adapter.last_token_logits_only = last_logits
    engine.prefill_batch_tokens = prefill_batch_tokens   # None = one prefill forward per request
    adapter.compile_decode, adapter.compile_backend = compile_decode, compile_backend
    if decode_backend == "prealloc":  # slots allocated ONCE per run(): every in-flight request needs one
        adapter.decode_slots = concurrency + 2
        adapter.decode_max_len = min(model.context_length, max(len(p) + n for _, p, n in reqs) + 1)
    engine.lm_engine.batched_sampling = batched_sampling
    forwards = [0]
    hook = model.head.register_forward_hook(lambda *_: forwards.__setitem__(0, forwards[0] + 1))
    cache_calls, orig_fwc = [0], model.forward_with_cache   # proves which path ran: a KV-cache arm that never called it, or a no-cache arm that did, is mislabelled

    def counted_fwc(*a, **k):
        cache_calls[0] += 1
        return orig_fwc(*a, **k)

    model.forward_with_cache = counted_fwc
    task = engine.run()

    todo, gate, in_engine = deque(reqs), deque(), set()
    waiting: deque = deque()   # open loop: arrived, not yet handed to the engine (cap reached)
    arr_i = 0
    sent_at: Dict[str, float] = {}
    tok_at: Dict[str, List[float]] = {}
    outputs: Dict[str, List[int]] = {}

    def to_engine(rid, prompt, n_new):
        store[rid] = RequestData(request_id=rid, user_id="demo", timestamp=time.time(),
                                 prompt_text="", max_token_length=len(prompt) + n_new)
        queue.put_nowait(TokenizedData(request_id=rid, tokens=list(prompt)))
        in_engine.add(rid)

    def client_send():
        rid, prompt, n_new = todo.popleft()
        sent_at[rid], tok_at[rid] = time.perf_counter(), []
        (to_engine if mode == "cb" else lambda *a: gate.append(a))(rid, prompt, n_new)

    try:
        if arrivals is not None and len(arrivals) != len(reqs):
            raise ValueError(f"{len(arrivals)} arrival times for {len(reqs)} requests")
        if arrivals is None:
            for _ in range(min(concurrency, len(todo))):
                client_send()
        t_start = time.perf_counter()
        done = 0
        dbg = os.environ.get("RECORD_DEBUG_MEM")   # print allocator state while the trial runs (capacity debugging)
        poll = 0
        while done < len(reqs):
            if mode == "seq" and not in_engine and gate:
                to_engine(*gate.popleft())
            now = time.perf_counter()
            if arrivals is not None:
                while arr_i < len(arrivals) and arrivals[arr_i] <= now - t_start:
                    rid, prompt, n_new = todo.popleft()
                    sent_at[rid], tok_at[rid] = t_start + arrivals[arr_i], []   # the SCHEDULED instant, not when this poll noticed it
                    waiting.append((rid, prompt, n_new))
                    arr_i += 1
                while waiting and len(in_engine) < concurrency:
                    (to_engine if mode == "cb" else lambda *a: gate.append(a))(*waiting.popleft())
            poll += 1
            if dbg and torch.cuda.is_available() and poll % 25 == 0:
                st_ = engine._batch.stats if getattr(engine._batch, "stats", None) else {}
                print(f"[mem] poll {poll} done {done} in_flight {len(engine._in_flight)} fifo {len(engine._prefill_fifo)} "
                      f"batch {len(engine._batch) if engine._batch is not None else 0} "
                      f"alloc {torch.cuda.memory_allocated() / 2**30:.2f} GiB reserved {torch.cuda.memory_reserved() / 2**30:.2f} GiB "
                      f"alloc_retries {torch.cuda.memory_stats().get('num_alloc_retries', 0)} ooms {torch.cuda.memory_stats().get('num_ooms', 0)} "
                      f"{st_}", flush=True)
            for rid in list(in_engine):
                seen = adapters.generated_so_far(engine, store, rid)
                tok_at[rid].extend([now] * (len(seen) - len(tok_at[rid])))
                if store[rid].status in (RequestStatus.DONE, RequestStatus.FAILED):
                    if store[rid].status == RequestStatus.FAILED:
                        raise RuntimeError(f"request {rid} FAILED")
                    outputs[rid] = list(store[rid].generated_tokens)
                    in_engine.discard(rid)
                    done += 1
                    if todo and arrivals is None:
                        client_send()
            await asyncio.sleep(0)
        t_end = time.perf_counter()
        await asyncio.wait_for(queue.join(), timeout=60)
        await asyncio.wait_for(engine.shutdown(draining=True), timeout=60)
        await asyncio.wait_for(task, timeout=10)
    finally:
        hook.remove()
        model.forward_with_cache = orig_fwc
    kv_stats = dict(engine._batch.stats) if getattr(engine._batch, "stats", None) else None
    return dict(sent_at=sent_at, tok_at=tok_at, outputs=outputs, t_end=t_end, forwards=forwards[0],
                kv_stats=kv_stats, prefill_group_sizes=list(engine.prefill_group_sizes),
                cache_calls=cache_calls[0])


def to_record(trial: Dict, reqs, arm_meta: Dict) -> Dict:
    """Shift all times so the first request is sent at t=0 and package the per-request tokens."""
    t0 = min(trial["sent_at"].values())
    out_reqs, ttfts = [], []
    for rid, prompt, _ in reqs:
        ts, ids = trial["tok_at"][rid], trial["outputs"][rid]
        assert len(ts) == len(ids), f"{rid}: {len(ts)} timestamps for {len(ids)} tokens"
        sent = trial["sent_at"][rid] - t0
        ttfts.append(ts[0] - trial["sent_at"][rid])
        out_reqs.append(dict(id=rid, prompt_len=len(prompt), prompt_ids=prompt, submit_t=sent,
                             tokens=[[round(t - t0, 6), tok] for t, tok in zip(ts, ids)]))
    n_tok = sum(len(r["tokens"]) for r in out_reqs)
    makespan = trial["t_end"] - t0
    stats = dict(makespan_s=makespan, generated_tokens=n_tok, throughput_tok_s=n_tok / makespan,
                 ttft_p50_s=statistics.median(ttfts), ttft_mean_s=statistics.fmean(ttfts),
                 forwards=trial["forwards"])
    return dict(meta={**arm_meta, **stats}, requests=out_reqs)


# ----------------------------------------------------------------------------------------
# Provenance
# ----------------------------------------------------------------------------------------


def _git(args: List[str], cwd: Path) -> str:
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return ""


def provenance() -> Dict:
    """The CB code was uncommitted when this was written, so a sha alone does not identify it:
    also hash the working-tree diff of the code that determines what ran."""
    diff = _git(["diff", "HEAD", "--", "serving/inferenceLM", "training/cs336-basics/cs336_basics",
                 "training/cs336_systems/FlashAttention"], SERVING.parent)
    untracked = sorted(_git(["ls-files", "--others", "--exclude-standard", "serving/inferenceLM"],
                            SERVING.parent).split())
    h = hashlib.sha256(diff.encode())
    for rel in untracked:
        p = SERVING.parent / rel
        if p.is_file():
            h.update(p.read_bytes())
    return dict(git_sha=_git(["rev-parse", "--short", "HEAD"], SERVING).strip(),
                worktree_diff_sha256=h.hexdigest()[:16], untracked_files=untracked)


# ----------------------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------------------


def load_adapters(path: Path):
    spec = importlib.util.spec_from_file_location("m2_adapters", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def record(args) -> None:
    if args.device == "cuda" and not args.no_tf32:
        torch.set_float32_matmul_precision("high")  # same setting for every arm, so numerics differ only by kernel
    if args.attn == "flash" and args.device != "cuda":
        raise SystemExit("--attn flash needs --device cuda (the Triton kernel has no CPU/MPS backend)")
    model_cfg = PRESETS[args.preset] if args.preset else json.load(open(args.config))["model"]
    if args.context_length:   # prompt + new tokens must fit; only the RoPE table and the n_positions check depend on it
        model_cfg = {**model_cfg, "context_length": args.context_length}
    attn = get_attention(args.attn)
    model = build_model(model_cfg, args.device, args.seed, args.checkpoint, attn,
                        dtype=getattr(torch, args.dtype), fast_rope_init=args.fast_rope_init)
    adapters = load_adapters(SERVING / "tests" / "m2_adapters.py")

    arrivals = None
    if args.arrival_rate:
        if args.mode != "cb":
            raise SystemExit("--arrival-rate (open loop) needs --mode cb")
        from arrivals import arrival_times, n_for_duration

        args.n_requests = n_for_duration(args.arrival_rate, args.duration)
        arrivals = arrival_times(args.arrival_rate, args.n_requests, args.seed, args.arrival_cv)
    if args.no_kv_cache and (args.mode != "seq" or args.attn == "fa2"):
        raise SystemExit("--no-kv-cache recomputes the whole prefix per token: it is the --mode seq baseline "
                         "(one request at a time) and cannot use the fa2 KV-cache kernel")
    n_new = [args.new_tokens] * args.n_requests
    if args.prompts_json:  # same token ids (and output lengths) as the HF / SGLang closed-loop workload
        wl = json.load(open(args.prompts_json))
        base = wl["prompts"]
        prompts = [list(base[i % len(base)]) for i in range(args.n_requests)]
        if wl.get("new_tokens_list"):
            nl = wl["new_tokens_list"]
            n_new = [int(nl[i % len(nl)]) for i in range(args.n_requests)]
    else:
        prompts = load_prompts(None if args.synthetic else args.prompts_npy, args.n_requests,
                               args.prompt_len[0], args.prompt_len[1], args.seed, model_cfg["vocab_size"])
    reqs = [(f"r{i:02d}", p, n_new[i]) for i, p in enumerate(prompts)]
    if args.attn == "fa2" and args.dtype == "float32":
        raise SystemExit("--attn fa2 needs --dtype bfloat16 or float16 (FlashAttention has no fp32 kernel)")
    if (args.compile_decode or args.prefill_batch_tokens) and args.mode != "cb":
        raise SystemExit("--compile-decode / --prefill-batch-tokens apply to the continuous-batching engine (--mode cb)")
    if args.compile_decode and args.attn != "fa2":
        raise SystemExit("--compile-decode compiles the pre-allocated-cache decode path: it needs --attn fa2")
    trial_kw = dict(decode_backend="prealloc" if args.attn == "fa2" else "ragged",
                    batched_sampling=args.batched_sampling, last_logits=args.prefill_last_logits,
                    prefill_batch_tokens=args.prefill_batch_tokens, compile_decode=args.compile_decode,
                    compile_backend=args.compile_backend, no_kv_cache=args.no_kv_cache)
    # untimed warm-up with the SAME shapes as the timed run: real prompts (256-900 tokens), full output length,
    # same concurrency, same N. It grows the caching allocator to the working-set size, picks cuBLAS algorithms
    # for the real GEMM shapes and compiles / tunes whatever is lazy, so the timed trial measures steady state.
    n_warm = args.warmup_requests if args.warmup_requests is not None else min(len(reqs), 2 * args.concurrency)
    warm = [(f"w{i:03d}", p, n) for i, (_, p, n) in enumerate(reqs[:n_warm])]
    warm_error = None
    if warm:
        try:
            asyncio.run(record_trial(model, adapters, args.mode, args.concurrency, warm, args.max_prefills_per_round,
                                     **trial_kw))
        except Exception as e:  # noqa: BLE001
            warm_error = e  # reported below as a status=OOM/FAILED record, same as a failing timed trial

    def meta_for(rep_i: int) -> Dict:
        return dict(arm=args.arm_name, rep=rep_i, status="ok", warmup_requests=len(warm),
                    attn=args.attn, dtype=args.dtype, mode=args.mode, concurrency=args.concurrency,
                    max_prefills_per_round=args.max_prefills_per_round,
                    n_requests=args.n_requests, new_tokens=args.new_tokens, prompt_len_range=args.prompt_len,
                    seed=args.seed, device=args.device, model_cfg=model_cfg, checkpoint=args.checkpoint,
                    prompts_npy=args.prompts_npy, prompts_json=args.prompts_json, synthetic=args.synthetic,
                    attn_calls=attn.calls, attn_mask_calls=attn.mask_calls,
                    decode_backend=trial_kw["decode_backend"], batched_sampling=args.batched_sampling,
                    prefill_last_logits=args.prefill_last_logits,
                    prefill_batch_tokens=args.prefill_batch_tokens, compile_decode=args.compile_decode,
                    compile_backend=args.compile_backend,
                    new_tokens_mixed=len(set(n for _, _, n in reqs)) > 1,
                    kv_cache=not args.no_kv_cache, open_loop=arrivals is not None,
                    arrival_rate=args.arrival_rate, arrival_cv=args.arrival_cv if arrivals else None,
                    arrival_duration_s=args.duration if arrivals else None,
                    autotune_cache_entries=autotune_cache_size(),
                    matmul_precision=torch.get_float32_matmul_precision(), torch=torch.__version__,
                    host=platform.node(),
                    gpu_name=torch.cuda.get_device_name(0) if args.device == "cuda" else "",
                    **provenance())

    out = Path(args.out)
    if warm_error is not None:
        ooms = torch.cuda.memory_stats().get("num_ooms", 0) if args.device == "cuda" else 0
        is_oom = ooms > 0 or "out of memory" in str(warm_error).lower()
        meta = {**meta_for(0), "status": "OOM" if is_oom else "FAILED", "phase": "warmup",
                "error": str(warm_error)[:300],
                "peak_mem_gb": (torch.cuda.max_memory_allocated() / 2**30) if args.device == "cuda" else None}
        path = out if args.reps == 1 else out.with_name(f"{out.stem}_r1{out.suffix}")
        path.write_text(json.dumps(dict(meta=meta, requests=[])))
        print(f"[{args.arm_name}] warm-up: {meta['status']} (peak {meta['peak_mem_gb']} GiB): {meta['error']}")
        raise SystemExit(3 if is_oom else 4)
    for rep_i in range(1, args.reps + 1):
        path = out if args.reps == 1 else out.with_name(f"{out.stem}_r{rep_i}{out.suffix}")
        attn.calls = attn.mask_calls = 0
        if args.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        prof = None
        if args.cprofile and rep_i == 1:   # only the timed trial: imports, model build and warm-up stay out of the profile
            import cProfile

            prof = cProfile.Profile()
            prof.enable()
        try:
            trial = asyncio.run(record_trial(model, adapters, args.mode, args.concurrency, reqs,
                                             args.max_prefills_per_round, arrivals=arrivals, **trial_kw))
            if prof:
                prof.disable()
                prof.dump_stats(args.cprofile)
        except Exception as e:  # noqa: BLE001  -- OOM is a RESULT of the scan, not a crash
            if prof:
                prof.disable()
            ooms = torch.cuda.memory_stats().get("num_ooms", 0) if args.device == "cuda" else 0
            is_oom = ooms > 0 or "out of memory" in str(e).lower()
            meta = {**meta_for(rep_i), "status": "OOM" if is_oom else "FAILED", "error": str(e)[:300],
                    "peak_mem_gb": (torch.cuda.max_memory_allocated() / 2**30) if args.device == "cuda" else None}
            path.write_text(json.dumps(dict(meta=meta, requests=[])))
            print(f"[{args.arm_name}] rep {rep_i}: {meta['status']} (peak {meta['peak_mem_gb']} GiB): {meta['error']}")
            raise SystemExit(3 if is_oom else 4)

        guard(args, attn, trial)
        meta = meta_for(rep_i)
        meta["kv_stats"] = trial.get("kv_stats")
        sizes = trial.get("prefill_group_sizes") or []
        meta["prefill_forwards"] = len(sizes)
        meta["prefill_group_mean"] = (sum(sizes) / len(sizes)) if sizes else None
        meta["prefill_group_max"] = max(sizes) if sizes else None
        if args.device == "cuda":
            meta["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
        rec = to_record(trial, reqs, meta)
        path.write_text(json.dumps(rec))
        m = rec["meta"]
        print(f"[{args.arm_name}] rep {rep_i}: {m['generated_tokens']} tokens in {m['makespan_s']:.2f}s = "
              f"{m['throughput_tok_s']:.1f} tok/s, TTFT p50 {m['ttft_p50_s']:.3f}s, forwards {m['forwards']}, "
              f"peak {m.get('peak_mem_gb', 0):.1f} GiB -> {path}")


def guard(args, attn: AttnCounter, trial: Optional[Dict] = None) -> None:
    if args.attn == "fa2":
        steps = (trial or {}).get("kv_stats", {}) or {}
        if attn.calls == 0:
            raise SystemExit("ABORT: --attn fa2 but flash_attn_func was never called for prefill.")
        if attn.mask_calls:
            raise SystemExit("ABORT: --attn fa2 but a masked attention call happened (ragged path was used).")
        if not steps.get("steps"):
            raise SystemExit("ABORT: --attn fa2 but PreallocDecodeBatch never ran a decode step: this arm would "
                             "have been recorded as fa2 while running the ragged path.")
    calls = (trial or {}).get("cache_calls")
    if calls is not None and args.no_kv_cache and calls:
        raise SystemExit(f"ABORT: --no-kv-cache but forward_with_cache was called {calls} times: this arm would have "
                         "been recorded as 'no KV cache' while using one.")
    if calls is not None and not args.no_kv_cache and calls == 0:
        raise SystemExit("ABORT: a KV-cache arm never called forward_with_cache.")
    if args.compile_decode and not ((trial or {}).get("kv_stats") or {}).get("compiled"):
        raise SystemExit("ABORT: --compile-decode but the decode batch was not compiled: this arm would have been "
                         "recorded as compiled while running eager.")
    if args.attn == "flash" and attn.calls == 0:
        raise SystemExit("ABORT: --attn flash but the flash kernel was never called. forward_with_cache "
                         "(multiheads_attention.py) still hard-codes the naive kernel; this arm would "
                         "have been recorded as 'flash' while running naive.")
    if args.attn == "flash" and args.mode == "cb" and attn.mask_calls == 0:
        raise SystemExit("ABORT: flash + cb but no attention call carried attn_mask: the ragged batched "
                         "decode did not go through the flash kernel.")
    n = autotune_cache_size()
    if args.attn == "flash" and n is not None and n > 32:
        print(f"WARNING: Triton autotune cache has {n} entries -- the autotune key probably includes the "
              "KV length, so decode steps were re-tuned and these timings are NOT valid.", file=sys.stderr)


def compare(paths: List[str]) -> None:
    recs = [json.load(open(p)) for p in paths]
    base = {r["id"]: r for r in recs[0]["requests"]}
    for rec in recs:
        prompts_ok = all(base[r["id"]]["prompt_ids"] == r["prompt_ids"] for r in rec["requests"])
        same = diverged = 0
        first_div = []
        for r in rec["requests"]:
            a = [t for _, t in base[r["id"]]["tokens"]]
            b = [t for _, t in r["tokens"]]
            if a == b:
                same += 1
            else:
                diverged += 1
                first_div.append(next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b))))
        m = rec["meta"]
        print(f"{m['arm']:<8} prompts_identical={prompts_ok}  identical_outputs={same}/{same + diverged}"
              + (f"  first divergence at token {sorted(first_div)}" if diverged else "")
              + f"  {m['throughput_tok_s']:.1f} tok/s")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        return compare(sys.argv[2:])
    ap = argparse.ArgumentParser()
    ap.add_argument("--cprofile", default=None, help="dump a cProfile of the FIRST timed trial here (that run is slower: never a throughput row)")
    ap.add_argument("--prefill-last-logits", action="store_true",
                    help="prefill computes the LM head for the last position only (SGLang / HF generate do the same)")
    ap.add_argument("--prefill-batch-tokens", type=int, default=None,
                    help="cb: pack the requests prefilled in a round into right-padded groups of at most this many "
                         "PADDED tokens per forward (default: one forward per request, the original behaviour)")
    ap.add_argument("--compile-decode", action="store_true",
                    help="fa2 arm: run the decode forward as one torch.compile'd graph (needs the GPU box for Inductor)")
    ap.add_argument("--compile-backend", default="inductor", help="torch.compile backend (aot_eager = trace only)")
    ap.add_argument("--batched-sampling", action="store_true",
                    help="ragged decode: one batched argmax + one tolist() per step instead of one .item() per row")
    ap.add_argument("--no-kv-cache", action="store_true",
                    help="seq baseline without a KV cache: every token re-runs the whole prefix (adapter use_kv_cache=False)")
    ap.add_argument("--arrival-rate", type=float, default=None,
                    help="OPEN loop: requests arrive at this mean rate (req/s) whether or not the engine keeps up; "
                         "--concurrency is then the cap on requests inside the engine at once; --n-requests is ignored")
    ap.add_argument("--arrival-cv", type=float, default=1.0,
                    help="open loop: coefficient of variation of the inter-arrival gaps (1 = Poisson, >1 = bursty)")
    ap.add_argument("--duration", type=float, default=30.0, help="open loop: length of the arrival schedule in seconds")
    ap.add_argument("--attn", choices=["naive", "flash", "sdpa", "fa2"], required=True,
                    help="sdpa = torch.nn.functional.scaled_dot_product_attention (FA-2 for prefill, "
                         "mem-efficient for the masked ragged decode); flash = the self-written Triton kernel; "
                         "fa2 = flash_attn prefill + flash_attn_with_kvcache decode on a pre-allocated KV cache")
    ap.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    ap.add_argument("--mode", choices=["seq", "cb"], required=True)
    ap.add_argument("--arm-name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", choices=list(PRESETS), default=None, help="random-init model; else --config")
    ap.add_argument("--config", default=None)
    ap.add_argument("--context-length", type=int, default=None,
                    help="override the preset's context_length (random-init weights only: a checkpoint was trained at its own)")
    ap.add_argument("--fast-rope-init", action="store_true",
                    help="build the RoPE table with a few tensor ops (identical values) instead of the per-entry Python loop; "
                         "only the start-up time of a long-context model changes")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--prompts-npy", default=None, help="GPT-2-token .npy, documents split by <|endoftext|>")
    ap.add_argument("--synthetic", action="store_true", help="random-id prompts (smoke only)")
    ap.add_argument("--prompts-json", default=None,
                    help="closed_loop workload json ({'prompts': [[token ids], ...]}); cycled to --n-requests")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-requests", type=int, default=16)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--reps", type=int, default=1,
                    help="timed repetitions inside ONE process after the warm-up (files <out>_r1.json, ...)")
    ap.add_argument("--warmup-requests", type=int, default=None,
                    help="untimed warm-up requests with the real shapes (default 2 x concurrency, 0 = none)")
    ap.add_argument("--max-prefills-per-round", type=int, default=None,
                    help="N: prefills run before each batched decode (default: all admitted; cb mode only)")
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--prompt-len", type=int, nargs=2, default=[256, 900])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-tf32", action="store_true")
    args = ap.parse_args()
    if not (args.preset or args.config):
        ap.error("need --preset or --config")
    if args.context_length and args.checkpoint:
        ap.error("--context-length changes the model shape: it cannot be combined with --checkpoint")
    if not (args.synthetic or args.prompts_npy or args.prompts_json):
        ap.error("need --prompts-npy, --prompts-json or --synthetic")
    record(args)


if __name__ == "__main__":
    main()
