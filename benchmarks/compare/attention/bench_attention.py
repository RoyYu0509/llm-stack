"""A1 / C1 / C6 -- attention kernels: his implementations vs PyTorch SDPA vs the flash-attn package.

Decision record: docs/sprints/week7-discussion.md D-7, "Attention". What changed from the first harness:

  * rows (names are the ones that go on the CV / README):
        attention                  his `vectorized_torch`: torch.matmul, materialises the N x N scores
        torch_compile_attention    the same function under torch.compile
        sdpa_flash                 F.scaled_dot_product_attention forced to the FLASH backend (PyTorch built-in FA2)
        flash_attn_pkg             the standalone `flash-attn` package (`flash_attn_func`) in its natural (B, S, H, D) layout, wheel only
        triton_fa_mine             his Triton FlashAttention kernel (forward is Triton; backward is PyTorch recompute)
    his einsum `torch_naive` attention is out of the comparison (still selectable with --systems for debugging);
  * shape: 12 heads, head_dim 64, causal, **bf16**, seq 128..16384 with **B*S fixed at 32768 tokens**
    (B = 256 ... 2), so every cell does comparable "token work" and short sequences are not under-filled;
  * **one fresh process per (system, seq, mode)** -- an OOM or an allocator state from one cell cannot leak into the
    next (this is how the 8192 'vectorized' OOM mystery of the old data is settled);
  * compile / autotune / CUDA init / first call are all outside the timer (explicit warm-up calls first);
  * old data is discarded, tolerances for the correctness checks are re-derived for bf16 (see `check_tolerances`).

Modes
  (default)               timing sweep -> CSV rows (module=attention; forward_only stated on every row)
  --check                 C1: output / gradient agreement against an fp32 reference, bf16 tolerances
  --check-flash-raises    C6: does forcing the FLASH backend raise when it cannot run, and does it launch a flash kernel?
  --check-inductor        does torch.compile rewrite his attention into SDPA? (counters["inductor"]["fuse_attention"])
  --single                (internal) one cell in this process, prints one JSON line

"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from schema import ResultWriter  # noqa: E402
import manifest  # noqa: E402

REPO = HERE.parents[2]

DEFAULT_SYSTEMS = ["attention", "torch_compile_attention", "sdpa_flash", "flash_attn_pkg", "triton_fa_mine"]
EXTRA_SYSTEMS = ["sdpa_efficient", "sdpa_math", "torch_naive", "flash_attn_pkg_flat"]
ALL_SYSTEMS = DEFAULT_SYSTEMS + EXTRA_SYSTEMS
REGISTRY_KEY = {
    "torch_naive": "scaled_dot_prod_attention",
    "attention": "vectorized_torch",
    "torch_compile_attention": "vectorized_torch_compiled",
    "triton_fa_mine": "flash_attention_triton",
}
DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
DEFAULT_SEQS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
TOKENS = 32768


# --------------------------------------------------------------------------------------
# systems. Canonical layout is (B*H, S, D), as in his kernels. `flash_attn_pkg` wants (B, S, H, D):
# the conversion is done *outside* the timer (to_native), the timed call is the library call itself.
# --------------------------------------------------------------------------------------
@dataclass
class SystemSpec:
    name: str
    fn: Callable                      # fn(q, k, v, causal) on NATIVE-layout tensors
    layout: str = "flat"              # "flat": (B*H, S, D)   "bshd": (B, S, H, D)   "flat1h": (B*H, S, 1, D)

    def to_native(self, t: torch.Tensor, B: int, H: int) -> torch.Tensor:
        if self.layout == "flat":
            return t
        if self.layout == "flat1h":      # same memory as "flat": B*H rows of (S, D), presented as batch=B*H, one head
            return t.unsqueeze(2)
        _, S, D = t.shape
        return t.view(B, H, S, D).permute(0, 2, 1, 3).contiguous()

    def from_native(self, t: torch.Tensor, B: int, H: int) -> torch.Tensor:
        if self.layout == "flat":
            return t
        if self.layout == "flat1h":
            return t.squeeze(2)
        _, S, _, D = t.shape
        return t.permute(0, 2, 1, 3).reshape(B * H, S, D)


def _sdpa_fn(backend: str):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    import torch.nn.functional as F

    be = {"flash": SDPBackend.FLASH_ATTENTION, "efficient": SDPBackend.EFFICIENT_ATTENTION,
          "math": SDPBackend.MATH}[backend]

    def fn(q, k, v, is_causal):
        # (B*H, S, D) -> (1, B*H, S, D): the fused kernels want 4D; this is a view, no copy.
        # is_causal=True and NO attn_mask: passing both raises on recent versions (2.14 docs) but is silently
        # intersected on torch 2.6; an explicit mask also makes flash ineligible.
        with sdpa_kernel([be]):
            o = F.scaled_dot_product_attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
                                               is_causal=is_causal)
        return o.squeeze(0)

    return fn


def _flash_attn_pkg_fn():
    from flash_attn import flash_attn_func  # raises ImportError when the wheel is not installed

    def fn(q, k, v, is_causal):
        return flash_attn_func(q, k, v, causal=is_causal)

    return fn


def get_spec(name: str) -> SystemSpec:
    if name.startswith("sdpa_"):
        return SystemSpec(name, _sdpa_fn(name.split("_", 1)[1]))
    if name == "flash_attn_pkg":
        return SystemSpec(name, _flash_attn_pkg_fn(), layout="bshd")
    if name == "flash_attn_pkg_flat":     # diagnostic row: the same library call with the SDPA-identical memory layout
        return SystemSpec(name, _flash_attn_pkg_fn(), layout="flat1h")
    from cs336_basics.transfromer.scaled_dot_prod_attention import ATTENTION_KERNEL_REGISTRY

    return SystemSpec(name, ATTENTION_KERNEL_REGISTRY[REGISTRY_KEY[name]])


def get_system(name: str, heads: int = 12):
    """fn(q, k, v, causal) on canonical (B*H, S, D) tensors for every system (conversion included).
    Used by the correctness checks; the timing path uses `get_spec` so conversion is never timed."""
    spec = get_spec(name)
    if spec.layout == "flat":
        return spec.fn

    def fn(q, k, v, is_causal, _spec=spec, H=heads):
        n, S, D = q.shape
        B = n // H
        out = _spec.fn(_spec.to_native(q, B, H), _spec.to_native(k, B, H), _spec.to_native(v, B, H), is_causal)
        return _spec.from_native(out, B, H)

    return fn


# --------------------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------------------
def _sync(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def time_ms(fn, device: str, grad_to_none=None, reps: int = 5,
            warmup_ms: float = 50, rep_ms: float = 200,
            manual_warmup: int = 10, manual_iters: int = 50, timer: str = "do_bench"):
    """Return (mean_ms, std_ms, n_reps, timer_name). `reps` independent measurements, each the *median* of many calls.

    CUDA + triton available: triton.testing.do_bench (warmup / rep are *milliseconds*). Otherwise a plain loop
    with explicit synchronisation (CPU smoke runs). timer="cudagraph": do_bench_cudagraph, for when eager launch
    overhead is erratic (gVisor containers: ~0.12 ms per launch)."""
    do_bench = None
    if device.startswith("cuda"):
        try:
            from triton.testing import do_bench  # type: ignore
        except Exception:
            do_bench = None
    vals = []
    if timer == "cudagraph":
        from triton.testing import do_bench_cudagraph  # type: ignore

        for _ in range(reps):
            vals.append(float(do_bench_cudagraph(fn, rep=rep_ms, grad_to_none=grad_to_none, return_mode="median")))
        timer_name = f"triton.do_bench_cudagraph(rep={rep_ms}ms,median)"
    elif do_bench is not None:
        for _ in range(reps):
            vals.append(float(do_bench(fn, warmup=warmup_ms, rep=rep_ms, grad_to_none=grad_to_none,
                                       return_mode="median")))
        timer_name = f"triton.do_bench(warmup={warmup_ms}ms,rep={rep_ms}ms,median)"
    else:
        for _ in range(reps):
            for _ in range(manual_warmup):
                fn()
            _sync(device)
            t = []
            for _ in range(manual_iters):
                if grad_to_none:
                    for x in grad_to_none:
                        x.grad = None
                _sync(device)
                t0 = time.perf_counter()
                fn()
                _sync(device)
                t.append((time.perf_counter() - t0) * 1e3)
            vals.append(statistics.median(t))
        timer_name = f"manual(warmup={manual_warmup},iters={manual_iters},median)"
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return statistics.fmean(vals), std, len(vals), timer_name


def counted_tflops(ms: float, B: int, H: int, S: int, D: int, causal: bool, forward_only: bool):
    """Mathematically required FLOPs (causal counted as half), the FlashAttention-paper convention:
    fwd = 4*B*H*S^2*D (halved if causal); bwd = 2.5x fwd. This is NOT the FLOPs the hardware executed -- his Triton
    kernel does not skip masked tiles."""
    fwd = 4.0 * B * H * S * S * D * (0.5 if causal else 1.0)
    total = fwd if forward_only else fwd * 3.5
    return total / (ms * 1e-3) / 1e12


def _is_oom(e: BaseException) -> bool:
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


# --------------------------------------------------------------------------------------
# one cell, one process
# --------------------------------------------------------------------------------------
def run_single(args) -> int:
    """Measure one (system, seq, mode) and print ONE json line. Everything that is not the steady-state call --
    import, CUDA context, Triton autotune, torch.compile, cuBLAS handles, allocator warm-up -- happens before
    the timer starts."""
    device = args.device
    dtype = DTYPES[args.dtype]
    S, H, D = args.seq, args.heads, args.head_dim
    B = args.batch or max(1, args.tokens // S)
    forward_only = args.mode == "fwd"
    out = dict(system=args.system, seq=S, batch=B, mode=args.mode, status="ok")
    try:
        spec = get_spec(args.system)
        g = torch.Generator(device=device).manual_seed(0)
        flat = [torch.randn(B * H, S, D, device=device, dtype=dtype, generator=g) for _ in range(3)]
        q, k, v = [spec.to_native(t, B, H) for t in flat]
        del flat
        if forward_only:
            run, to_none = (lambda: spec.fn(q, k, v, args.causal)), None
        else:
            q.requires_grad_(); k.requires_grad_(); v.requires_grad_()
            dout = torch.randn_like(spec.fn(q, k, v, args.causal).detach())

            def run():
                spec.fn(q, k, v, args.causal).backward(dout)

            to_none = [q, k, v]
        for _ in range(args.warmup_calls):  # compile / autotune / lazy init, not timed
            run()
        _sync(device)
        ms, std, n, timer = time_ms(run, device, to_none, reps=args.reps, timer=args.timer)
        peak = float("nan")
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            run()
            _sync(device)
            peak = (torch.cuda.max_memory_allocated() - base) / 2**20
        out.update(ms=ms, std=std, n=n, timer=timer, peak_mb=peak,
                   tflops=counted_tflops(ms, B, H, S, D, args.causal, forward_only))
    except ImportError as e:
        out.update(status="unavailable", note=f"{type(e).__name__}: {e}")
    except Exception as e:  # noqa: BLE001
        if _is_oom(e):
            out.update(status="oom", note=str(e).splitlines()[0][:200])
        else:
            out.update(status="error", note=f"{type(e).__name__}: {str(e)[:300]}")
    print("CELL " + json.dumps(out), flush=True)
    return 0


def run_cell_subprocess(args, system: str, seq: int, mode: str, timeout_s: int = 900) -> dict:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--single", "--system", system, "--seq", str(seq),
           "--mode", mode, "--dtype", args.dtype, "--heads", str(args.heads), "--head-dim", str(args.head_dim),
           "--tokens", str(args.tokens), "--reps", str(args.reps), "--timer", args.timer,
           "--warmup-calls", str(args.warmup_calls), "--device", args.device,
           "--causal" if args.causal else "--non-causal"]
    if args.batch:
        cmd += ["--batch", str(args.batch)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return dict(system=system, seq=seq, mode=mode, status="timeout", note=f">{timeout_s}s")
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("CELL ")]
    if not lines:
        tail = (p.stderr or p.stdout)[-300:].replace("\n", " | ")
        return dict(system=system, seq=seq, mode=mode, status="crash", note=f"rc={p.returncode} {tail}")
    return json.loads(lines[-1][5:])


def run_sweep(args) -> int:
    w = ResultWriter(args.out, REPO)
    modes = {"fwd": ["fwd"], "fwdbwd": ["fwdbwd"], "both": ["fwd", "fwdbwd"]}[args.mode]
    H, D = args.heads, args.head_dim
    for S in args.seqs:
        for mode in modes:
            for name in args.systems:
                cell = run_cell_subprocess(args, name, S, mode)
                B = cell.get("batch") or args.batch or max(1, args.tokens // S)
                forward_only = mode == "fwd"
                row = dict(module="attention", experiment="A1", system=name,
                           variant=f"hd{D}/{H}h/B*S={args.tokens}/b{B}", dtype=args.dtype, causal=args.causal,
                           forward_only=forward_only, seq_len=S, batch=B, n_gpus=1)
                st = cell["status"]
                tag = "fwd" if forward_only else "fwd+bwd"
                if st == "ok":
                    w.write(**row, metric="latency_ms", value=round(cell["ms"], 4), unit="ms",
                            std=round(cell["std"], 4), n_reps=cell["n"], notes=cell["timer"])
                    w.write(**row, metric="tflops_counted", value=round(cell["tflops"], 3), unit="TFLOP/s",
                            n_reps=cell["n"], notes="counted FLOPs (causal halved, bwd=2.5x fwd), not executed FLOPs")
                    if cell["peak_mb"] == cell["peak_mb"]:
                        w.write(**row, metric="peak_mem_delta_mb", value=round(cell["peak_mb"], 1), unit="MiB", n_reps=1)
                    print(f"S={S:6d} B={B:4d} {tag:8s} {name:26s} {cell['ms']:9.3f} ms (±{cell['std']:.3f})  "
                          f"{cell['tflops']:6.1f} TFLOP/s", flush=True)
                elif st == "oom":
                    w.write(**row, metric="oom", value=1, unit="bool", n_reps=1, notes=cell.get("note", ""))
                    print(f"S={S:6d} B={B:4d} {tag:8s} {name:26s} OOM", flush=True)
                else:  # unavailable | error | crash | timeout: recorded, never silently dropped
                    w.write(**row, metric=st, value=1, unit="bool", n_reps=1, notes=cell.get("note", "")[:300])
                    print(f"S={S:6d} B={B:4d} {tag:8s} {name:26s} {st.upper()} {cell.get('note', '')[:100]}", flush=True)
    return 0


# --------------------------------------------------------------------------------------
# C1: correctness vs fp32 reference, bf16 tolerances
# --------------------------------------------------------------------------------------
def reference_fp32(q, k, v, is_causal: bool):
    """Plain fp32 softmax(QK^T/sqrt(d))V on (N, S, D); the trusted baseline."""
    qf, kf, vf = q.float(), k.float(), v.float()
    s = qf @ kf.transpose(-1, -2) / (qf.shape[-1] ** 0.5)
    if is_causal:
        S = s.shape[-1]
        s = s.masked_fill(~torch.tril(torch.ones(S, S, dtype=torch.bool, device=s.device)), float("-inf"))
    return torch.softmax(s, dim=-1) @ vf


def _reference_grads(base, dout, causal):
    qr, kr, vr = [t.clone().requires_grad_() for t in base]
    ref_out = reference_fp32(qr, kr, vr, causal)
    ref_out.backward(dout)
    return dict(out=ref_out.detach(), dq=qr.grad, dk=kr.grad, dv=vr.grad)


def check_correctness(systems, device: str, dtype: torch.dtype, seq: int = 512,
                      batch: int = 2, heads: int = 12, head_dim: int = 64, causal: bool = True,
                      seed: int = 0) -> dict[str, dict[str, float]]:
    """Return {system: {out, dq, dk, dv: max_abs_diff vs the fp32 reference}}."""
    g = torch.Generator(device=device).manual_seed(seed)
    base = [torch.randn(batch * heads, seq, head_dim, device=device, dtype=torch.float32, generator=g)
            for _ in range(3)]
    dout = torch.randn(batch * heads, seq, head_dim, device=device, dtype=torch.float32, generator=g)
    ref = _reference_grads(base, dout, causal)

    results = {}
    for name in systems:
        fn = get_system(name, heads)
        q, k, v = [t.to(dtype).clone().requires_grad_() for t in base]
        out = fn(q, k, v, causal)
        out.backward(dout.to(dtype))
        got = dict(out=out.detach(), dq=q.grad, dk=k.grad, dv=v.grad)
        results[name] = {key: float((got[key].float() - ref[key]).abs().max()) for key in ref}
    return results


def rounding_floor(device: str, dtype: torch.dtype, seq: int, batch: int, heads: int, head_dim: int,
                   causal: bool, seed: int = 0) -> dict[str, float]:
    """The error an *exact* computation would still show after being stored in `dtype`: the fp32 reference rounded
    to `dtype`. No bf16 kernel can beat this, so tolerances are expressed as multiples of it."""
    g = torch.Generator(device=device).manual_seed(seed)
    base = [torch.randn(batch * heads, seq, head_dim, device=device, dtype=torch.float32, generator=g)
            for _ in range(3)]
    dout = torch.randn(batch * heads, seq, head_dim, device=device, dtype=torch.float32, generator=g)
    ref = _reference_grads(base, dout, causal)
    return {key: float((ref[key].to(dtype).float() - ref[key]).abs().max()) for key in ref}


def check_tolerances(dtype: torch.dtype, floor: dict, peer: dict, f_out: float = 4.0, f_grad: float = 4.0,
                     min_out: float = 1e-4, min_grad: float = 1e-3) -> dict[str, float]:
    """Pass thresholds that follow from the dtype instead of from a guess.
       out  : <= f_out  x the rounding floor of the output (an exact kernel is within 0.5 ulp of it);
       grad : <= f_grad x max(rounding floor, PyTorch's own `sdpa_math` backward error in the same dtype) --
              'no worse than PyTorch's reference implementation in this precision'.
    fp32 keeps tight absolute minima so the rule stays meaningful there."""
    t_out = max(f_out * floor["out"], min_out if dtype == torch.float32 else 0.0)
    t_grad = {}
    for key in ("dq", "dk", "dv"):
        t_grad[key] = max(f_grad * max(floor[key], peer[key]), min_grad if dtype == torch.float32 else 0.0)
    return dict(out=t_out, **t_grad)


def run_check(args) -> int:
    w = ResultWriter(args.out, REPO)
    dtype = DTYPES[args.dtype]
    bad = 0
    for seq in args.check_seqs:
        res = check_correctness(args.systems + (["sdpa_math"] if "sdpa_math" not in args.systems else []),
                                args.device, dtype, seq=seq, causal=args.causal)
        floor = rounding_floor(args.device, dtype, seq, 2, 12, 64, args.causal)
        tol = check_tolerances(dtype, floor, res["sdpa_math"], args.tol_factor, args.tol_factor)
        print(f"C1 seq={seq} dtype={args.dtype}  rounding floor out={floor['out']:.2e}  "
              f"tol out={tol['out']:.2e} dq={tol['dq']:.2e} dk={tol['dk']:.2e} dv={tol['dv']:.2e}")
        for name in args.systems:
            d = res[name]
            ok = d["out"] <= tol["out"] and all(d[k] <= tol[k] for k in ("dq", "dk", "dv"))
            bad += (not ok)
            print(f"C1 {name:26s} out={d['out']:.3e} dq={d['dq']:.3e} dk={d['dk']:.3e} dv={d['dv']:.3e}  "
                  f"{'PASS' if ok else 'FAIL'}")
            for key, val in d.items():
                w.write(module="attention", experiment="C1", system=name, metric=f"max_abs_diff_{key}",
                        value=val, unit="abs", n_reps=1, dtype=args.dtype, causal=args.causal,
                        forward_only=False, seq_len=seq,
                        notes=f"vs fp32 reference; tol={tol[key]:.3e} (floor-based, factor {args.tol_factor}); "
                              f"{'PASS' if ok else 'FAIL'}")
    return 1 if bad else 0


# --------------------------------------------------------------------------------------
# C6: forced FLASH backend
# --------------------------------------------------------------------------------------
def run_flash_check(args) -> int:
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if not torch.cuda.is_available():
        print("C6 needs a CUDA GPU", file=sys.stderr)
        return 2
    w = ResultWriter(args.out, REPO)
    half = DTYPES[args.dtype] if args.dtype != "float32" else torch.bfloat16
    half_name = str(half).split(".")[-1]
    B, H, S, D = 1, 12, 1024, 64
    mk = lambda dt: [torch.randn(B, H, S, D, device="cuda", dtype=dt) for _ in range(3)]
    out_rc = 0

    def attempt(dt, **kw):
        q, k, v = mk(dt)
        try:
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                F.scaled_dot_product_attention(q, k, v, **kw)
            return False, ""
        except Exception as e:  # noqa: BLE001
            return True, f"{type(e).__name__}: {str(e).splitlines()[0][:140]}"

    raised, msg = attempt(torch.float32, is_causal=True)
    print(f"C6.1 fp32 + FLASH only            -> raised={raised}  {msg}")
    w.write(module="attention", experiment="C6", system="sdpa_flash", variant="fp32_input",
            metric="forced_flash_raises", value=int(raised), unit="bool", n_reps=1,
            dtype="float32", forward_only=True, notes=msg)
    if not raised:
        out_rc = 1

    mask = torch.ones(S, S, dtype=torch.bool, device="cuda").tril()
    raised2, msg2 = attempt(half, attn_mask=mask)
    print(f"C6.2 {half_name} + attn_mask + FLASH only -> raised={raised2}  {msg2}")
    w.write(module="attention", experiment="C6", system="sdpa_flash", variant="explicit_mask",
            metric="forced_flash_raises", value=int(raised2), unit="bool", n_reps=1,
            dtype=half_name, forward_only=True, notes=msg2)

    q, k, v = mk(half)
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            F.scaled_dot_product_attention(q, k, v, is_causal=True)
        torch.cuda.synchronize()
    # FunctionEventAvg (key_averages) exposes `.key`, not `.name` (torch 2.6: AttributeError on .name)
    names = sorted({e.key for e in prof.key_averages() if e.device_type.name == "CUDA"})
    is_flash = any("flash" in n.lower() for n in names)
    print(f"C6.3 {half_name} + is_causal + FLASH only -> kernels={names}  flash_kernel_launched={is_flash}")
    w.write(module="attention", experiment="C6", system="sdpa_flash", variant=f"{half_name}_is_causal",
            metric="flash_kernel_launched", value=int(is_flash), unit="bool", n_reps=1,
            dtype=half_name, forward_only=True, notes=";".join(names)[:300])
    return out_rc if is_flash else 1


# --------------------------------------------------------------------------------------
# does torch.compile turn his attention into SDPA?
# --------------------------------------------------------------------------------------
def run_inductor_check(args) -> int:
    from torch._dynamo.utils import counters

    w = ResultWriter(args.out, REPO)
    dev = args.device
    dt = DTYPES[args.dtype]
    S, N, D = 512, 12, 64
    rc = 0

    # Positive control: a textbook 4D softmax(QK^T/sqrt(d))V that Inductor's fuse_attention patterns are known to
    # match. Without it a count of 0 cannot be told apart from "counter not wired up".
    def textbook(q, k, v):
        return torch.matmul(torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / 8.0, dim=-1), v)

    torch._dynamo.reset()
    counters.clear()
    q, k, v = [torch.randn(1, N, S, D, device=dev, dtype=dt) for _ in range(3)]
    torch.compile(textbook)(q, k, v)
    n_ctl = counters["inductor"]["fuse_attention"]
    print(f"inductor fuse_attention rewrites for positive control (4D textbook attention): {n_ctl}")
    w.write(module="attention", experiment="A1", system="textbook_attention_4d_control",
            variant="inductor_sdpa_rewrite", metric="fuse_attention_count", value=n_ctl, unit="count",
            n_reps=1, dtype=args.dtype, forward_only=True,
            notes="positive control; expect >=1, else the counter below proves nothing")
    if n_ctl < 1:
        print("CONTROL FAILED: counter did not move on a textbook pattern; counts below are not evidence")
        rc = 1
    from cs336_basics.transfromer.scaled_dot_prod_attention import ATTENTION_KERNEL_REGISTRY

    fn = ATTENTION_KERNEL_REGISTRY["vectorized_torch_compiled"]
    torch._dynamo.reset()
    counters.clear()
    q, k, v = [torch.randn(N, S, D, device=dev, dtype=dt) for _ in range(3)]
    try:
        fn(q, k, v, True)
        n, note = counters["inductor"]["fuse_attention"], ""
    except Exception as e:  # noqa: BLE001
        n, note, rc = -1, f"{type(e).__name__}: {str(e)[:160]}", 1
    print(f"inductor fuse_attention rewrites for torch_compile_attention: {n} {note}")
    w.write(module="attention", experiment="A1", system="torch_compile_attention", variant="inductor_sdpa_rewrite",
            metric="fuse_attention_count", value=n, unit="count", n_reps=1, dtype=args.dtype,
            forward_only=True, notes=note or "counters['inductor']['fuse_attention']")
    return rc


# --------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE.parent / "results" / "final" / "attention.csv"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16", choices=list(DTYPES))
    ap.add_argument("--seqs", type=int, nargs="+", default=DEFAULT_SEQS)
    ap.add_argument("--systems", nargs="+", default=DEFAULT_SYSTEMS, choices=ALL_SYSTEMS)
    ap.add_argument("--mode", default="both", choices=["fwd", "fwdbwd", "both"])
    ap.add_argument("--tokens", type=int, default=TOKENS, help="B*S is held at this many tokens (B = tokens // S)")
    ap.add_argument("--batch", type=int, default=0, help="override: fixed batch for every seq (disables --tokens)")
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--causal", dest="causal", action="store_true", default=True)
    ap.add_argument("--non-causal", dest="causal", action="store_false")
    ap.add_argument("--reps", type=int, default=5, help="independent timing repetitions per cell")
    ap.add_argument("--warmup-calls", type=int, default=5, help="untimed calls before the timer (compile / autotune)")
    ap.add_argument("--timer", default="do_bench", choices=["do_bench", "cudagraph"])
    ap.add_argument("--check", action="store_true", help="C1 correctness vs fp32 reference")
    ap.add_argument("--check-seqs", type=int, nargs="+", default=[512, 1000],
                    help="1000 is not a multiple of any tile size: real prompts are not either")
    ap.add_argument("--tol-factor", type=float, default=4.0, help="multiple of the bf16 rounding floor / sdpa_math error")
    ap.add_argument("--check-flash-raises", action="store_true", help="C6")
    ap.add_argument("--check-inductor", action="store_true")
    ap.add_argument("--single", action="store_true", help="(internal) one cell in this process")
    ap.add_argument("--system", default=None, choices=ALL_SYSTEMS)
    ap.add_argument("--seq", type=int, default=0)
    args = ap.parse_args(argv)

    if args.single:
        args.mode = "fwd" if args.mode == "both" else args.mode
        return run_single(args)
    stage = ("attention_check" if args.check else "attention_flash_raises" if args.check_flash_raises
             else "attention_inductor" if args.check_inductor else "attention_sweep")
    manifest.record(stage, out_dir=Path(args.out).parent, extra=dict(dtype=args.dtype, seqs=args.seqs, tokens=args.tokens))
    if args.check:
        return run_check(args)
    if args.check_flash_raises:
        return run_flash_check(args)
    if args.check_inductor:
        return run_inductor_check(args)
    return run_sweep(args)


if __name__ == "__main__":
    raise SystemExit(main())
