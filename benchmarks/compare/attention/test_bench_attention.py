"""CPU tests for the A1 / C1 harness (no GPU, no triton).

They test the *harness* -- that the fp32 reference is right, that the bf16 tolerances accept a correct kernel
and reject a wrong one, that the fixed-token rule and the fresh-process cells behave -- so that a green run on
the GPU means something.
"""
import csv
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import bench_attention as ba  # noqa: E402

pytest.importorskip("cs336_basics.transfromer.scaled_dot_prod_attention")


# ------------------------------------------------------------------------------ reference and correctness check
def test_reference_matches_torch_math_sdpa():
    g = torch.Generator().manual_seed(0)
    q, k, v = [torch.randn(6, 33, 16, generator=g) for _ in range(3)]
    for causal in (True, False):
        ref = ba.reference_fp32(q, k, v, causal)
        sdpa = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)
        assert (ref - sdpa).abs().max() < 1e-5, (
            f"fp32 reference disagrees with torch SDPA (causal={causal}); C1 compares against "
            "this reference, so fix it before trusting any C1 result"
        )


def test_his_attention_and_naive_pass_c1_on_cpu_fp32():
    res = ba.check_correctness(["torch_naive", "attention"], "cpu", torch.float32,
                               seq=48, batch=1, heads=2, head_dim=16)
    for name, d in res.items():
        assert d["out"] < 1e-4, f"{name} forward differs from fp32 reference: {d}"
        assert max(d["dq"], d["dk"], d["dv"]) < 1e-3, f"{name} backward differs: {d}"


def test_c1_has_teeth_wrong_kernel_is_flagged(monkeypatch):
    """A kernel that ignores the causal mask must fail the check, otherwise C1 proves nothing."""
    def not_causal(q, k, v, is_causal):
        return ba.reference_fp32(q, k, v, False).to(q.dtype)

    monkeypatch.setattr(ba, "get_system", lambda name, heads=12: not_causal)
    res = ba.check_correctness(["x"], "cpu", torch.float32, seq=32, batch=1, heads=2, head_dim=16)
    assert res["x"]["out"] > 1e-2, "dropping the causal mask should change the output visibly"


# ------------------------------------------------------------------------------ bf16 tolerances
def _bf16_setup(seq=64, batch=1, heads=2, head_dim=16):
    floor = ba.rounding_floor("cpu", torch.bfloat16, seq, batch, heads, head_dim, True)
    peer = ba.check_correctness(["sdpa_math"], "cpu", torch.bfloat16, seq=seq, batch=batch, heads=heads,
                                head_dim=head_dim)["sdpa_math"]
    return floor, peer


def test_bf16_rounding_floor_is_positive_and_fp32_floor_is_zero():
    f16 = ba.rounding_floor("cpu", torch.bfloat16, 64, 1, 2, 16, True)
    f32 = ba.rounding_floor("cpu", torch.float32, 64, 1, 2, 16, True)
    assert f16["out"] > 1e-4, "bf16 has ~3 significant digits: an exact result still loses precision when stored"
    assert f32["out"] == 0.0


def test_bf16_tolerance_accepts_a_correct_kernel_and_rejects_wrong_ones(monkeypatch):
    floor, peer = _bf16_setup()
    tol = ba.check_tolerances(torch.bfloat16, floor, peer, 4.0, 4.0)

    def exact_then_rounded(q, k, v, is_causal):          # a kernel that is right and only pays the bf16 storage cost
        return ba.reference_fp32(q, k, v, is_causal).to(q.dtype)

    def no_causal(q, k, v, is_causal):                   # wrong: ignores the mask
        return ba.reference_fp32(q, k, v, False).to(q.dtype)

    def noisy(q, k, v, is_causal):                       # wrong in a subtler way: 20x the rounding floor of noise
        o = ba.reference_fp32(q, k, v, is_causal)
        return (o + 20 * floor["out"] * torch.sign(torch.randn_like(o))).to(q.dtype)

    results = {}
    for label, fn in (("ok", exact_then_rounded), ("no_causal", no_causal), ("noisy", noisy)):
        monkeypatch.setattr(ba, "get_system", lambda name, heads=12, _f=fn: _f)
        results[label] = ba.check_correctness(["x"], "cpu", torch.bfloat16, seq=64, batch=1, heads=2, head_dim=16)["x"]
    assert results["ok"]["out"] <= tol["out"], f"a correct kernel must pass: {results['ok']} vs {tol}"
    assert results["no_causal"]["out"] > tol["out"], "the tolerance is so loose that a missing causal mask passes"
    assert results["noisy"]["out"] > tol["out"], "the tolerance cannot see 20x-floor noise"


def test_tolerance_is_a_function_of_dtype():
    f_b = {"out": 4e-3, "dq": 4e-3, "dk": 4e-3, "dv": 4e-3}
    f_h = {"out": 5e-4, "dq": 5e-4, "dk": 5e-4, "dv": 5e-4}
    peer = {"out": 0, "dq": 0, "dk": 0, "dv": 0}
    assert ba.check_tolerances(torch.bfloat16, f_b, peer)["out"] > ba.check_tolerances(torch.float16, f_h, peer)["out"]
    assert ba.check_tolerances(torch.float32, {"out": 0, "dq": 0, "dk": 0, "dv": 0}, peer)["out"] >= 1e-4


# ------------------------------------------------------------------------------ systems / layout / timing
def test_bshd_layout_round_trip_and_values():
    spec = ba.SystemSpec("x", fn=None, layout="bshd")
    B, H, S, D = 2, 3, 5, 4
    flat = torch.arange(B * H * S * D, dtype=torch.float32).view(B * H, S, D)
    nat = spec.to_native(flat, B, H)
    assert nat.shape == (B, S, H, D) and nat.is_contiguous()
    assert torch.equal(nat[1, 2, 0], flat[1 * H + 0, 2]), "(batch 1, token 2, head 0) must come from flat row b*H+h"
    assert torch.equal(spec.from_native(nat, B, H), flat)


def test_flash_attn_pkg_is_unavailable_without_the_wheel_and_not_a_crash():
    try:
        import flash_attn  # noqa: F401
        pytest.skip("flash_attn installed here")
    except ImportError:
        pass
    with pytest.raises(ImportError):
        ba.get_spec("flash_attn_pkg")


def test_counted_tflops_convention():
    # causal fwd: 2*B*H*S^2*D flops; 1 ms -> 2*1*1*1024^2*64 / 1e-3 / 1e12 TFLOP/s
    got = ba.counted_tflops(1.0, 1, 1, 1024, 64, causal=True, forward_only=True)
    assert abs(got - (2 * 1024 * 1024 * 64) / 1e-3 / 1e12) < 1e-9
    fb = ba.counted_tflops(1.0, 1, 1, 1024, 64, causal=True, forward_only=False)
    assert abs(fb / got - 3.5) < 1e-9, "fwd+bwd should be counted as 3.5x forward (bwd = 2.5x)"


def test_is_oom_recognises_messages():
    assert ba._is_oom(RuntimeError("CUDA out of memory. Tried to allocate 6.00 GiB"))
    assert not ba._is_oom(RuntimeError("shape mismatch"))


# ------------------------------------------------------------------------------ fresh-process cells and the sweep
def _args(**kw):
    ns = ba.argparse.Namespace(dtype="float32", heads=2, head_dim=16, tokens=256, reps=2, timer="do_bench",
                               warmup_calls=1, device="cpu", causal=True, batch=0)
    ns.__dict__.update(kw)
    return ns


def test_cell_runs_in_its_own_process_and_reports_status():
    cell = ba.run_cell_subprocess(_args(), "sdpa_math", 64, "fwd", timeout_s=120)
    assert cell["status"] == "ok" and cell["ms"] > 0 and cell["batch"] == 4, cell   # 256 tokens // 64
    unavailable = ba.run_cell_subprocess(_args(), "flash_attn_pkg", 64, "fwd", timeout_s=120)
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        assert unavailable["status"] == "unavailable", unavailable


def test_sweep_keeps_the_token_count_fixed_and_labels_every_row(tmp_path):
    out = tmp_path / "a.csv"
    args = _args(out=str(out), mode="fwd", seqs=[32, 64], systems=["sdpa_math", "attention"], dtype="float32")
    assert ba.run_sweep(args) == 0
    rows = list(csv.DictReader(out.open()))
    lat = [r for r in rows if r["metric"] == "latency_ms"]
    assert {(r["system"], int(r["seq_len"]), int(r["batch"])) for r in lat} == {
        ("sdpa_math", 32, 8), ("attention", 32, 8), ("sdpa_math", 64, 4), ("attention", 64, 4)}, \
        "B must be tokens // S so that B*S stays fixed"
    assert all(r["forward_only"] == "True" for r in rows)
    assert all("B*S=256" in r["variant"] for r in lat)


def test_flat1h_layout_is_the_same_memory_as_flat_and_round_trips():
    spec = ba.SystemSpec("x", fn=None, layout="flat1h")
    flat = torch.arange(6 * 5 * 4, dtype=torch.float32).view(6, 5, 4)
    nat = spec.to_native(flat, 2, 3)
    assert nat.shape == (6, 5, 1, 4) and nat.data_ptr() == flat.data_ptr(), "must be a view: no copy, same layout as SDPA's"
    assert torch.equal(spec.from_native(nat, 2, 3), flat)
