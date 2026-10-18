"""benchmarks/serving_metrics.py 的测试:指标定义本身必须先对,benchmark 的数字才有意义。

两类:
  · 手算用例 —— 每条断言的期望值都在注释里手算过,不是把实现的输出抄回来。
  · 外部核对 —— 成本模型(FLOPs / 参数量)拿 torch.utils.flop_counter 和真实的 TransformerLM 对。
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from serving_metrics import (  # noqa: E402
    ForwardRecord, ModelCost, Slo, Trace, e2e, forward_bytes, forward_flops, hw_summary, itls,
    kv_cache_bytes, meets_slo, percentile, summarize, tpot, ttft,
)


# ---------------------------------------------------------------- 分位数


def test_percentile_linear_interpolation():
    xs = [1, 2, 3, 4]
    assert percentile(xs, 50) == 2.5  # 位置 (4-1)*0.5 = 1.5 → 2 与 3 的中点
    assert percentile(xs, 90) == pytest.approx(3.7)  # 位置 2.7 → 3 + 0.7×(4-3)
    assert percentile(xs, 99) == pytest.approx(3.97)
    assert percentile([7.0], 99) == 7.0
    assert percentile([4, 1, 3, 2], 50) == 2.5, "输入没排序也要对"
    assert math.isnan(percentile([], 50))


# ---------------------------------------------------------------- 单条请求的延迟


def _trace(times, arrival=10.0, prompt_len=100, **kw):
    return Trace("r", arrival, prompt_len, times, **kw)


def test_single_request_latencies():
    t = _trace([10.5, 10.6, 10.8, 11.0])
    assert ttft(t) == pytest.approx(0.5)  # 10.5 − 10.0
    assert e2e(t) == pytest.approx(1.0)  # 11.0 − 10.0
    assert tpot(t) == pytest.approx(0.5 / 3)  # (11.0 − 10.5) / (4 − 1),分母是间隔数不是 token 数
    assert itls(t) == pytest.approx([0.1, 0.2, 0.2])


def test_e2e_identity():
    """E2E = TTFT + TPOT × (n_out − 1) —— 指标之间必须互相自洽。"""
    t = _trace([10.31, 10.4, 10.77, 11.2, 11.21, 12.0])
    assert e2e(t) == pytest.approx(ttft(t) + tpot(t) * (len(t.token_times) - 1))


def test_tpot_undefined_for_one_token():
    assert tpot(_trace([10.5])) is None
    assert itls(_trace([10.5])) == []


# ---------------------------------------------------------------- 吞吐与 goodput


def test_throughputs_by_hand():
    a = Trace("a", 0.0, 100, [1.0, 2.0, 3.0, 4.0, 5.0])  # 100 in / 5 out
    b = Trace("b", 0.0, 50, [float(i) for i in range(1, 11)])  # 50 in / 10 out
    s = summarize([a, b], window_s=10.0)
    assert s["request_throughput"] == pytest.approx(0.2)  # 2 / 10
    assert s["output_token_throughput"] == pytest.approx(1.5)  # 15 / 10
    assert s["total_token_throughput"] == pytest.approx(16.5)  # (150 + 15) / 10


def test_failed_and_aborted_are_not_throughput_but_are_misses():
    ok = Trace("ok", 0.0, 10, [0.5, 0.6])
    bad = Trace("bad", 0.0, 10, [], ok=False, status="failed")
    cut = Trace("cut", 0.0, 10, [0.4, 0.5, 0.6], ok=False, status="aborted")  # 被超时中止,有部分 token
    s = summarize([ok, bad, cut], window_s=1.0)
    assert (s["n_total"], s["n_done"], s["n_failed"], s["n_aborted"]) == (3, 1, 1, 1)
    assert s["output_token_throughput"] == 2.0, "被中止请求的部分 token 不算吞吐"
    assert s["goodput"] == 1.0
    assert s["slo_attainment"] == pytest.approx(1 / 3), "分母是全部请求,不是成功的请求"


def test_goodput_needs_both_slos():
    slo = Slo(ttft_s=1.0, tpot_s=0.1)
    good = _trace([10.5, 10.55, 10.6])  # ttft .5, tpot .05
    slow_first = _trace([11.5, 11.55, 11.6])  # ttft 1.5 > 1.0
    slow_decode = _trace([10.5, 10.7, 10.9])  # ttft .5 ok, tpot .2 > .1
    assert meets_slo(good, slo)
    assert not meets_slo(slow_first, slo)
    assert not meets_slo(slow_decode, slo)
    s = summarize([good, slow_first, slow_decode, _trace([], ok=False, status="failed")], window_s=2.0, slo=slo)
    assert s["goodput"] == pytest.approx(0.5)  # 1 条达标 / 2 s
    assert s["slo_attainment"] == pytest.approx(0.25)


def test_slo_boundary_is_inclusive_and_single_token_has_no_tpot_constraint():
    slo = Slo(ttft_s=0.5, tpot_s=0.1)
    assert meets_slo(_trace([10.5]), slo), "n_out=1:TPOT 无定义,只看 TTFT,且恰好等于阈值算达标"
    assert not meets_slo(_trace([10.5000001]), slo)
    assert meets_slo(_trace([10.4, 10.5, 10.6]), slo)  # tpot 恰好 0.1(浮点上略有误差也应该算达标)


def test_p99_reliability_flag():
    ts = [Trace(str(i), 0.0, 1, [1.0, 2.0]) for i in range(99)]
    assert not summarize(ts, 10.0)["p99_reliable"]
    assert summarize(ts + [Trace("x", 0.0, 1, [1.0, 2.0])], 10.0)["p99_reliable"]


def test_tail_is_not_hidden_by_mean():
    """90 条 TTFT=1 s + 10 条 100 s(10% 的请求排了很久的队):p50 看不出问题,均值有一点,p95 / p99 才暴露。"""
    ts = [Trace(str(i), 0.0, 1, [1.0, 1.5]) for i in range(90)]
    ts += [Trace(f"slow{i}", 0.0, 1, [100.0, 100.5]) for i in range(10)]
    s = summarize(ts, 200.0)
    assert s["ttft"]["p50"] == 1.0
    assert s["ttft"]["p90"] == pytest.approx(1.0 + 0.1 * 99)  # 位置 99×0.9 = 89.1 → xs[89]=1, xs[90]=100,插值 0.1
    assert s["ttft"]["p95"] == 100.0 and s["ttft"]["p99"] == 100.0
    assert s["ttft"]["mean"] == pytest.approx(0.9 * 1.0 + 0.1 * 100.0)  # 10.9 s,同时和 p50 差 10 倍


# ---------------------------------------------------------------- MFU / MBU 手算


COST = ModelCost(matmul_params=1000, weight_bytes=4000, num_layers=2, num_heads=2, head_dim=4, d_model=8,
                 elem_bytes=4)


def test_decode_forward_cost_by_hand():
    r = ForwardRecord(seconds=1e-6, rows=3, new_tokens=1, kv_widths=[10, 10])
    # dense = 2 · (3×1 tokens) · 1000 = 6000;attention 每层 4·B·H·Sq·Sk·dh = 4·3·2·1·10·4 = 960,两层 1920;
    # RoPE 8 · 2 层 · 3 · 1 · d_model 8 = 384
    assert forward_flops(r, COST) == 8304
    # 权重 4000 + 每层 K,V 各 B·H·Sk·dh·4 字节 = 3·2·10·4·4 = 960,×2(K 和 V)×2 层 = 3840
    assert forward_bytes(r, COST) == 7840
    assert kv_cache_bytes(r, COST) == 3840  # 2(K,V)·2 层·3·2·10·4·4


def test_prefill_forward_cost_by_hand():
    r = ForwardRecord(seconds=1e-6, rows=1, new_tokens=5, kv_widths=[5, 5])
    # dense = 2·5·1000 = 10000;attention 每层 4·1·2·5·5·4 = 800,两层 1600;RoPE 8·2·1·5·8 = 640
    assert forward_flops(r, COST) == 12240


def test_hw_summary_splits_phases_and_computes_utilisation():
    pre = ForwardRecord(2e-6, 1, 5, [5, 5])
    dec = ForwardRecord(1e-6, 3, 1, [10, 10])
    h = hw_summary([pre, dec], COST, peak_tflops=0.0396, peak_gbps=78.4, window_s=4e-6)
    assert h["decode"]["forwards"] == 1 and h["prefill"]["forwards"] == 1
    assert h["decode"]["tflops"] == pytest.approx(8304 / 1e-6 / 1e12)  # 0.008304
    assert h["decode"]["mfu"] == pytest.approx(0.008304 / 0.0396)
    assert h["decode"]["gbps"] == pytest.approx(7.84)  # 7840 B / 1 µs
    assert h["decode"]["mbu"] == pytest.approx(0.1)
    assert h["all"]["seconds"] == pytest.approx(3e-6)
    assert h["forward_time_fraction"] == pytest.approx(0.75)  # 3 µs 在前向里 / 4 µs 窗口
    assert h["peak_kv_cache_bytes"] == 3840  # decode 那次(3 行 × 宽度 10)比 prefill 那次(1 行 × 5)大


def test_unknown_peak_gives_none_not_a_guess():
    h = hw_summary([ForwardRecord(1e-6, 1, 1, [4, 4])], COST, peak_tflops=None, peak_gbps=None)
    assert h["decode"]["mfu"] is None and h["decode"]["mbu"] is None


def test_max_concurrent_requests_by_hand():
    # 每 token KV = 2·2 层·8·4 B = 128 B;ctx 1000 → 128000 B/请求;
    # 显存 1_000_000 × 0.9 − 权重 4000 = 896000 → 896000 // 128000 = 7
    assert COST.max_concurrent_requests(1_000_000, ctx=1000) == 7
    assert COST.max_concurrent_requests(1000, ctx=1000) == 0


# ---------------------------------------------------------------- 成本模型对真实模型 / 外部计数器


@pytest.fixture(scope="module")
def tiny():
    torch = pytest.importorskip("torch")
    pytest.importorskip("cs336_basics")
    from cs336_basics.lm import TransformerLM

    cfg = dict(vocab_size=64, context_length=64, num_layers=2, d_model=32, heads_num=4, d_ff=48, theta=10000.0)
    torch.manual_seed(0)
    model = TransformerLM(**cfg, device="cpu", dtype=torch.float32).eval()
    return model, cfg


def test_param_counts_match_architecture(tiny):
    model, cfg = tiny
    c = ModelCost.from_model(model, num_heads=cfg["heads_num"])
    D, F, L, V = cfg["d_model"], cfg["d_ff"], cfg["num_layers"], cfg["vocab_size"]
    # 每层:Q K V O 四个 D×D + SwiGLU 的 W1 W3 (F×D) 和 W2 (D×F);再加输出头 V×D。embedding 是查表,不算。
    assert c.matmul_params == L * (4 * D * D + 3 * D * F) + V * D
    # 权重字节 = matmul 参数 + RMSNorm 增益(每层 2 个 + 最后 1 个,各 D 个)
    assert c.weight_bytes == 4 * (c.matmul_params + (2 * L + 1) * D)
    assert c.head_dim == D // cfg["heads_num"]


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_flops_match_torch_flop_counter(tiny, phase):
    """用 torch 自带的 FlopCounterMode 数真实前向的矩阵乘 FLOPs,和公式对。"""
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    model, cfg = tiny
    c = ModelCost.from_model(model, num_heads=cfg["heads_num"])
    B, S = 3, 7
    ids = torch.randint(0, cfg["vocab_size"], (B, S))
    with torch.no_grad():
        _, past = model.forward_with_cache(ids)
        if phase == "prefill":
            counter = FlopCounterMode(display=False)
            with counter:
                model.forward_with_cache(ids)
            rec = ForwardRecord(1.0, B, S, [S] * cfg["num_layers"])
        else:
            counter = FlopCounterMode(display=False)
            with counter:
                model.forward_with_cache(ids[:, :1], past_kv=past)
            rec = ForwardRecord(1.0, B, 1, [S + 1] * cfg["num_layers"])
    measured = counter.get_total_flops()
    assert forward_flops(rec, c) == pytest.approx(measured, rel=0.001), (
        f"{phase}: 公式 {forward_flops(rec, c):.0f} vs torch 计数 {measured:.0f} —— "
        "成本模型和真实前向对不上,MFU / MBU 就不可信"
    )
