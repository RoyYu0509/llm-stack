"""Serving 指标的定义 —— 一处定义,benchmark 和测试共用。纯 Python,不依赖 torch。

四类指标(定义对照 vLLM / DistServe 的惯例):

  1. 延迟(每条请求一个值,看分布不看均值):
       TTFT = 第一个 token 可见的时刻 − 请求到达时刻          (含排队)
       TPOT = (最后一个 token 时刻 − 第一个 token 时刻) / (n_out − 1)   每条请求一个均值,n_out=1 时无定义
       ITL  = 同一条请求相邻两个 token 的时间差                  (所有请求的所有间隔混在一起成一个分布)
       E2E  = 最后一个 token 可见的时刻 − 请求到达时刻
     恒等式:E2E = TTFT + TPOT × (n_out − 1)(由定义直接得出,测试里钉死)。
  2. 吞吐(窗口 = 第一条到达 → 最后一条完成,或超时中止的时刻):
       输出 token 吞吐 / 总 token 吞吐(prompt + output)/ 请求吞吐,都只算成功完成的请求。
       Goodput = 窗口内「TTFT ≤ SLO 且 TPOT ≤ SLO」的请求数 / 窗口。失败、被中止的请求算未达标。
  3. 硬件效率(从每次前向实际看到的形状反推,不是 nvidia-smi):
       MFU = 实际 FLOPs / 峰值 FLOPs        MBU = 实际必需字节数 / 峰值带宽
       「必需字节」= 每次前向读一遍全部矩阵权重 + 读一遍 KV(含 padding)。引擎每步 torch.cat 整份 KV 的
       额外拷贝没有算进去,所以 MBU 是真实访存量的下界。
  4. 显存:权重 + KV cache 占用 / 显存总量,以及按显存估算的最大并发数。

时间的「origin」不重要,只用差值。所有时间单位:秒。
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

PERCENTILES = (50, 90, 95, 99)
# p99 靠 ≥100 个样本才有意义:n < 100 时 p99 就是在最大值附近插值
MIN_SAMPLES_FOR_P99 = 100


# ----------------------------------------------------------------------------------------
# 延迟与吞吐
# ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Trace:
    """一条请求的完整时间线。token_times[i] = 第 i+1 个输出 token 对客户端可见的时刻。"""

    request_id: str
    arrival: float  # 客户端「想要被服务」的时刻 —— 开环压测里是调度的到达时刻,不是引擎实际收到的时刻
    prompt_len: int
    token_times: Sequence[float] = field(default_factory=list)
    ok: bool = True  # 成功完成且没有被中止
    status: str = "done"  # done / failed / aborted


@dataclass(frozen=True)
class Slo:
    ttft_s: float = 2.0
    tpot_s: float = 0.1  # 10 tok/s/请求 = serving/docs/designs/DESIGN.md 里写的目标
    e2e_s: Optional[float] = None


def percentile(xs: Iterable[float], q: float) -> float:
    """线性插值分位数(与 numpy 默认一致)。q ∈ [0, 100]。空输入返回 nan。"""
    ys = sorted(xs)
    if not ys:
        return float("nan")
    k = (len(ys) - 1) * q / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return ys[lo] + (ys[hi] - ys[lo]) * (k - lo)


def ttft(t: Trace) -> float:
    return t.token_times[0] - t.arrival


def e2e(t: Trace) -> float:
    return t.token_times[-1] - t.arrival


def tpot(t: Trace) -> Optional[float]:
    n = len(t.token_times)
    return None if n < 2 else (t.token_times[-1] - t.token_times[0]) / (n - 1)


def itls(t: Trace) -> List[float]:
    return [b - a for a, b in zip(t.token_times, t.token_times[1:])]


def meets_slo(t: Trace, slo: Slo) -> bool:
    if not t.ok or not t.token_times:
        return False
    p = tpot(t)
    return (
        ttft(t) <= slo.ttft_s
        and (p is None or p <= slo.tpot_s)
        and (slo.e2e_s is None or e2e(t) <= slo.e2e_s)
    )


def distribution(xs: Sequence[float]) -> Dict[str, float]:
    d: Dict[str, float] = {"n": len(xs)}
    if not xs:
        return d
    d["mean"] = statistics.fmean(xs)
    d["max"] = max(xs)
    for q in PERCENTILES:
        d[f"p{q}"] = percentile(xs, q)
    return d


def summarize(traces: Sequence[Trace], window_s: float, slo: Slo = Slo()) -> Dict:
    """一组请求在一个观测窗口里的全部延迟 / 吞吐 / goodput 指标。"""
    done = [t for t in traces if t.ok and t.token_times]
    n_out = sum(len(t.token_times) for t in done)
    n_in = sum(t.prompt_len for t in done)
    good = sum(meets_slo(t, slo) for t in traces)
    tpots = [p for p in (tpot(t) for t in done) if p is not None]
    return dict(
        n_total=len(traces),
        n_done=len(done),
        n_failed=sum(t.status == "failed" for t in traces),
        n_aborted=sum(t.status == "aborted" for t in traces),
        window_s=window_s,
        request_throughput=len(done) / window_s,
        output_token_throughput=n_out / window_s,
        total_token_throughput=(n_in + n_out) / window_s,
        goodput=good / window_s,
        slo_attainment=good / len(traces) if traces else float("nan"),
        slo=dict(ttft_s=slo.ttft_s, tpot_s=slo.tpot_s, e2e_s=slo.e2e_s),
        ttft=distribution([ttft(t) for t in done]),
        tpot=distribution(tpots),
        itl=distribution([x for t in done for x in itls(t)]),
        e2e=distribution([e2e(t) for t in done]),
        p99_reliable=len(done) >= MIN_SAMPLES_FOR_P99,
    )


# ----------------------------------------------------------------------------------------
# 硬件效率:MFU / MBU / 显存
# ----------------------------------------------------------------------------------------

# 厂商规格(dense,无稀疏)。未列出的设备必须从命令行给峰值,否则 MFU / MBU 输出 n/a,不猜。
# 3090:FP32 与 TF32 tensor 的 dense 峰值都是 35.6 TFLOPS,显存带宽 936 GB/s(NVIDIA GA102 白皮书)。
PEAKS = {
    "RTX 3090": dict(tflops=35.6, gbps=936.0),
    "M4 Pro": dict(tflops=None, gbps=273.0),  # Apple 公布的统一内存带宽;CPU / GPU 的 FP32 峰值没有可靠的公开数字
}


def lookup_peak(device_name: str) -> Dict[str, Optional[float]]:
    for key, v in PEAKS.items():
        if key.lower() in device_name.lower():
            return dict(v)
    return dict(tflops=None, gbps=None)


@dataclass(frozen=True)
class ModelCost:
    """一个模型的静态开销。matmul_params = 参与矩阵乘的参数量(ndim==2 且不是 embedding 查表)。"""

    matmul_params: int
    weight_bytes: int  # 每次前向要读一遍的权重字节数(不含 embedding 查表)
    num_layers: int
    num_heads: int
    head_dim: int
    d_model: int
    elem_bytes: int  # KV cache 的元素字节数

    @classmethod
    def from_model(cls, model, num_heads: int, elem_bytes: int = 4) -> "ModelCost":
        """model 需要有 named_parameters()、in_embedding、num_layers(cs336_basics.lm.TransformerLM)。"""
        skip = {id(p) for p in model.in_embedding.parameters()}
        params = [p for p in model.parameters() if id(p) not in skip]
        d_model = next(iter(model.in_embedding.parameters())).shape[-1]
        return cls(
            matmul_params=sum(p.numel() for p in params if p.ndim == 2),
            weight_bytes=sum(p.numel() * p.element_size() for p in params),
            num_layers=model.num_layers, num_heads=num_heads, head_dim=d_model // num_heads,
            d_model=d_model, elem_bytes=elem_bytes,
        )

    def kv_bytes_per_token(self) -> int:
        return 2 * self.num_layers * self.d_model * self.elem_bytes  # K 和 V,所有层

    def max_concurrent_requests(self, vram_bytes: int, ctx: int, util: float = 0.9) -> int:
        """按显存估算:每条请求独占 ctx 个 token 的 KV,不分页、不共享、不算激活和碎片 —— 上界。"""
        free = vram_bytes * util - self.weight_bytes
        return max(0, int(free // (self.kv_bytes_per_token() * ctx)))


@dataclass
class ForwardRecord:
    """一次模型前向。kv_widths 每个元素 = 一层注意力看到的 K 宽度(含 padding),每层一个。"""

    seconds: float
    rows: int  # B
    new_tokens: int  # 每行本次喂进去的 token 数(prefill = prompt 长度,decode = 1)
    kv_widths: List[int]


def forward_flops(r: ForwardRecord, c: ModelCost) -> float:
    """2·T·P_matmul(所有线性层)+ Σ 层 4·B·H·Sq·Sk·dh(QK^T 与 AV)+ RoPE。

    朴素注意力不跳过因果上三角,所以不减半。这份实现的 RoPE 是 2×2 旋转的 einsum,每层对 Q、K 各做一次,
    会被计成矩阵乘:8·B·Sq·D 每层。190M 上只占 0.03%,留着是为了和 torch 的 FLOP 计数器逐项对得上。
    """
    dense = 2.0 * r.rows * r.new_tokens * c.matmul_params
    attn = sum(4.0 * r.rows * c.num_heads * r.new_tokens * sk * c.head_dim for sk in r.kv_widths)
    rope = 8.0 * len(r.kv_widths) * r.rows * r.new_tokens * c.d_model
    return dense + attn + rope


def forward_bytes(r: ForwardRecord, c: ModelCost) -> float:
    """必需字节:权重读一遍 + 每层读 K 和 V 各一遍。"""
    kv = sum(2.0 * r.rows * c.num_heads * sk * c.head_dim * c.elem_bytes for sk in r.kv_widths)
    return c.weight_bytes + kv


def kv_cache_bytes(r: ForwardRecord, c: ModelCost) -> float:
    """这次前向时 KV cache 占的字节(含 padding):2·L·B·H·Sk·dh·elem,Sk 取最宽的一层。"""
    return 2.0 * len(r.kv_widths) * r.rows * c.num_heads * max(r.kv_widths, default=0) * c.head_dim * c.elem_bytes


def hw_summary(records: Sequence[ForwardRecord], c: ModelCost, peak_tflops: Optional[float],
               peak_gbps: Optional[float], window_s: Optional[float] = None) -> Dict:
    """按阶段(prefill = 每行 >1 个新 token;decode = 1 个)汇总 MFU / MBU,再给一个总的。"""
    out: Dict = {}
    for name, sel in (("prefill", lambda r: r.new_tokens > 1), ("decode", lambda r: r.new_tokens == 1),
                      ("all", lambda r: True)):
        rs = [r for r in records if sel(r)]
        secs = sum(r.seconds for r in rs)
        fl = sum(forward_flops(r, c) for r in rs)
        by = sum(forward_bytes(r, c) for r in rs)
        row = dict(forwards=len(rs), seconds=secs, rows_mean=statistics.fmean([r.rows for r in rs]) if rs else 0.0,
                   tflops=fl / secs / 1e12 if secs else float("nan"),
                   gbps=by / secs / 1e9 if secs else float("nan"))
        row["mfu"] = row["tflops"] / peak_tflops if peak_tflops and secs else None
        row["mbu"] = row["gbps"] / peak_gbps if peak_gbps and secs else None
        out[name] = row
    if window_s:  # 窗口里有多少时间真的在前向里,剩下的是调度 / Python / 空等
        out["forward_time_fraction"] = out["all"]["seconds"] / window_s
    out["peak_kv_cache_bytes"] = max((kv_cache_bytes(r, c) for r in records), default=0.0)
    out["weight_bytes"] = c.weight_bytes
    return out
