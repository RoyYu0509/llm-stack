# serving/benchmarks

量 serving 引擎的脚本。**指标的定义只有一处:[`serving_metrics.py`](serving_metrics.py)**(有手算测试和外部计数器核对,见 `tests/serving_metrics_test.py`)。

## 指标覆盖(2026-10-08 核对)

| 指标 | 之前 | 现在 | 在哪里 |
|---|---|---|---|
| TTFT | p50 / p90 / mean(`cb_benchmark`),p50 / mean(`record_demo`) | p50 / p90 / p95 / p99 | `serving_bench.py` |
| TPOT | **没有**(只有 ITL) | 每条请求一个值,取分布 | 同上 |
| ITL | p50 / p90(只有 `cb_benchmark`) | p50 / p90 / p95 / p99 | 同上 |
| E2E | p50(只有 `cb_benchmark`) | p50 / p90 / p95 / p99 | 同上 |
| 输出 token 吞吐 | ✅ | ✅ | 三个脚本都有 |
| 总 token 吞吐(in+out) | **没有** | ✅ | `serving_bench.py` |
| 请求吞吐 | **没有** | ✅ | 同上 |
| Goodput(SLO 约束下的 req/s) | **没有** | ✅ `--slo-ttft` `--slo-tpot` | 同上 |
| MFU / MBU | **没有** | ✅ 按 prefill / decode 分开(`--profile`) | 同上 |
| GPU 利用率 | **没有** | ✅ 只作对照,和 MFU 摆在一起看(`--profile`,需要 pynvml) | 同上 |
| 显存 | **没有** | ✅ 权重、KV 每 token 字节、峰值占用、按显存估算的并发上限 | 同上 |
| 开环(泊松到达)负载 | **没有**(三个脚本全是闭环) | ✅ `run` | 同上 |
| 变长 prompt / 输出、EOS 之外的长尾 | **没有**(固定 64 token,均匀 prompt) | ✅ `--workload chat/rag/fixed` | 同上 |

## 用法

```bash
# 在 serving/ 下。已有的 3090 录像按完整指标重算(不占 GPU,几秒):
uv run --no-sync python benchmarks/serving_bench.py analyze ../../../docs/videos/systems/data/sweep

# 开环压测冒烟(CPU,小模型,约 1 分钟):
uv run --no-sync python benchmarks/serving_bench.py run --preset small --workload chat --duration 4 --profile

# GPU 上的真实压测(vast 3090,脚本里有守卫和路径):
bash serving/benchmarks/run_cb_sweep.sh load   # 延迟 / goodput vs 到达率
bash serving/benchmarks/run_cb_sweep.sh hw     # MFU / MBU / 显存
```

## 口径与局限(读数字之前先读这里)

- **Token 的「可见时刻」= 引擎每让出一次控制权,客户端轮询一次**。引擎一轮 = 所有新请求各 prefill 一次 + 所有在飞请求 decode 一次,
  所以同一轮里产生的 token 时间戳相同,TTFT 里包含了「同一轮里排在前面的别人的 prefill」—— 这是 prefill-first 调度的真实代价,不是误差。
- **闭环录像(`cb_benchmark` / `record_demo`)的 TTFT 不是真实延迟**:C 条请求同时发出、同时完成,下一批同时进来,
  所有请求排成同步的波(C=8 的 cb 里 p50 = p99 就是这个原因)。排队延迟被客户端自己的节奏限制住了。要看排队,用开环的 `run`。
- **开环的 arrival = 调度到达时刻**,不是引擎循环腾出手收到的时刻(见 `serving_bench.py` 模块文档:coordinated omission)。
- **过载时 seq 的队列无限长**:到达结束后只等 `--drain` 秒,没完成的记 aborted,算 SLO 未达标、不算吞吐。
- **MFU 的 FLOPs 按实际执行计**:朴素注意力不跳过因果上三角,所以 prefill 的注意力 FLOPs 没有减半;含 RoPE 的 einsum(0.03%)。
  公式对 `torch.utils.flop_counter` 逐项核对过(prefill / decode 各一条,误差 < 0.1%)。
- **MBU 是下界**:只算「必需字节」= 每次前向读一遍矩阵权重 + 读一遍 KV(含 padding)。引擎每步 `torch.cat` 整份 KV 的额外读写没算。
- **峰值规格**:3090 = 35.6 TFLOPS(FP32 与 TF32 dense 相同)/ 936 GB/s,来自 NVIDIA 资料,**用之前对一下数据手册**;
  M4 Pro 只有带宽(273 GB/s)。未知设备 MFU / MBU 输出 n/a,可用 `--peak-tflops` / `--peak-gbps` 自己给。
- **p99 要 ≥100 个请求才有意义**;样本不够的行会标 `*`。
- **工作负载是「形状参考」**:`chat` 是对数正态的长尾长度,不是 ShareGPT 数据集本身(没有下载任何数据)。分布改 `WORKLOADS`。
- **本目录的脚本不和别的引擎对比**(vLLM / SGLang 的同口径对比在 `benchmarks/compare/`)。seq 是自己引擎的 max-batch=1 退化,衡量的是 continuous batching 本身的收益。
- **闭环录像没记录 git_sha**(录制时代码未提交),它们对应的代码版本无法精确还原。

## 测试

```bash
uv run --no-sync python -m pytest tests/serving_metrics_test.py tests/serving_bench_test.py -q
```

指标库 18 条(手算 + 对 torch FLOP 计数器),测量 harness 18 条(用耗时已知的假引擎校验 TTFT / coordinated omission / seq 闸门 / abort / failed,
再用真引擎 + 真模型校验前向计时器)。两组都做过变异检查(指标库 13 类、harness 13 类,全部被抓)。
