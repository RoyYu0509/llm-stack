# benchmarks/compare — 自研组件 vs 主流实现的对标 harness

被测的是自研的模型 / attention kernel / DDP / 推理引擎 / BPE;对标对象是 PyTorch SDPA / `flash-attn` 包、`torch` DDP、HuggingFace、SGLang、vLLM、HF `tokenizers`。
结果 CSV 里的每个数字都带版本、GPU、dtype、`forward_only`、git sha,
因为面试官会直接打开它。

> **2026-10-16 重做(决定记录 `docs/sprints/week7-discussion.md` D-7)。** 此前的结果(v2 / v3 等)整体移到 `results/archive/`,**不引用**;
> 新一轮的结果只写进 `results/final/`。范围:tokenizer 只测 pretokenize、C5 引擎正确性、Attention(5 行 × 8 个长度,固定 B·S=32768、bf16、fresh 进程)、
> serving(阶段 B,Attention gate 之后)、DDP 不新测。执行步骤见 [RUNBOOK.md](RUNBOOK.md)。
> Attention 的名字:原 `vectorized` → **`attention`**,原 `compiled` → **`torch_compile_attention`**,`naive` einsum 实现不再参与比较。
> workload 输入文件在 `workloads/`,不在 `results/`。
>
> **现状(2026-10-17)**:阶段 A(attention)、B / C / D(serving:阶梯、SGLang、vLLM、开环)和 KV cache 尺寸扫描都已在 RTX 3090 Ti 上跑完,结果在 `results/final/`;
> 阶段 D 的 44 格汇总见 `results/final/serving_stage_d_summary.txt`。`results/archive/`(重做前的旧数据)、`results/final/serving_raw/`(原始 dump)、`*.tgz` 打包、
> Nsight trace 都**不进 git**(见根目录 `.gitignore`)—— 文中提到 `archive/` 的地方在仓库里是找不到的,这是有意的。
> **结果行里的 `git_sha` 只是跑数据时的基线提交(`a3cac72`),`git_dirty` 也不可靠**:代码是 rsync 到租用机器上的未提交工作树。
> 对应的代码现在分布在 #31(CB 主体)、#32(优化路径)、#33(serving 工具)等 PR 里;要精确复现,以这些 PR 合并后的 `main` 为准。
```bash
cd benchmarks/compare
../../.venv/bin/python -m pytest -q        # 193 条(另有 4 条按环境 skip:缺 190M checkpoint / video1 产物 / 录像),CPU 即可(含需要 C++ 编译器的 driver 测试);GPU 项只测 harness 逻辑
```

执行顺序和命令见 [RUNBOOK.md](RUNBOOK.md)。结果落在 `results/final/*.csv`(统一列,见 `schema.py`)。

## 编号 → 脚本 → 现状

| 编号 | 回答什么 | 脚本 | 本机(无 GPU)已验证 | 还需 3090 |
|---|---|---|---|---|
| C3 | 自研 190M ≡ HF `LlamaForCausalLM`? | `convert_hf/convert_190m_to_hf.py` | ✅ 真权重,max\|Δlogit\| 2.2e-5 ~ 3.2e-5(阈值 1e-3),argmax 100% 一致,seq 48 / 256 / 1024 | — |
| C2 | C++ BPE 的 merges 与 Python 一致吗? | `tokenizer/c2_merges_equal.py` | ✅ 见下「C2 的发现」 | 真语料上复核 |
| C1 / C6 | kernel 数值对不对(**bf16**,容差由 bf16 舍入下限推出);强制 FLASH 是否真的报错 | `attention/bench_attention.py --check / --check-flash-raises` | ✅ harness 逻辑(fp32 参考、变异检查、容差能分出对错) | ✅ 需要 |
| A1 | `attention` / `torch_compile_attention` / SDPA flash / `flash_attn_pkg` / 自写 Triton 前向的延迟与显存(固定 B·S) | `attention/bench_attention.py` | ✅ harness 逻辑 | ✅ 需要 |
| A1-profile | 同 5 行在不同 prefill 长度上的 kernel 级 profile(torch.profiler + Nsys),按 kernel 类型分组 | `attention/profile_prefill.py` | ✅ 解析器、分组、busy 比例的算术 | ✅ 需要 |
| C5 | 引擎 vs HF `generate` 的 greedy 输出一致吗(seq 与 cb 两条路径) | `serving/c5_greedy_equal.py` | ✅ 小模型 + 变异检查(忽略 padding mask 的 bug 会被抓出) | fp32 CPU 即可;GPU 上再验 |
| T1 / T2 | 自写 DDP vs `torch` DDP;bucket sweep | `ddp/run_t1.sh` + `ddp/ingest_lm_matrix.py` | ✅ 入库脚本复现 video1 的 77.1% / 82.3%;**与 torch DDP 的对比已有两组旧数据(+0.1% / −1.6%,均 n=1)** | 可选(需 2 张卡;补 n=12 的 torch DDP 一行) |
| T7 | 2×3090 的 allreduce 带宽上限 | `ddp/run_t1.sh nccl`(nccl-tests) | — | 可选(需 2 张卡) |
| K1-pretok | **只测 pretokenize**:Python(对齐 `simple_split`)vs C++;LOC、一次性编译时间、预热后运行时间 | `tokenizer/bench_pretokenize.py`(+ `pretok_loc.py`、`pretok_driver.cpp`) | ✅ 本机实测(OWT valid 290 MB,先过「C++ 与行级 Python 参考逐位一致」的等价性检查) | — |
| K1(旧) | 整条 BPE 训练耗时 | `tokenizer/bench_train.py` | 保留,不再作为头条 | — |
| I1 | 自引擎 / HF generate / SGLang 吞吐与延迟 | `serving/closed_loop.py` + `serving/run_sglang.sh` | ✅ 假服务器校验 TTFT / TPOT / 并发上限;重放录像复现 427.4 vs 28.5 tok/s | ✅ 需要 |

## 已在本机核实的两件事

**C3 — 转换正确。** 键名与形状对应、RoPE 配对(自研 interleaved → HF half-split,按 head 重排 q/k 行)、`tie_word_embeddings=False`、SwiGLU 的 W1→gate / W3→up / W2→down。
`convert_hf/test_convert_hf.py` 里有两条**变异检查**(不重排 q/k;交换 gate/up)必须失败,否则这个测试分不出对错。

**C2 — 「merges 与 Python 一致」不是无条件成立的**(2026-10-15 发现)。同一份 200 KB 中英混合语料、vocab 700(443 个 merge):

| Python 切词规则 | 语料处理 | 结果 |
|---|---|---|
| 与 C++ `simple_split` 对齐 | 换行换成空格 | **443 / 443 全部一致** —— 两边的 merge 算法(最高频、并列取字节序更大的 pair)确实相同 |
| 与 C++ 对齐 | 保留换行 | 第 75 个 merge 起分歧(Python `(b'\n', b'-')`,C++ `(b'\xe6', b'\x97')`) |
| Python 默认 GPT-2 正则 | 保留换行 | 第 36 个 merge 起分歧(Python `(b' ', b' ')`,C++ `(b'\xe5', b'\x86')`) |
| Python 默认 GPT-2 正则 | 换行换成空格 | 第 16 个 merge 起分歧(Python `(b' ', b' ')`,C++ `(b'\xe4\xb8', b'\x80')`) |

原因:C++ 训练端用 `std::getline` 逐行读,**`\n` 在切词前就被丢掉了**;Python 的切词看得到 `\n`,会产生 `"\n-"`、`"\nThe"` 这类 token。另外两边切词规则本身不同
(C++ 按空白切;Python 用 GPT-2 正则)。结论:简历里不要写「merges 完全一致」,可以说「同一切词规则、无换行语料下与 Python 实现逐 merge 一致」。

**DDP 的桶长什么样 —— 190M 在 25 MB 阈值下是 18 个桶**(`ddp/bucket_plan.py`,复用真实的 `Bucket` 类;梯度 761.9 MB):
输出头 154.4 MB 自己一个桶(单个张量就超过阈值);中间 16 个桶约 26–28 MB;**最后一个桶 163.8 MB = layer 0 的张量 + 154 MB 的 embedding**。
反向时梯度从输出侧向输入侧依次就绪,所以最后这个桶在反向的最末尾才 ready —— 它的 all-reduce 没有任何剩余计算可以盖住。

**暴露通信的实测量级**(`ddp/ingest_lm_matrix.py` 对 video1 的 190M / ctx1024 / b8 每卡):1 卡 861.5 ms/步;2 卡 Bucketed 1046.9 ms/步,**每步多 185 ms**;
Naive DDP 1116.8 ms/步,多 255 ms。同一每卡 batch 下计算量相同,多出来的就是没被藏起来的通信 / 同步(重叠省掉约 70 ms)。
这是「为什么是 82% 而不是 95%」的第一手数字;它与 `nccl-tests` 实测带宽对照后才能说瓶颈在带宽还是在重叠。

## 配置对齐表(公平比较的前提;每次出数字都要对着这张表)

**Attention(A1)**

| | 自写 naive | 自写 vectorized | 自写 Triton FA | `F.scaled_dot_product_attention` |
|---|---|---|---|---|
| 实现 | einsum + masked_fill + 手写 softmax | `torch.matmul` 物化 N×N(自写 autograd.Function) | **前向**为 Triton kernel;backward 为 `@torch.compile` 的稠密重算(物化 (B·H,N,N)) | FlashAttention-2 / mem-efficient / math,`sdpa_kernel` 指定 |
| causal | `is_causal` 参数 | 同左 | tile 内 mask,**不跳过上三角 tile** | `is_causal=True`,不传 mask |
| 老数据(02-26)的口径 | 8K OOM | 177.8 ms | 27.2 ms(仅前向、causal=True、fp16) | 无 |

**DDP(T1)** —— 自写 `DDPOverlapBucketed`(`FlashDDP.py`)与 `torch.nn.parallel.DistributedDataParallel`

| | 自写 | torch DDP(benchmark 里的构造) |
|---|---|---|
| 分桶 | 逆序遍历参数;满 `bucket_size_mb`(类默认 5,配置 25)即封桶,桶可超阈值一个张量 | `bucket_cap_mb=<同值>`;文档:未指定则 25 MiB,按参数逆序分桶 |
| **阈值单位** | **十进制 MB**(`_get_tensor_size` 除以 1e6) | **MiB**(本机 torch 2.6.0:`int(bucket_cap_mb * 1024 * 1024)`)→ 同样写 25,字节数差 ≈ 4.9% |
| **首桶** | 与其它桶同阈值 | 固定 1 MiB(`torch.distributed._DEFAULT_FIRST_BUCKET_BYTES = 1048576`),让最先 ready 的梯度更早开始通信 |
| 触发 | `register_post_accumulate_grad_hook`;下一个桶 ready 才发,保证各 rank 顺序一致 | autograd hook;按 bucket **index** 序发 allreduce |
| 通信流 | 独立 CUDA stream + event 同步 | NCCL 内部 |
| 平均 | `div_(world_size)` 后 `SUM` | 默认平均 |
| 梯度累积 / `no_sync` | **没有** | 有 |
| 其它 | 要求 `zero_grad(set_to_none=False)` | `gradient_as_bucket_view=False`、`find_unused_parameters=False`、`broadcast_buffers` 默认 |

**Serving(I1)**

| | 自引擎 | HF `generate()` | SGLang `cb_only` | SGLang `default` |
|---|---|---|---|---|
| 批处理 | `cb`:iteration-level、全部放行、每轮同步 forward;`seq`:max-batch=1 | 逐请求(锁串行) | 调度器本身即 CB,关不掉 | 同左 + 优化 |
| KV cache | 每步 `torch.cat` 的连续张量,无分页 | HF DynamicCache | token 级 KV pool | 同左 + radix 前缀复用 |
| CUDA graph | 无 | 无 | **关** | 开 |
| radix / overlap / chunked prefill | 无 | 无 | **关** | 开 |
| attention | 写死 naive | `--attn eager|sdpa`(**显式指定,别用默认**) | `--attention-backend`(显式指定并记录实际使用的) | 同左 |
| dtype | fp32 + TF32 matmul(`matmul_precision: high`) | 同左 | `--dtype float32`(后端不支持则退 bf16,**并标注 dtype 不对齐**) | bf16 为其自然配置,单列 |

「同功能集」只能写成「关掉能关的」;关不掉的(token 级 KV pool、kernel、进程拆分)是差距的一部分,报告差距时必须列出。

SGLang 还有一个 `cb_off` 变体(`cb_only` + `--max-running-requests 1`),是它自己的串行臂,对应自引擎的 `seq`。
这样「CB 带来的倍数」可以在两边各算一次 —— 比较的是**CB 的收益**(自引擎 15× vs SGLang 的 N×),而不只是绝对速度;
两种比较都要报,且各自写清分母。

## 公平性陷阱(每条都有来历)

1. SDPA 同时给 `is_causal` 和 `attn_mask`:2.14 文档写会抛错,但 torch 2.6.0(CPU)实测**不抛错、静默取交集**(mask AND causal)。行为随版本变 → 只传 `is_causal=True`、不传 mask(物化 mask 还会让 flash 后端不可用),并在 CSV 记 torch 版本。
2. `sdpa_kernel` 的官方页**没写**不满足时是报错还是回退 → `--check-flash-raises` 实测,并用 profiler 确认 launch 的 kernel 名。
3. `torch.compile` 会把 `matmul → div/mul → softmax → matmul` 改写成 SDPA(Inductor `fuse_attention`)→ `--check-inductor` 读计数器,别把 compile 版当成「另一种实现」。
4. `triton.autotune(key=[...])`:key 变了(新的 seq)就重新调优 → 计时前每个 shape 先 warmup;`do_bench` 的 warmup / rep 单位是**毫秒**。
5. dtype:训练是 fp32/TF32,attention sweep 是 fp16 → 分开报。
6. 扩展效率只用弱扩展定义 `tps₂ / (2·tps₁)`,每卡 batch 相同;1→2 卡只有一个点,不写「可线性扩展」。
7. 闭环并发语义:同一时刻 C 个请求在途;`arrival` = 拿到槽位的时刻;流提前结束(EOS 没被忽略)的请求记 failed,不计入吞吐。
8. 吞吐/延迟只来自实测 CSV;harness 输出里的 `p99` 在样本 <100 时会标注不可靠。
9. MFU 用完整公式 `6N + 12·L·H·Q·T`(PaLM 附录);3090 的峰值请自己从数据手册取,不要沿用任何文档里的数。

## 旧数据的口径(别把它们当成这个目录产出的)

| 数字 | 出处 | 口径 |
|---|---|---|
| 6.54× | git `0579d6a:artifacts/attention_sweep_results.csv` | **仅前向**;8192、hd64、12 头、b4;`causal=True` 写死;分母是**未编译**的 vectorized(naive 在 8K OOM);对 compile 版 1.95× |
| 85.9% | git `dc5137f:artifacts/lm_matrix_results.csv` | tiny 配置 12L / vocab 1 万 / ctx 512 / b20 每卡;每格 n=1;**同表 torch DDP 85.8%** |
| 82.3% | `training/artifacts/video1/ddp_flash_ctx1024_b8` | 190M / ctx1024 / b8 每卡 / 12 个计时 epoch;**无 torch DDP 行**(T1 补) |
| 14.99× | `serving/benchmarks/results/analyze_3090_closed_loop.txt` | 分母是自引擎 max-batch=1;naive attention;闭环;64 token;3 次重复 |
| 15.2× | `training/artifacts/serving_benchmark.json` | **另一个数**:KV cache 对全量重算的 p50 解码延迟比(CPU),与上一行不是同一回事 |

## 未对比及原因

| 对象 | 原因 |
|---|---|
| ~~vLLM~~ | **已对比**(2026-10-16 夜):`serving/run_vllm.sh`,同功能集行 = `nograph_noprefix`(保留 `torch.compile`、关 CUDA graph 与前缀缓存),`default` 行只当「工业上限」参照 |
| Megatron-LM(TP/PP) | 依赖高带宽互联,2×3090 上不公平,环境重 |
| Lightning / Accelerate | DDP 的薄封装,没有新信息 |
| nanoGPT / llm.c | 架构不同,不同口径 |
| torchtitan / DeepSpeed | 接入成本高;时间富余再说 |

## 已知局限

- 只有 1→2 卡一个扩展点;2×3090 的互联方式(PCIe / NVLink 桥)仓库里没有记录,`ddp/run_t1.sh topo` 会记下。
- Triton FlashAttention 只有前向是 Triton kernel;backward 是 PyTorch 重算,所以长序列的 fwd+bwd 对比里自写实现会 OOM —— 这是**如实的结果**,CSV 的 `forward_only` 列把两种分开。
- 自引擎基线路径(ragged CB)的 attention 是 naive;优化路径(预分配 KV + FA-2 decode + 合批 prefill + `torch.compile`,默认关闭)在 #32;两者都没有 CUDA graph、没有分页 KV、**没有准入控制**(在飞请求超过 KV 槽位数时主循环会静默死掉,开环压测由 harness 限制在飞数)。
- 单 seed、单语料的 190M 预训练结果,无外部同口径基线。
