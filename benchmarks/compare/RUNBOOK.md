# RUNBOOK — 2026-10-16 重做(决定记录:`docs/sprints/week7-discussion.md` D-7)

> 谁来跑:你。**脚本不会开机、续费、新租或销毁实例**(花钱的动作由你决定)。所有结果写进 `results/final/`;`results/archive/` 是旧一轮的证据,不引用。
> 标 ⚠ 的命令没有在 3090 上实跑过,第一次跑要看输出,别直接信。

## 范围(D-7)

| 模块 | 做什么 | 在哪跑 |
|---|---|---|
| Tokenizer | 只测 pretokenize:Python(对齐 `simple_split`)vs C++,LOC、一次性编译时间、预热后运行时间 | **本机**(Apple M4 Pro) |
| C5 | 引擎 vs HF `generate` 的 greedy 一致性(fp32 逐 token 完全一致;bf16 报分叉位置) | 本机 CPU 可跑 fp32;GPU 上再跑一遍 |
| Attention | 5 行 × 8 个长度,固定 B·S=32768、bf16、每格 fresh 进程;profiling 单独脚本(torch.profiler + Nsys) | 单卡 3090 |
| Serving | metrics 对比 + 头条归因(阶段 B,**gate 通过后**才写/跑矩阵脚本) | 单卡 3090 |
| DDP | **不新测**,沿用旧数据(`results/archive/` 里的 v2 之前的结论;措辞见 D-7) | — |

## 阶段 0 — 本机(已完成的部分)

```bash
cd benchmarks/compare
../../.venv/bin/python -m pytest -q                                   # 全部 CPU 测试
../../.venv/bin/python tokenizer/pretok_loc.py                        # pretokenize 相关行数(口径写在脚本头)
../../.venv/bin/python tokenizer/bench_pretokenize.py \
    --corpus /path/to/owt_valid.txt --reps 5 --warmup 1               # 要在 Terminal 面板里跑(后台启动会被降优先级)
../../.venv/bin/python serving/c5_greedy_equal.py \
    --checkpoint ../../training/ckpt_190m_fwe/model_only.pt --workload workloads/workload_wiki128.json \
    --n-prompts 16 --new-tokens 32 --dtype float32 --device cpu       # CPU 也行,fp32 慢一点
```

注意:计时类脚本不要在后台(`&` / 工具的 background)里跑——macOS 会把它们降到 nice 5,P 核/E 核调度会改变数字。

## 阶段 A — 单卡 GPU(开机/销毁由你决定;2026-10-16 实跑过一次,下面是踩过坑之后的顺序)

```bash
# 0) 上传代码(盒子上没有 .git;不要传 results/ 和 ckpt)。必须包含 training/pyproject.toml 和 training/README.md,
#    否则 editable 安装会失败:cs336_systems/__init__.py 读 importlib.metadata.version("llm-training"),没有回退
rsync -a --exclude=__pycache__ --exclude=results benchmarks/compare/ <box>:/workspace/llm-stack/benchmarks/compare/
rsync -a training/cs336-basics training/cs336_systems training/pyproject.toml training/README.md <box>:/workspace/llm-stack/training/
rsync -a --exclude=results serving/ <box>:/workspace/llm-stack/serving/

# 1) 准备环境(~5 分钟,幂等):venv-fa = torch 2.6.0+cu126 + flash-attn 2.8.3 wheel + 两个 editable 包 + nsys(apt)
bash /workspace/llm-stack/benchmarks/compare/remote/setup_gpu_box.sh        # 进度:/workspace/SETUP_STATUS.txt

# 2) 先 env_check(Vast API 里的 extra_env 写 gvisor 的那台,容器里其实是 runc:以实测为准,别看标签)
source /workspace/venv-fa/bin/activate && cd /workspace/llm-stack/benchmarks/compare && python env_check.py --strict

# 3) 正确性(bf16)+ 计时矩阵 + gate;git 状态从本机导出(盒子没有 .git)
BENCH_GIT_SHA=$(git rev-parse --short=10 HEAD) BENCH_GIT_DIRTY=1 bash remote/run_stage_a.sh     # 进度:/workspace/STAGE_A_STATUS.txt
python attention/summarize.py            # 延迟 / 对 sdpa_flash 的比值 / OOM / 显存;'~' = 重复间 std 超过 10%
python attention/gate.py                 # 退出码 0 继续,3 = 触发(停下讨论),4 = 无法评估

# 4) 仅当 gate 没触发或他同意继续:profiling(单独脚本;计时区间不含编译/启动)
python attention/profile_prefill.py --tool both
```

**不要在 sweep 运行期间改盒子上的 `bench_attention.py` / `schema.py` / `manifest.py`**:每个格子都是重新启动的进程,会读到新代码,前后格子就不是同一份代码了。

**Gate(D-7):** 跑完后先看 `results/final/attention.csv`。若**所有 S 上 `sdpa_flash` 对 `flash_attn_pkg` 的中位数延迟比都在 0.9–1.1,停下来和 manager 讨论**,不要继续阶段 B。
每个阶段结束**立刻**把 `results/final/` rsync 回本机;长任务前先 `vastai show user` 看余额(余额耗尽实例会被自动停机,盘上的数据拿不出来)。

## 阶段 B — serving(gate 通过后)

范围和设置见 D-7「Serving」:C = 1/8/32/64,bf16,混合输出长度 32–96,请求数 8×C,预热 2×C,3 次重复;
头条 = ragged 引擎(naive attention)vs SGLang `cb_only` / `default` vs HF `generate`(eager / sdpa / flash_attention_2);
归因 = 替换阶梯 naive→sdpa→fa2 + cProfile(引擎 CPU 侧,单独跑,不进吞吐表)+ 单步 decode torch profiler。
**矩阵脚本和新的汇总脚本在 gate 之后再写**;现有的 `serving/run_sglang.sh`(默认写 `results/final/`)、`serving/closed_loop.py`、`serving/make_mixed_workload.py` 可直接复用。

```bash
python serving/c5_greedy_equal.py --checkpoint training/ckpt_190m_fwe/model_only.pt --workload workloads/workload_wiki128.json \
    --dtype float32 --device cuda            # 同一脚本在 GPU 上再验一遍;--dtype bfloat16 只报分叉位置
```

## 数字去向

| 简历/README 上的数字 | 来自 |
|---|---|
| pretokenize:LOC / 编译 / 运行时间 | `results/final/tokenizer.csv`(experiment `K1-pretok`) |
| attention | `results/final/attention.csv`(A1)+ `attention_profile.csv` + `profile_prefill/SUMMARY.md` |
| 引擎正确性 | `results/final/serving_correctness.csv`(C5) |
| 推理 vs SGLang / HF | `results/final/serving.csv`(阶段 B) |
| DDP | 旧数据,措辞「与 torch DDP 持平(历史数据,n=1,未在同机重测)」 |
