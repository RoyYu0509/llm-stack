# ROADMAP

> 两部分：**Part I 技术路线**（接下来做什么）和 **Part II 学习路线**（怎么把已经做过的东西真正记住）。
> 最后更新 2026-07-29。

配套讲义：[`docs/handout/assignment-b.tex`](docs/handout/assignment-b.tex) —《Closing the Loop：从语料到一条被服务的请求》。下文的 `Checkpoint (slug)` 都是那份讲义里的自测题块。

---

## Part I — 技术路线

### 1. 瓶颈是数据，不是步数

这是最高优先级，也是唯一一条能明显改善模型质量的。

语料只有约 **174M unique token**，却已经训了 **492M token**（2.8 遍）。继续加步数是在重复同样的数据。

具体动作：把 `cs336_systems/data_harvest/build_mixed_corpus.py` 的字符预算从 780MB 提到约 **3GB**（C4 和 Wikipedia 都是流式的，取之不尽）→ 约 700M token，再训到约 **2B token**（10.5 token/param，接近 Chinchilla 区间）。预计 val loss 从 3.59 降到约 3.2。

成本：建语料 2–3 小时 + 训练约 12 小时。

**动手之前先做第 2 条**，否则新一轮的验证数字仍然不可信。

### 2. 修掉验证集污染（阻塞第 1 条）

`build_instruction()` 会 cycle 小语料池来凑字符预算。这**对 train split 是正确的**（就是 LLaMA 对小域跑多个 epoch 的做法），但同一段逻辑**也被用在了 valid split 上**，把 2600 条 held-out Alpaca 重复了 1.91 遍。

后果不是「稍微乐观一点」：重复过的验证数据是**假的容易**（同一区段 loss 2.3863 vs 不重复的 2.7345，白送 0.35 nats），而且**把微调 delta 的符号弄反了**。

修法已经写在 `build_mixed_corpus.py` 里 `build_instruction()` 的 docstring 中。见 `Checkpoint (valid_contamination)`。

### 3. 把留在远端的东西取回来或明确放弃

vast.ai instance `45963637` 已 `stop`（状态 `exited`），算力不计费但**磁盘费仍在走**。

| 东西 | 本地 | 判断 |
|---|---|---|
| `checkpoints_sft_lr1e5_replay`（Pareto 最优 arm） | ✅ 2.29GB 完整，已 `torch.load` 验证 | 真正不可替代的已经在手 |
| `artifacts/`、`run_logs/`、`configs_sft/`、全部代码 | ✅ | — |
| `checkpoints_pretrain_v1` | ⚠️ 29% | 只在要**继续**训练时才需要；而继续训练本来就要重新租 GPU |
| `checkpoints_sft_lr1e5_plain` | ⚠️ 66% | 只是对照组，结论已量化记录 |
| `data/mixed_corpus` | ⚠️ 0% | 重建要 2–3h，**但第 1 条本来就要重建成 3GB 版本** |

结论：**没有什么是必须抢救的**。如果要取，`vastai start instance 45963637` 之后**必须用 ControlMaster 单连接**一次拉完——上次就是几百条短 SSH 连接把自己敲进了限流。取完确认不需要了就 `vastai destroy`。

### 4. 三个只存在于远端的编排脚本

`continue_pretrain_pipeline.sh`、`post_train_window.sh`、`stage8_leftover_benchmarks.sh` **不在本地**，只在那个实例上。它们是这次冲刺里工程含量最高的一块（吞吐标定 → NaN 门禁 → 反推 `max_iters` → 无人值守窗口）。

如果按第 3 条开机，**把这三个 `.sh` 一起拉回来**，别只拉 checkpoint。否则下次要重新想一遍同样的编排。

### 5. DDP 复测：换实验设计，不是加重复次数

目前唯一站得住的结论是「Bucketed 快于 Naive，5/5 全胜，sign test p=0.062」，并且**报的是中位数 +6.5% 而不是均值 +10.3%**（有一个 +29.9% 的离群值）。

两件事必须一起记住：

- **n=5 时双侧 sign test 的理论最小 p 就是 0.0625**。这个实验从设计上就不可能达到 p<0.05，再多跑几个 rep 也改不了这个上限。
- 去测过「配对到底有没有降低方差」，**答案是没有**。这说明噪声来自**单次运行内部**，所以正确方向是**每轮跑更久**（更多 step、更长 timed window），不是多跑几轮。

见 `Checkpoint (n5_ceiling)` 和 `Checkpoint (pairing_did_not_help)`。

### 6. 补齐一个真正的 SFT

现在的「微调」严格说是 domain-adaptive continued pretraining：`cs336_basics` 的 cross entropy **没有 loss masking**，prompt 部分也在算 loss。

加上 loss masking 之后，`Checkpoint (not_really_sft)` 里那条诚实声明就可以删掉，而 replay 的 Pareto 结论值得在真 SFT 下重测一遍——它有可能变强，也有可能消失。

### 7. 收尾项

- 4 个 `.DS_Store` 和 3 个编译产物（`test_bridge`、`bpe_benchmark`、`bpe_train`）**被 track 了**，每次 `git status` 都是 modified。要 `git rm --cached` + 加 `.gitignore`，是一次不可逆的历史改动，单独做。
- `tests/test_bpe_cpp_vs_python.py` 是红的，**与本次冲刺无关的老坑**（`interview_prep/PROGRESS.md` 里已经这么记过）。根因是 `vocab.txt` / `merges.txt` 两个 fixture 从未被提交、本地也不存在，所以 C++ bridge 拿到的是空数据。`_cpp_train()` 已经实现了，最省事的修法是在 session 级 fixture 里现场训一份。
- `artifacts/` 下 54 个 PNG 仍未提交。纯文本的 `NIGHT_REPORT.md` 和各 `*.csv` 值得进版本控制，PNG 不值得。

---

## Part II — 学习路线

依据是 [`../interview_prep/RECALL_LOG.md`](../interview_prep/RECALL_LOG.md)。**先说清楚覆盖范围**：这份讲义讲的是数据、训练、ablation、KV cache、serving 接缝和测量方法论，**不讲 RoPE / FlashAttention / DDP 的机制本身**——那些的原始教材是两份 CS336 PDF。下面分开列。

### 🔴 Bucketed Overlapped DDP —— 唯一一个完全想不起来的

回忆测试时的原话是「非常模糊」，全部从代码重学。测试轮再问，核心因果对了，但**漏了关键的一环**。

要记住的因果链，缺一不可：

1. backward 从后往前算，所以**按反向参数顺序装桶**，能让最先算完梯度的参数落进最先发出的桶。
2. **调度器强制严格按 bucket 编号顺序发 all-reduce**（防 NCCL 死锁）。← 测试轮漏掉的就是这一环。
3. 因此正向装桶会让 bucket 0（最先注册、最后 ready）卡住后面所有已经 ready 的桶，通信全部堆到 backward 结束，**overlap 完全失效**。

当时把第 3 步解释成「参数互相离得远」，那是不准确的——真正的机制是第 2 步那个顺序约束。

⚠️ **一个容易搞混的地方**：仓库里有两份同类实现——`FlashDDP.py` 里的 `DDPOverlapBucketed` 和 `BucketedOverlapDDP.py` 里的 `BucketedOverlapDDP`。`run_pipeline.py --ddp_wrapper flashddp` 用的是**前者**。

- 机制本身：CS336 Assignment 2 PDF + `cs336_systems/Parallelization/FlashDDP/FlashDDP.py`
- 它的**测量**和为什么第一版结论被撤回：讲义 §5，`Checkpoint (know_your_noise)`、`Checkpoint (which_comparisons_survive)`
- 两个正确性 bug（缺 `dist.broadcast`、`set_to_none=True` 斩断 grad view）：`git log` 里的 `fix(ddp): broadcast initial weights and re-attach grad views`

### 🟡 FlashAttention Triton kernel

两轮都是 🟡，错的点不一样，说明是真的没扎实：

- 第一轮：online softmax 的 correction 递推记不清——`m_i` / `l_i` / `o_i` 三个跑动统计量，新 max 出现时怎么 rescale 旧的累积值。
- 第二轮：把第二个 `program_id` 误认为是 K/V tile index，**实际是 batch index**。K/V tile 是 kernel 内部顺序 for 循环过的，不是并行网格维度——**这一点是结构性的**：online softmax 的递推本身就要求 K/V 必须顺序处理。另外说成「两个 statistics」，实际是三个。

讲义**不覆盖**这一条。去读 CS336 Assignment 2 PDF §1.3 + `cs336_systems/FlashAttention/flash_attention_triton.py:199-246`。

### 🟡 RoPE

旋转机制记对了（2 维一组、角度依 position 和 dim index），**搞错了作用对象**：以为加在 token embedding 上，实际是每层作用在 Q/K 上。

讲义只覆盖一个侧面：`Checkpoint (rope_before_cache)` —— 为什么 RoPE 必须按**绝对位置**旋转之后再进 KV cache（否则缓存里的 key 带着错位置）。机制本身去看 CS336 Assignment 1 PDF + `cs336-basics/cs336_basics/transfromer/positionalNencoding.py`。

### 🟢 已经扎实的，不用回头

BPE 训练与 `build_dataset` 逐行 tokenize、TransformerLM 整体 pipeline、PreNorm 架构选择、RMSNorm（🔴→🟢，关键点是它**去掉**了减均值而不是加了 bias correction）、SwiGLU（🟡→🟢）。

### 从未被问到、但这次冲刺产出的最有价值的东西

这些在 RECALL_LOG 里一条都没有，因为它们是冲刺**之后**才发生的。按「讲不出来最可惜」排序：

1. **FFN 初始化 bug 的诊断过程** — `Checkpoint (lr_hypothesis_refuted)`、`Checkpoint (prenorm_hides_init_bug)`。价值 100% 在诊断，不在修复（修复只是三行传参）。关键是：**哪一个具体事实**否掉了「缺 LR warmup」这个假说。
2. **KV cache 的两个不变量** — `Checkpoint (is_causal_from_shape)`、`Checkpoint (rope_before_cache)`。
3. **KV cache 的加速比不随 prompt 长度增长** — `Checkpoint (flat_speedup)`。以及那个更难看的发现：`Checkpoint (benchmark_wrote_conclusion_first)`。
4. **25% replay 是严格 Pareto 改进** — `Checkpoint (pareto_not_tradeoff)`、`Checkpoint (forgetting_efficiency)`。
5. **配对评测才是 arm 可分辨的原因** — `Checkpoint (paired_eval)`。差距只有 0.05~0.15，靠的不是样本量。
6. **静默失效的三种形态** — `Checkpoint (silent_zero_step_resume)`、`Checkpoint (val_curve_is_not_what_it_says)`、`Checkpoint (valid_contamination)`。这三个加上 DDP 那两个 bug，共同点是**都不报错**。

### 建议的读法

第一遍顺读讲义，Checkpoint 全部跳过。第二遍只做 Checkpoint。想要空白自测版：

```bash
cd docs/handout && latexmk -xelatex -jobname=assignment-b-quiz -pretex='\def\HANDOUTQUIZ{}' -usepretex assignment-b.tex
```
