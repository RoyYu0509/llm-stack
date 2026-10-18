"""M3(LMS-11 / GitHub #18)—— continuous batching 吞吐 / 延迟 benchmark。

比较两种服务方式,除了「同一时刻引擎里最多几条请求」之外其余全部相同:
  seq —— 顺序处理:引擎一次只处理一条请求,其余在门口排队(= LMS-7 之前的行为,Orca 论文里
         max batch size = 1 的退化情形)。实现方式是 benchmark 自己在引擎前面挂一个「一次只放
         一条进去」的闸门;引擎代码、模型、请求完全相同。
  cb  —— continuous batching:请求一到就交给引擎,由引擎自己按轮次合批。

公平性(面试会被追问的那几条,每条都落在代码里):
  · 同一个模型对象、同一份权重(随机初始化,固定 seed —— 吞吐只取决于形状和算子,与权重数值无关;
    唯一的例外是 EOS,所以 EOS 被关掉了);
  · 同一批请求:prompt 长度由固定 seed 的均匀分布生成,两种方式拿到逐字节相同的 prompt;
  · 同一输出长度:忽略 EOS,每条请求都恰好生成 --new-tokens 个 token(Anyscale 和 Orca 的实验都这么做);
  · 同一客户端:闭环,任何时刻最多 C 条请求在途,一条完成立刻补下一条;每个并发度 C 发 N = max(16, 4C) 条;
  · 预热 + 至少 3 次重复,两种方式在每次重复里交替先后(抵消机器温度 / 频率漂移),报均值 ± 标准差;
  · 线程数固定并写进结果文件。

指标(全部从每个 token 第一次被看到的时刻算出来):
  吞吐    = 生成 token 总数 / (最后一条完成 − 第一条发出)
  TTFT    = 第一个生成 token 被看到的时刻 − 这条请求被客户端发出的时刻(含排队)
  ITL     = 同一条请求相邻两个 token 被看到的时间差
  tok/fwd = 生成 token 总数 / 模型前向次数(model.head 调用次数):seq 恒为 1,cb 接近平均批大小
「被看到」= 引擎每让出一次控制权,benchmark 调一次 m2_adapters.generated_so_far()。

依赖 tests/m2_adapters.py 里的 build_engine 和 generated_so_far —— M2 接好之前这个脚本跑不起来。

用法(在 serving/ 下):
    uv run --no-sync python benchmarks/cb_benchmark.py --preset small     # 冒烟,几十秒
    uv run --no-sync python benchmarks/cb_benchmark.py --preset 190m      # 验收用,M4 Pro CPU 上实测约 22 分钟
    可选:--device mps(未作为验收口径,见 spec)、--threads N、--concurrency 1 2 4 8、--repeats 5、
          --config ../training/configs_sft/lr1e5_replay.json [--checkpoint ckpt.pt]
产出:<out-dir>/cb_benchmark_<preset>.json 和同名 .png;终端打印汇总表和验收判定。
"""

import argparse
import asyncio
import importlib.util
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

SERVING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVING))

from inferenceLM.data.request import RequestData  # noqa: E402
from inferenceLM.data.request_status import RequestStatus  # noqa: E402
from inferenceLM.data.tokenized_data import TokenizedData  # noqa: E402

# ----------------------------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------------------------

PRESETS = {
    # 和他 190M checkpoint 同一个形状(training/configs_sft/lr1e5_replay.json 的 model 段)
    "190m": dict(
        model=dict(vocab_size=50257, context_length=1024, num_layers=16, d_model=768, num_heads=12,
                   d_ff=2048, rope_theta=10000.0),
        prompt_len=(32, 256), new_tokens=64,
    ),
    # 冒烟用:同样的结构,缩小到几十秒能跑完
    "small": dict(
        model=dict(vocab_size=8192, context_length=512, num_layers=4, d_model=256, num_heads=8,
                   d_ff=688, rope_theta=10000.0),
        prompt_len=(16, 128), new_tokens=32,
    ),
}

# 验收门槛 —— 数值与理由见 docs/learning/m2-m3-cb-spec.html#accept。
# 键:(指标, 并发度) -> 门槛。speedup = cb 吞吐均值 / seq 吞吐均值(同一并发度)。
# 来历:manager 标准答案在本机(Apple M4 Pro CPU, 10 线程, torch 2.6)实测 3 次重复的均值,
# 下限类取 75%、上限类取 1/0.75,再取整:
#   speedup @C=8   实测 1.69 ± 0.14 -> 门槛 >= 1.25
#   speedup @C=32  实测 1.89 ± 0.11 -> 门槛 >= 1.40
#   TTFT p50 cb/seq 同负载 @C=8 实测 0.116,@C=32 实测 0.136 -> 门槛 <= 0.20
#   TTFT p50 cb@C=8 / seq@C=1   实测 6.76 -> 门槛 <= 9.0(允许的劣化,原因见 spec)
GATES_190M_CPU = {
    ("speedup_min", 8): 1.25,
    ("speedup_min", 32): 1.40,
    ("ttft_p50_vs_seq_same_load_max", 8): 0.20,
    ("ttft_p50_vs_seq_same_load_max", 32): 0.20,
    ("ttft_p50_vs_seq_c1_max", 8): 9.0,
}

NO_EOS = -1


def load_adapters(path: Path):
    spec = importlib.util.spec_from_file_location("m2_adapters", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_model(model_cfg: Dict, device: str, seed: int, checkpoint: Optional[str]):
    from cs336_basics.lm import TransformerLM

    torch.manual_seed(seed)
    model = TransformerLM(
        vocab_size=model_cfg["vocab_size"], context_length=model_cfg["context_length"],
        num_layers=model_cfg["num_layers"], d_model=model_cfg["d_model"], heads_num=model_cfg["num_heads"],
        d_ff=model_cfg["d_ff"], theta=model_cfg.get("rope_theta", 10000.0), device=device, dtype=torch.float32,
    )
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device)
        model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()})
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def make_requests(n: int, vocab: int, prompt_len: Tuple[int, int], new_tokens: int, seed: int):
    rng = random.Random(seed)
    reqs = []
    for i in range(n):
        L = rng.randint(*prompt_len)
        reqs.append((f"r{i:04d}", [rng.randrange(vocab) for _ in range(L)], new_tokens))
    return reqs


def pct(xs: List[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


# ----------------------------------------------------------------------------------------
# 一次 trial
# ----------------------------------------------------------------------------------------


class ForwardCounter:
    def __init__(self, model) -> None:
        self.n = 0
        self._h = model.head.register_forward_hook(self._hook)

    def _hook(self, *_):
        self.n += 1

    def close(self):
        self._h.remove()


async def run_trial(model, adapters, arm: str, concurrency: int, reqs) -> Dict:
    queue: asyncio.Queue = asyncio.Queue()
    store: Dict[str, RequestData] = {}
    engine = adapters.build_engine(model, queue, store, NO_EOS)
    counter = ForwardCounter(model)
    task = engine.run()

    todo = deque(reqs)
    gate: deque = deque()  # seq 臂:客户端已发出、还没被放进引擎的请求
    in_engine: set = set()
    sent_at: Dict[str, float] = {}
    tok_at: Dict[str, List[float]] = {}
    outputs: Dict[str, List[int]] = {}
    failed: List[str] = []

    def to_engine(rid, prompt, n_new):
        store[rid] = RequestData(request_id=rid, user_id="bench", timestamp=time.time(),
                                 prompt_text="", max_token_length=len(prompt) + n_new)
        queue.put_nowait(TokenizedData(request_id=rid, tokens=list(prompt)))
        in_engine.add(rid)

    def client_send():
        rid, prompt, n_new = todo.popleft()
        sent_at[rid] = time.perf_counter()
        tok_at[rid] = []
        if arm == "cb":
            to_engine(rid, prompt, n_new)
        else:
            gate.append((rid, prompt, n_new))

    try:
        for _ in range(min(concurrency, len(todo))):
            client_send()
        done = 0
        while done < len(reqs):
            if arm == "seq" and not in_engine and gate:
                to_engine(*gate.popleft())
            now = time.perf_counter()
            for rid in list(in_engine):
                seen = adapters.generated_so_far(engine, store, rid)
                while len(tok_at[rid]) < len(seen):
                    tok_at[rid].append(now)
                if store[rid].status in (RequestStatus.DONE, RequestStatus.FAILED):
                    if store[rid].status == RequestStatus.FAILED:
                        failed.append(rid)
                    outputs[rid] = list(store[rid].generated_tokens)
                    in_engine.discard(rid)
                    done += 1
                    if todo:
                        client_send()
            await asyncio.sleep(0)
        t_end = time.perf_counter()
        await asyncio.wait_for(queue.join(), timeout=60)
        await asyncio.wait_for(engine.shutdown(draining=True), timeout=60)
        await asyncio.wait_for(task, timeout=10)
    finally:
        counter.close()

    if failed:
        raise RuntimeError(f"{arm} C={concurrency}: {len(failed)} 条请求 FAILED,例如 {failed[:3]}")
    t_start = min(sent_at.values())
    n_tok = sum(len(v) for v in outputs.values())
    ttft = [tok_at[r][0] - sent_at[r] for r in outputs if tok_at[r]]
    itl = [b - a for r in outputs for a, b in zip(tok_at[r], tok_at[r][1:])]
    e2e = [tok_at[r][-1] - sent_at[r] for r in outputs if tok_at[r]]
    return dict(
        arm=arm, concurrency=concurrency, n_requests=len(reqs), generated_tokens=n_tok,
        makespan_s=t_end - t_start, throughput_tok_s=n_tok / (t_end - t_start),
        forwards=counter.n, tokens_per_forward=n_tok / max(counter.n, 1),
        ttft_p50_s=pct(ttft, 0.5), ttft_p90_s=pct(ttft, 0.9), ttft_mean_s=statistics.fmean(ttft),
        itl_p50_s=pct(itl, 0.5), itl_p90_s=pct(itl, 0.9), e2e_p50_s=pct(e2e, 0.5),
        _outputs=outputs,
    )


# ----------------------------------------------------------------------------------------
# 汇总、画图、验收
# ----------------------------------------------------------------------------------------

METRICS = ["throughput_tok_s", "ttft_p50_s", "ttft_p90_s", "itl_p50_s", "itl_p90_s", "e2e_p50_s", "tokens_per_forward"]


def aggregate(trials: List[Dict]) -> Dict:
    agg: Dict = {}
    for t in trials:
        agg.setdefault(t["arm"], {}).setdefault(t["concurrency"], []).append(t)
    out: Dict = {}
    for arm, by_c in agg.items():
        for c, ts in by_c.items():
            row = {}
            for m in METRICS:
                xs = [t[m] for t in ts]
                row[m] = dict(mean=statistics.fmean(xs), std=statistics.stdev(xs) if len(xs) > 1 else 0.0, n=len(xs))
            out.setdefault(arm, {})[c] = row
    return out


def derived(agg: Dict, concurrencies: List[int]) -> Dict:
    d = {}
    seq1 = agg["seq"][concurrencies[0]]["ttft_p50_s"]["mean"] if 1 in concurrencies else float("nan")
    for c in concurrencies:
        s, b = agg["seq"][c], agg["cb"][c]
        sp = b["throughput_tok_s"]["mean"] / s["throughput_tok_s"]["mean"]
        # 比值的相对误差按一阶误差传播近似
        rel = math.hypot(b["throughput_tok_s"]["std"] / b["throughput_tok_s"]["mean"],
                         s["throughput_tok_s"]["std"] / s["throughput_tok_s"]["mean"])
        d[c] = dict(
            speedup=sp, speedup_std=sp * rel,
            ttft_p50_vs_seq_same_load=b["ttft_p50_s"]["mean"] / s["ttft_p50_s"]["mean"],
            ttft_p50_vs_seq_c1=b["ttft_p50_s"]["mean"] / seq1,
            itl_p50_vs_seq_same_load=b["itl_p50_s"]["mean"] / s["itl_p50_s"]["mean"],
        )
    return d


def check_gates(der: Dict, gates: Dict) -> List[Tuple[str, int, float, float, bool]]:
    res = []
    for (name, c), thr in gates.items():
        if c not in der:
            continue
        if name == "speedup_min":
            v, ok = der[c]["speedup"], der[c]["speedup"] >= thr
        elif name == "ttft_p50_vs_seq_same_load_max":
            v, ok = der[c]["ttft_p50_vs_seq_same_load"], der[c]["ttft_p50_vs_seq_same_load"] <= thr
        elif name == "ttft_p50_vs_seq_c1_max":
            v, ok = der[c]["ttft_p50_vs_seq_c1"], der[c]["ttft_p50_vs_seq_c1"] <= thr
        else:
            continue
        res.append((name, c, v, thr, ok))
    return res


def plot(agg: Dict, der: Dict, concurrencies: List[int], title: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"seq": "#8a877c", "cb": "#2563eb"}
    labels = {"seq": "顺序处理 (seq)", "cb": "continuous batching (cb)"}
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))
    xs = concurrencies

    def series(ax, metric, scale=1.0, arms=("seq", "cb"), ls="-", suffix=""):
        for arm in arms:
            m = [agg[arm][c][metric]["mean"] * scale for c in xs]
            s = [agg[arm][c][metric]["std"] * scale for c in xs]
            ax.errorbar(xs, m, yerr=s, marker="o", capsize=3, color=colors[arm], ls=ls, label=labels[arm] + suffix)

    ax = axes[0]
    series(ax, "throughput_tok_s")
    ax.set_title("吞吐 (tokens/s,均值 ± 标准差)")
    ax.set_xlabel("并发度 C")
    ax = axes[1]
    ax.errorbar(xs, [der[c]["speedup"] for c in xs], yerr=[der[c]["speedup_std"] for c in xs],
                marker="o", capsize=3, color=colors["cb"])
    ax.axhline(1.0, color=colors["seq"], lw=1, ls=":")
    for c in xs:
        ax.annotate(f"{der[c]['speedup']:.1f}×", (c, der[c]["speedup"]), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=9)
    ax.set_title("吞吐倍数 cb / seq(同一并发度)")
    ax.set_xlabel("并发度 C")
    ax = axes[2]
    series(ax, "ttft_p50_s", 1000)
    ax.set_yscale("log")
    ax.set_title("TTFT p50 (ms,对数轴,含排队)")
    ax.set_xlabel("并发度 C")
    ax = axes[3]
    series(ax, "itl_p50_s", 1000, suffix=" p50")
    series(ax, "itl_p90_s", 1000, ls="--", suffix=" p90")
    ax.set_title("ITL (ms)")
    ax.set_xlabel("并发度 C")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(c) for c in xs])
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[3].legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def environment(device: str) -> Dict:
    cpu = platform.processor()
    try:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip() or cpu
    except Exception:
        pass
    return dict(date=datetime.now().isoformat(timespec="seconds"), python=platform.python_version(),
                torch=torch.__version__, device=device, cpu=cpu, threads=torch.get_num_threads(),
                platform=platform.platform())


# ----------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------


async def amain(args) -> None:
    preset = dict(PRESETS[args.preset])
    model_cfg = dict(preset["model"])
    if args.config:
        model_cfg = json.load(open(args.config))["model"]
    prompt_len = tuple(args.prompt_len or preset["prompt_len"])
    new_tokens = args.new_tokens or preset["new_tokens"]
    if args.threads:
        torch.set_num_threads(args.threads)

    adapters = load_adapters(Path(args.adapters))
    t = time.perf_counter()
    model = build_model(model_cfg, args.device, args.seed, args.checkpoint)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[bench] preset={args.preset} params={n_params / 1e6:.1f}M device={args.device} "
          f"threads={torch.get_num_threads()} init={time.perf_counter() - t:.1f}s")
    print(f"[bench] prompt_len~U{prompt_len} new_tokens={new_tokens} concurrency={args.concurrency} repeats={args.repeats}")

    # 预热:两种方式各跑一个小 trial,不计入结果
    warm = make_requests(4, model_cfg["vocab_size"], prompt_len, min(8, new_tokens), args.seed + 999)
    for _ in range(args.warmup):
        for arm in ("seq", "cb"):
            await run_trial(model, adapters, arm, 2, warm)

    trials: List[Dict] = []
    agreement = {}
    for rep in range(args.repeats):
        for c in args.concurrency:
            reqs = make_requests(max(16, 4 * c), model_cfg["vocab_size"], prompt_len, new_tokens, args.seed + c)
            order = ("seq", "cb") if (rep + c) % 2 == 0 else ("cb", "seq")
            outs = {}
            for arm in order:
                r = await run_trial(model, adapters, arm, c, reqs)
                outs[arm] = r.pop("_outputs")
                r["rep"] = rep
                trials.append(r)
                print(f"  rep={rep} C={c:>2} {arm:>3}: {r['throughput_tok_s']:8.1f} tok/s  "
                      f"TTFT p50 {r['ttft_p50_s'] * 1000:8.1f} ms  ITL p50 {r['itl_p50_s'] * 1000:7.1f} ms  "
                      f"tok/fwd {r['tokens_per_forward']:.2f}", flush=True)
            same = sum(outs["seq"][rid] == outs["cb"][rid] for rid in outs["seq"])
            agreement.setdefault(c, []).append(same / len(outs["seq"]))

    agg = aggregate(trials)
    der = derived(agg, args.concurrency)
    gates = check_gates(der, GATES_190M_CPU) if args.preset == "190m" and args.device == "cpu" and not args.config else []

    print("\n  C |  seq tok/s        |  cb tok/s          | 倍数        | TTFT p50 cb/seq(同负载) | cb ITL p50/p90 (ms)")
    for c in args.concurrency:
        s, b, d = agg["seq"][c], agg["cb"][c], der[c]
        print(f"{c:>3} | {s['throughput_tok_s']['mean']:7.1f} ± {s['throughput_tok_s']['std']:5.1f} | "
              f"{b['throughput_tok_s']['mean']:7.1f} ± {b['throughput_tok_s']['std']:6.1f} | "
              f"{d['speedup']:5.2f} ± {d['speedup_std']:4.2f} | {d['ttft_p50_vs_seq_same_load']:8.3f}"
              f"                | {b['itl_p50_s']['mean'] * 1000:6.1f} / {b['itl_p90_s']['mean'] * 1000:6.1f}")
    if gates:
        print("\n验收(门槛见 spec):")
        for name, c, v, thr, ok in gates:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name} @C={c}: 实测 {v:.3f},门槛 {thr}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"cb_benchmark_{args.preset}" + (f"_{args.device}" if args.device != "cpu" else "")
    result = dict(
        environment=environment(args.device), preset=args.preset, model=model_cfg, params=n_params,
        workload=dict(prompt_len_uniform=prompt_len, new_tokens=new_tokens, eos="ignored",
                      requests_per_level="max(16, 4C)", client="closed-loop, at most C outstanding",
                      seed=args.seed),
        concurrency=args.concurrency, repeats=args.repeats, warmup=args.warmup,
        aggregate={arm: {str(c): v for c, v in by_c.items()} for arm, by_c in agg.items()},
        derived={str(c): v for c, v in der.items()},
        output_agreement_seq_vs_cb={str(c): v for c, v in agreement.items()},
        gates=[dict(metric=n, concurrency=c, value=v, threshold=t, passed=ok) for n, c, v, t, ok in gates],
        trials=trials,
    )
    json_path = out_dir / f"{stem}.json"
    json.dump(result, open(json_path, "w"), indent=1, ensure_ascii=False)
    title = (f"{args.preset}: {n_params / 1e6:.0f}M params, {args.device}, {torch.get_num_threads()} threads, "
             f"prompt~U{prompt_len}, {new_tokens} new tokens (EOS ignored), N=max(16,4C) closed-loop, {args.repeats} repeats")
    plot(agg, der, args.concurrency, title, out_dir / f"{stem}.png")
    print(f"\n[bench] wrote {json_path} and {out_dir / f'{stem}.png'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=sorted(PRESETS), default="small")
    p.add_argument("--config", help="读取这个 json 的 model 段(覆盖 preset 的模型形状)")
    p.add_argument("--checkpoint", help="可选:加载真实权重(吞吐与权重无关,默认不加载)")
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--prompt-len", type=int, nargs=2, metavar=("MIN", "MAX"))
    p.add_argument("--new-tokens", type=int)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=0, help="0 = torch 默认")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--adapters", default=str(SERVING / "tests" / "m2_adapters.py"))
    p.add_argument("--out-dir", default=str(SERVING / "benchmarks" / "results"))
    p.add_argument("--rejudge", metavar="JSON", help="不重跑:读一个已有结果文件,按当前门槛重新判定并打印")
    args = p.parse_args()
    if args.rejudge:
        res = json.load(open(args.rejudge))
        der = {int(c): v for c, v in res["derived"].items()}
        for name, c, v, thr, ok in check_gates(der, GATES_190M_CPU):
            print(f"  [{'PASS' if ok else 'FAIL'}] {name} @C={c}: 实测 {v:.3f},门槛 {thr}")
        return
    if args.repeats < 3:
        print("[bench] 注意:repeats < 3,结果不能用于验收(spec 要求至少 3 次重复)")
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
