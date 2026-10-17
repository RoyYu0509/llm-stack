"""
M2(LMS-10 / GitHub #17)—— 变长批量 decode 的验收门禁。

这份文件验两件事,缺一不可:
  ① 算得对:变长的几行合成一次前向之后,每一行的结果和「这一行单独、老老实实算」一致;
  ② 真的批了:所有 DECODING 请求的下一步确实是**一次**前向,不是换个地方继续逐条调模型。

怎么跑(LMS-7 合进 main、开始 M2 之后):
    cd serving
    LMS_M2=1 uv run --no-sync pytest tests/batched_decode_test.py -v
不设 LMS_M2=1 时整份文件 skip —— 你在写 LMS-7 时跑全量,不会看到这里的红。

测试只通过 tests/m2_adapters.py(CS336 作业同款 adapter)和引擎公开接口调用你的代码。
第一次跑,每条都会报 NotImplementedError —— 先把 m2_adapters.py 的六个函数接到你的实现上。

两层,建议按这个顺序让它们变绿:
  test_m*  模型层:不经过引擎和 asyncio,直接验「一批变长的行,一次前向」。
  test_e*  引擎层黑盒:请求塞进队列,只看最终输出和前向次数。
  模型层全绿、引擎层还红 → 问题在引擎接线(哪个 token 喂给了哪一行、新请求什么时候进批、
  结束的行有没有移出),不在 attention。

参照答案从哪来:一律是 `model(整条序列)` —— 训练用的那条 forward,不带缓存、不批,是这份
仓库里唯一被训练过程反复验证过的路径。批量 decode 的每一步都要和它比。

容差:logits 比较用 atol=rtol=1e-4。float32 下「一次算 5 行」和「一次算 1 行」的矩阵乘法
累加顺序不同,manager 的参照实现实测最大偏差约 1e-6;1e-4 留了约 100 倍余量。而位置、mask、
KV 对齐这几类真错误,在 manager 的变异测试里造成的最小偏差是 0.059 —— 两者不会混淆。
token 比较是逐 token 精确相等;为了不被「两个候选 logit 几乎打平」这种数值巧合误伤,每条引擎层
测试先检查参照答案每一步的 top-1 与 top-2 差距 >= 1e-4(测试数据本身的前提,见 _greedy_reference)。

「一次前向」的计数方法:在 `model.head`(输出投影)上挂 forward hook。任何前向,不管批不批、
padding 还是拉平,最后都要经过一次 head —— 所以 head 被调用的次数 = 前向次数,
每次 head 输入的「行数」(除最后一维以外各维的乘积)= 这次前向处理了多少个 token 位置。

每条断言的报错信息都写了:①现象;②最可能错的部件;③该回看哪份教材的哪一节
(教材在工作区 llm/docs/learning/ 下)。
"""

import asyncio
import math
import os
import random
from typing import Callable, Dict, Hashable, List, Optional, Tuple

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("LMS_M2") != "1",
    reason=(
        "M2(LMS-10)变长批量 decode 的验收测试,默认关闭。LMS-7 合进 main、开始 M2 时设 LMS_M2=1 打开:"
        "cd serving && LMS_M2=1 uv run --no-sync pytest tests/batched_decode_test.py -v"
    ),
)

TransformerLM = pytest.importorskip(
    "cs336_basics.lm", reason="需要训练包 cs336_basics(uv workspace 里的 training/)"
).TransformerLM

import m2_adapters as m2  # noqa: E402  —— tests/ 下的同名文件,你来接线

from inferenceLM.data.request import RequestData  # noqa: E402
from inferenceLM.data.request_status import RequestStatus  # noqa: E402
from inferenceLM.data.tokenized_data import TokenizedData  # noqa: E402

# ---------------------------------------------------------------------------------------
# 常量:测试模型配置、容差、教材索引
# ---------------------------------------------------------------------------------------

MODEL_CFG = dict(
    vocab_size=512, context_length=128, num_layers=2, d_model=64, heads_num=4, d_ff=176, theta=10000.0
)
ATOL = 1e-4
RTOL = 1e-4
MARGIN_MIN = 1e-4  # 参照答案 top-1 / top-2 至少要差这么多,测试数据才算「没有平局」
NO_EOS = -1

_D = "docs/learning/"
DOC = {
    "cb1": _D + "continuous-batching-mechanics.html#s1「batching 划算这件事,浪费还能藏在哪」",
    "cb3": _D + "continuous-batching-mechanics.html#s3「攒够一批再发车,浪费换了个位置,没有消失」",
    "cb4": _D + "continuous-batching-mechanics.html#s4「把请求的生命周期和引擎的一次迭代拆开」",
    "cb5": _D + "continuous-batching-mechanics.html#s5「拆开之后,一条请求的家当放哪」",
    "cb6": _D + "continuous-batching-mechanics.html#s6「一次迭代里形状不齐,怎么对齐」",
    "p1": _D + "phase1-kv-cache-batching.html#s1「为什么要把请求拼在一起跑」",
    "p2": _D + "phase1-kv-cache-batching.html#s2「拼在一起会不会串味」",
    "p3": _D + "phase1-kv-cache-batching.html#s3「长度不一样怎么办」(padding 和 mask 必须一起上;补齐后 RoPE 位置会错位)",
    "r0": _D + "ragged-attention-walkthrough.html#s0「缓存里到底装的是什么张量」(RoPE 先按绝对位置转、再拼进缓存)",
    "r1": _D + "ragged-attention-walkthrough.html#s1「decode 那一步,is_causal 该填 True 还是 False」",
    "r3": _D + "ragged-attention-walkthrough.html#s3「两条不等长请求,为什么连 torch.cat 都过不去」",
    "r4": _D + "ragged-attention-walkthrough.html#s4「拉平成一条,靠什么账本记住谁是谁」",
    "r5": _D + "ragged-attention-walkthrough.html#s5「prefill 和 decode 混在一起,mask 长什么样」末尾陷阱框:位置 / mask / KV 长度三处必须是同一个数",
    "r7": _D + "ragged-attention-walkthrough.html#s7「门禁:合并版必须和逐条版逐 token 一致」",
    "spec": _D + "m2-m3-cb-spec.html#adapters(adapter 契约)",
}

PART = {
    "causal_or_concat": (
        "decode 这一步的基本路径 —— 新 token 仍按 is_causal=True 画了三角 mask(1×T 的 tril 只放行 key 0)、"
        "或者新 token 的 K/V 没拼到这一行自己的缓存末尾、或者位置差了一",
        ["r1", "r0", "r5"],
    ),
    "all_rows": (
        "多行同时全错。先看 test_m1(单行)是红是绿:\n"
        "      · test_m1 也红 → decode 基本路径:新 token 仍按 is_causal=True 画三角 mask / 新 K、V 没拼到这一行缓存末尾 / 位置差一;\n"
        "      · test_m1 是绿 → 行和 token 的对应关系:喂进去的 token 是按 dict 的迭代顺序排成张量的,而不是按 key 对到它自己那一行"
        "(本测试每一步都故意打乱 tokens 的顺序)",
        ["r1", "r0", "r5", "cb5"],
    ),
    "position_or_pad": (
        "只有比最长行短的那些行错 —— 最可能:①位置用了全批共用的宽度(所有行一个 past_len / 缓存宽度)"
        "而不是这一行自己的真实长度;②padding 出来的 key 位置没有被 mask 掉,被当成真 token attend 到了;"
        "③补齐的方向和 mask 的约定不一致(KV 补在一侧、mask 却按另一侧来标 —— 新行加入、或已有行被加宽时最容易出)",
        ["r5", "p3", "r3"],
    ),
    "pad_leak": (
        "padding 泄漏 —— 同一行只因为「同批的邻居变了」结果就变了:补出来的位置被 attention 看到了,"
        "或者这一行的位置编号跟着批宽度走了(补得越多,位置偏得越多)",
        ["p3", "r3", "r5", "p2"],
    ),
    "join": (
        "新请求加入时的对齐 —— 新行的 KV 补到当前批宽度的方式、已有行被加宽时补在哪一侧、"
        "新行下一步的位置是不是它自己的 prompt 长度,这三件事要和 mask 对得上",
        ["r5", "r3", "cb5"],
    ),
    "leave": (
        "结束移除 —— 一行移出时,它的 token、KV 的那一行、它的长度/位置记录必须一起删,且删的是同一行;"
        "删完以后如果批宽度收缩,各行真实内容和 mask 仍要对齐",
        ["cb4", "cb5", "r5"],
    ),
    "one_forward": (
        "批量前向 —— decode 这一步仍然在逐行调模型(每行一次 forward),或者把一步拆成了多次前向",
        ["cb1", "p1", "cb6"],
    ),
    "engine_wiring": (
        "引擎接线(模型层 test_m* 若已全绿,问题就在这里)—— 喂给某一行的 token 其实是别的请求的上一个 token"
        "(按顺序对应而不是按请求对应)/ 新请求进批前后多走或少走了一步 / 行结束后没移出或移错了行",
        ["cb5", "cb4", "r7"],
    ),
    "admission": (
        "调度准入 —— 新请求没有和已经在 decode 的请求合进同一次前向:要么还是一条跑到底,"
        "要么每条请求各自一次 forward(LMS-7 的交替推进,但没有合批)",
        ["cb4", "cb6", "cb1"],
    ),
    "shutdown": (
        "批量路径上的 non-draining shutdown —— 在飞集合里的每一条(不管它此刻在不在批里)都要标 FAILED 并各自 task_done 一次;"
        "LMS-7 的 T7 在逐条路径上验过同一件事,批量路径是另一段代码,要单独保证",
        ["cb5", "cb4"],
    ),
    "zombie": (
        "结束移除 —— 已经 DONE 的请求还留在批里陪跑:它的行每一步还在被算,批不收缩",
        ["cb3", "cb4", "cb5"],
    ),
}


def explain(part: str) -> str:
    what, refs = PART[part]
    lines = [f"  最可能错的部件:{what}", "  回看:"]
    lines += [f"    - {DOC[r]}" for r in refs]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------
# 夹具与工具
# ---------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(20260929)
    m = TransformerLM(**MODEL_CFG, device="cpu", dtype=torch.float32).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _prompt(seed: int, length: int) -> List[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, MODEL_CFG["vocab_size"], (length,), generator=g).tolist()


class HeadCounter:
    """在 model.head 上挂 forward hook:head 每被调用一次 = 一次前向。

    shapes[i] 是第 i 次前向时 head 的输入形状;rows(i) = 除最后一维外各维的乘积,
    也就是这次前向处理了几个 token 位置(padding 算在内)。on_call 让测试能在
    「第 n 次前向刚发生」的瞬间做事(比如往队列里塞一条新请求)。
    """

    def __init__(self, model) -> None:
        self.shapes: List[Tuple[int, ...]] = []
        self.on_call: Optional[Callable[[int], None]] = None
        self._handle = model.head.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output) -> None:
        self.shapes.append(tuple(inputs[0].shape))
        if self.on_call is not None:
            self.on_call(len(self.shapes))

    @property
    def n(self) -> int:
        return len(self.shapes)

    def rows(self, i: int) -> int:
        return math.prod(self.shapes[i][:-1])

    def close(self) -> None:
        self._handle.remove()


@torch.no_grad()
def _ref_logits(model, seq: List[int]) -> torch.Tensor:
    return model(torch.tensor([seq], dtype=torch.long))[0, -1, :]


@torch.no_grad()
def _greedy_reference(model, prompt: List[int], n_new: int) -> List[int]:
    """逐条、不带缓存、不批的贪心解码 —— 引擎层的参照答案。顺带检查测试数据没有「平局」。"""
    seq, out = list(prompt), []
    for step in range(n_new):
        logits = _ref_logits(model, seq)
        top2 = torch.topk(logits, 2).values
        margin = float(top2[0] - top2[1])
        if margin < MARGIN_MIN:
            pytest.fail(
                f"测试数据本身的前提不成立(不是你的实现的问题):参照答案在第 {step} 步 top-1/top-2 只差 "
                f"{margin:.2e} < {MARGIN_MIN},数值巧合就可能翻转 argmax。把这条报给 manager 换一组 prompt。"
            )
        t = int(torch.argmax(logits))
        out.append(t)
        seq.append(t)
    return out


def _close(a: torch.Tensor, b: torch.Tensor) -> bool:
    return bool(torch.allclose(a.float(), b.float(), atol=ATOL, rtol=RTOL))


class RowBook:
    """测试这一侧的账本:记下每一行到目前为止的完整序列,用来算逐行参照答案。

    每一步喂给某一行的 token 一律取「参照答案在上一位置的 argmax」(teacher forcing),
    所以即使你的某一行算错了,后面几步喂进去的序列也不会跟着漂 —— 每一步的偏差都能单独归因。
    """

    def __init__(self, model, test_id: str, rng_seed: int = 0) -> None:
        self.model = model
        self.test_id = test_id
        self.batch = m2.new_decode_batch(model)
        self.seqs: Dict[Hashable, List[int]] = {}
        self.next_tok: Dict[Hashable, int] = {}
        self.rng = random.Random(rng_seed)
        self.step_no = 0
        self.history: Dict[Hashable, List[torch.Tensor]] = {}

    def lengths(self) -> Dict[Hashable, int]:
        return {k: len(s) for k, s in self.seqs.items()}

    def add(self, key: Hashable, prompt: List[int]) -> None:
        got = m2.add_to_batch(self.model, self.batch, key, list(prompt))
        ref = _ref_logits(self.model, prompt)
        assert tuple(got.shape) == tuple(ref.shape), (
            f"[{self.test_id}] add_to_batch('{key}') 返回的形状是 {tuple(got.shape)},契约要求 ({ref.shape[0]},) "
            f"—— prompt 最后一个位置的 logits。\n  回看:{DOC['spec']}"
        )
        assert _close(got, ref), (
            f"[{self.test_id}] add_to_batch('{key}')(prompt 长度 {len(prompt)})返回的 logits 与 "
            f"model(prompt)[0,-1] 不一致,最大偏差 {float((got - ref).abs().max()):.3g}。这还只是 prefill,"
            f"没进批量 decode —— 先查 prefill 路径本身(ragged 改动有没有碰坏单条 prefill 的 causal mask / 位置)。\n"
            + explain("causal_or_concat")
        )
        self.seqs[key] = list(prompt)
        self.next_tok[key] = int(torch.argmax(ref))
        self.history[key] = []

    def remove(self, key: Hashable) -> None:
        m2.remove_from_batch(self.model, self.batch, key)
        del self.seqs[key], self.next_tok[key]

    def step(self, counter: HeadCounter, fail_part: Optional[str] = None) -> None:
        """批量推进一步,并逐行和参照答案比。"""
        keys = list(self.seqs)
        self.rng.shuffle(keys)  # 契约:dict 顺序不保证等于加行顺序
        tokens = {k: self.next_tok[k] for k in keys}
        before = counter.n
        out = m2.decode_batch_step(self.model, self.batch, tokens)
        n_fwd = counter.n - before
        self.step_no += 1
        assert set(out) == set(tokens), (
            f"[{self.test_id}] 第 {self.step_no} 步:decode_batch_step 返回的 key 是 {sorted(map(str, out))},"
            f"喂进去的是 {sorted(map(str, tokens))}。\n  回看:{DOC['spec']}"
        )
        for k in tokens:
            self.seqs[k].append(tokens[k])
        lens = self.lengths()
        bad = {}
        for k in keys:
            ref = _ref_logits(self.model, self.seqs[k])
            self.history[k].append(out[k].detach().clone())
            if not _close(out[k], ref):
                bad[k] = float((out[k].float() - ref.float()).abs().max())
            self.next_tok[k] = int(torch.argmax(ref))
        if bad:
            longest = max(lens.values())
            only_short = all(lens[k] < longest for k in bad) and any(lens[k] == longest for k in lens)
            if fail_part:
                part = fail_part
            elif len(lens) == 1:
                part = "causal_or_concat"
            elif only_short:
                part = "position_or_pad"
            elif set(bad) == set(lens):
                part = "all_rows"
            else:
                part = "position_or_pad"
            raise AssertionError(
                f"[{self.test_id}] 第 {self.step_no} 步批量 decode 之后,行 {sorted(map(str, bad))} 的 logits 与逐行参照答案"
                f"(model(这一行完整序列)[0,-1])不一致。各行偏差:{ {str(k): f'{v:.3g}' for k, v in bad.items()} }"
                f"(容差 atol=rtol={ATOL})。此刻各行真实长度:{ {str(k): v for k, v in lens.items()} }。\n"
                + explain(part)
            )
        assert n_fwd == 1, (
            f"[{self.test_id}] 第 {self.step_no} 步:批里有 {len(tokens)} 行,decode_batch_step 却触发了 {n_fwd} 次前向"
            f"(model.head 被调用 {n_fwd} 次,每次处理的位置数 {[counter.rows(i) for i in range(before, counter.n)]})。"
            f"结果是对的,但没有批 —— 契约要求一步只做一次前向。\n" + explain("one_forward")
        )


# ---------------------------------------------------------------------------------------
# 模型层(不经过引擎)
# ---------------------------------------------------------------------------------------


def test_m1_single_row_degenerates_to_plain_cached_decode(model):
    """batch=1 退化:只有一行时,批量路径必须和不带缓存的 forward 逐步一致,且每步一次前向。

    ▸ 挂了的含义:连「没有邻居、没有 padding」的最简单情形都不对 → 问题在 decode 的基本路径
      (is_causal / KV 拼接 / 位置差一),和变长无关。先修这条,其余模型层测试才有意义。
    """
    counter = HeadCounter(model)
    try:
        book = RowBook(model, "M1")
        book.add("solo", _prompt(1, 7))
        for _ in range(10):
            book.step(counter, fail_part="causal_or_concat")
    finally:
        counter.close()


def test_m2_ragged_rows_match_per_row_forward(model):
    """核心门禁:长度 1 / 6 / 23 / 57 的四行进同一个批,联合 decode 8 步,每一步每一行都和它单独算一致。

    ▸ 为什么这四个长度:1 是「prompt 只有一个 token」的边界,57 远长于其余三行,制造大量 padding。
    ▸ 挂了的含义看报错里「哪几行错」:只有短行错 → 位置/padding mask;连最长行也错 → decode 基本路径。
    """
    counter = HeadCounter(model)
    try:
        book = RowBook(model, "M2", rng_seed=2)
        for key, (seed, L) in {"L1": (11, 1), "L6": (12, 6), "L23": (13, 23), "L57": (14, 57)}.items():
            book.add(key, _prompt(seed, L))
        for _ in range(8):
            book.step(counter)
    finally:
        counter.close()


def test_m3_row_result_does_not_depend_on_its_batch_neighbours(model):
    """padding 的多少和位置不能影响结果:同一行 A(长度 9),分别①单独、②和一条更短的行(长度 2)、
    ③和一条长得多的行(长度 70,先进批,A 是后加入的短行)各跑 5 步 —— A 每一步的 logits 三种情况必须相同。

    ▸ 三种情况里 A 被补的 padding 从 0 到 61 个,位置也可能落在批里不同的列上。结果只要跟着变,
      就说明 padding 被看见了、或者位置跟着批宽度走了。
    """
    prompt_a = _prompt(21, 9)
    runs = {}
    counter = HeadCounter(model)
    try:
        for name, neighbours in {
            "单独": [],
            "邻居更短(长度 2)": [("S", _prompt(22, 2))],
            "邻居更长(长度 70,先进批)": [("L", _prompt(23, 70))],
        }.items():
            book = RowBook(model, f"M3/{name}", rng_seed=3)
            for key, p in neighbours:
                book.add(key, p)
            book.add("A", prompt_a)
            for _ in range(5):
                book.step(counter, fail_part="pad_leak" if neighbours else "causal_or_concat")
            runs[name] = book.history["A"]
    finally:
        counter.close()
    base = runs["单独"]
    for name, hist in runs.items():
        for i, (x, y) in enumerate(zip(base, hist)):
            assert _close(x, y), (
                f"[M3] 行 A 在「{name}」时第 {i + 1} 步的 logits 和它「单独」时不一样(最大偏差 "
                f"{float((x - y).abs().max()):.3g})。A 的内容一个 token 都没变,变的只有同批的邻居。\n"
                + explain("pad_leak")
            )


def test_m4_leaving_rows_do_not_disturb_the_rest(model):
    """某行先结束:4 行联合 decode;第 2 步后移走**第一个加入**的行,第 4 步后移走**最长**的行
    (它一走,批的最大长度就缩了),其余两行继续跑到第 8 步,每一步都要和逐行参照一致。

    ▸ 为什么专挑这两行:移走第一行最容易暴露「按下标删、删完下标整体错一位」;移走最长行最容易暴露
      「删完以后批宽度收缩/裁剪 padding 时裁多了或裁错了一侧」。
    """
    counter = HeadCounter(model)
    try:
        book = RowBook(model, "M4", rng_seed=4)
        for key, (seed, L) in {"first": (31, 12), "mid": (32, 5), "long": (33, 44), "tail": (34, 19)}.items():
            book.add(key, _prompt(seed, L))
        for step in range(1, 9):
            book.step(counter, fail_part="leave" if step > 2 else None)
            if step == 2:
                book.remove("first")
            if step == 4:
                book.remove("long")
    finally:
        counter.close()


def test_m5_rows_joining_mid_flight_are_aligned(model):
    """新请求中途加入:两行(长度 4、10)先 decode 3 步;然后同时加入一条**比现有各行都长**的(长度 30,
    已有行要被加宽)和一条**很短**的(长度 2,新行要被补齐),再联合 decode 6 步。

    ▸ 加入之后如果老行错了 → 老行被加宽时补的位置和 mask 没对上;新行错了 → 新行的 KV 补齐方式
      或它下一步的位置(应当是它自己的 prompt 长度)不对。
    """
    counter = HeadCounter(model)
    try:
        book = RowBook(model, "M5", rng_seed=5)
        book.add("old4", _prompt(41, 4))
        book.add("old10", _prompt(42, 10))
        for _ in range(3):
            book.step(counter)
        book.add("new30", _prompt(43, 30))
        book.add("new2", _prompt(44, 2))
        for _ in range(6):
            book.step(counter, fail_part="join")
    finally:
        counter.close()


def test_m6_one_decode_step_is_one_forward_over_all_rows(model):
    """真的批了(模型层):5 行的批,连续 4 步,每一步 model.head 恰好被调用 1 次,
    而且这一次调用覆盖了全部 5 行(处理的位置数 >= 5)。

    ▸ 其余模型层测试也都在查「一步一次前向」;这条单独存在,是为了在结果全对的情况下,
      给「没批」一个不和正确性问题混在一起的、专门的红。
    """
    counter = HeadCounter(model)
    try:
        book = RowBook(model, "M6", rng_seed=6)
        for i, L in enumerate([3, 8, 1, 15, 6]):
            book.add(f"r{i}", _prompt(60 + i, L))
        for _ in range(4):
            before = counter.n
            book.step(counter)
            assert counter.rows(before) >= 5, (
                f"[M6] 这一步的唯一一次前向只处理了 {counter.rows(before)} 个位置(head 输入形状 "
                f"{counter.shapes[before]}),批里有 5 行 —— 有的行这一步没被算进去。\n" + explain("one_forward")
            )
    finally:
        counter.close()


def test_m7_extreme_length_gap_up_to_the_context_end(model):
    """极端长度差:一行 prompt 只有 1 个 token,另一行 118 个(上下文 128),联合 decode 10 步,
    长行最后一步喂进去的 token 落在位置 127 —— 上下文的最后一格。

    ▸ 同时压两件事:padding 多到 117 个时 mask 还对不对;长行的位置一路走到 RoPE 表最后一行时,
      用的是不是这一行自己的位置(不是批宽度,也不会越界或回绕)。
    """
    counter = HeadCounter(model)
    try:
        book = RowBook(model, "M7", rng_seed=7)
        book.add("tiny", _prompt(71, 1))
        book.add("huge", _prompt(72, MODEL_CFG["context_length"] - 10))
        for _ in range(10):
            book.step(counter)
    finally:
        counter.close()


def test_m8_batch_drains_to_empty_and_refills(model):
    """批被移空再重新装:两行跑 3 步后全部移走,再加入两条长度完全不同的新行跑 4 步。

    ▸ 挂了的含义:批空了以后还残留着旧的宽度 / 长度 / 位置记录,新行被接在了旧状态后面。
    """
    counter = HeadCounter(model)
    try:
        book = RowBook(model, "M8", rng_seed=8)
        book.add("a", _prompt(81, 20))
        book.add("b", _prompt(82, 3))
        for _ in range(3):
            book.step(counter)
        book.remove("a")
        book.remove("b")
        book.add("c", _prompt(83, 6))
        book.add("d", _prompt(84, 33))
        for _ in range(4):
            book.step(counter, fail_part="leave")
    finally:
        counter.close()


def test_m9_identical_prompts_are_independent_rows(model):
    """两行 prompt 一模一样(key 不同)+ 一行别的:都要各自正确;移走其中一条之后,另一条照常。

    ▸ 挂了的含义:行的身份被 prompt 内容而不是 key 决定了(比如拿 prompt 当 dict 的键、或按内容去重),
      两条相同的请求被合成了一行,移走一条等于两条都没了。
    """
    counter = HeadCounter(model)
    try:
        book = RowBook(model, "M9", rng_seed=9)
        same = _prompt(91, 8)
        book.add("dup-a", same)
        book.add("dup-b", list(same))
        book.add("other", _prompt(92, 13))
        for _ in range(3):
            book.step(counter)
        book.remove("dup-a")
        for _ in range(3):
            book.step(counter, fail_part="leave")
    finally:
        counter.close()


# ---------------------------------------------------------------------------------------
# 引擎层(黑盒)
# ---------------------------------------------------------------------------------------

Req = Tuple[str, List[int], int]  # (request_id, prompt, 要生成几个 token)


def _push(queue: asyncio.Queue, store: Dict[str, RequestData], rid: str, prompt: List[int], n_new: int) -> None:
    store[rid] = RequestData(
        request_id=rid,
        user_id=f"user_{rid}",
        timestamp=0.0,
        prompt_text=f"m2 prompt {rid}",
        status=RequestStatus.PENDING,
        max_token_length=len(prompt) + n_new,
    )
    queue.put_nowait(TokenizedData(request_id=rid, tokens=list(prompt)))


class EngineRun:
    """跑一次引擎:start 里的请求在 run() 之前入队;late 里的 (k, 请求) 在第 k 次前向刚发生之后入队。"""

    def __init__(self, model, test_id: str) -> None:
        self.model = model
        self.test_id = test_id
        self.queue: asyncio.Queue = asyncio.Queue()
        self.store: Dict[str, RequestData] = {}
        self.engine = m2.build_engine(model, self.queue, self.store, NO_EOS)
        self.counter: Optional[HeadCounter] = None
        self.pushed_at: Dict[str, Tuple[int, Dict[str, RequestStatus]]] = {}

    async def go(self, start: List[Req], late: List[Tuple[int, Req]] = (), timeout: float = 60.0,
                 monitor: Optional[Callable[[], None]] = None) -> None:
        for rid, prompt, n in start:
            _push(self.queue, self.store, rid, prompt, n)
        self.counter = HeadCounter(self.model)
        events = [(k, asyncio.Event()) for k, _ in late]

        def on_call(n: int) -> None:
            for k, ev in events:
                if n >= k:
                    ev.set()

        self.counter.on_call = on_call

        async def producer() -> None:
            for (k, (rid, prompt, n)), (_, ev) in zip(late, events):
                await ev.wait()
                started = {r: d.status for r, d in self.store.items()}
                _push(self.queue, self.store, rid, prompt, n)
                self.pushed_at[rid] = (self.counter.n, started)

        stop = asyncio.Event()

        async def watch() -> None:
            while not stop.is_set():
                monitor()
                await asyncio.sleep(0)

        try:
            task = self.engine.run()
            prod = asyncio.create_task(producer())
            mon = asyncio.create_task(watch()) if monitor else None
            try:
                await asyncio.wait_for(prod, timeout=timeout)
            except asyncio.TimeoutError:
                raise AssertionError(
                    f"[{self.test_id}] 等了 {timeout}s,前向次数只到 {self.counter.n},没能触发中途加入的请求 —— "
                    f"要么引擎卡住了,要么在飞请求跑完之前根本没有前向发生。各请求状态:"
                    f"{ {r: d.status.name for r, d in self.store.items()} }\n" + explain("admission")
                )
            try:
                await asyncio.wait_for(self.queue.join(), timeout=timeout)
            except asyncio.TimeoutError:
                raise AssertionError(
                    f"[{self.test_id}] {timeout}s 内 pending_queue.join() 没返回:有请求一直没进终态、或者 task_done 漏了。"
                    f"各请求状态:{ {r: d.status.name for r, d in self.store.items()} }\n" + explain("leave")
                )
            stop.set()
            if mon:
                await mon
            await asyncio.wait_for(self.engine.shutdown(draining=True), timeout=10)
            assert task.done(), f"[{self.test_id}] shutdown(draining=True) 返回了,引擎的 task 还没结束。"
        finally:
            stop.set()
            self.counter.close()

    def assert_mid_flight(self, rid: str) -> None:
        """前提检查:中途加入的请求入队那一刻,确实有别的请求正在飞(否则这条没测到「中途加入」)。"""
        n, statuses = self.pushed_at[rid]
        flying = [r for r, s in statuses.items() if s not in (RequestStatus.PENDING, RequestStatus.DONE, RequestStatus.FAILED)]
        assert flying, (
            f"[{self.test_id}] 前提没满足:{rid} 在第 {n} 次前向后入队时,没有任何请求在飞(状态:"
            f"{ {r: s.name for r, s in statuses.items()} })—— 这条测试没测到它声称要测的「中途加入」。"
            f"如果状态一直停在 PENDING,说明准入/状态写回有问题。\n" + explain("admission")
        )

    def assert_outputs(self, expected: Dict[str, List[int]]) -> None:
        for rid, exp in expected.items():
            d = self.store[rid]
            assert d.status == RequestStatus.DONE, (
                f"[{self.test_id}] 请求 {rid} 最终状态是 {d.status.name},应为 DONE。它生成了 {len(d.generated_tokens)} 个 token。"
                f"各请求状态:{ {r: x.status.name for r, x in self.store.items()} }\n" + explain("engine_wiring")
            )
            got = list(d.generated_tokens)
            if got != exp:
                i = next((j for j, (a, b) in enumerate(zip(got, exp)) if a != b), min(len(got), len(exp)))
                raise AssertionError(
                    f"[{self.test_id}] 请求 {rid} 的输出和逐条贪心参照不一致:第一处分歧在第 {i} 个生成 token"
                    f"(期望 {exp[i] if i < len(exp) else '<结束>'},实际 {got[i] if i < len(got) else '<结束>'});"
                    f"期望长度 {len(exp)},实际 {len(got)}。\n"
                    "  如果 test_m* 全绿:问题在引擎接线,不在 attention。\n" + explain("engine_wiring")
                )

    def assert_actually_batched(self, reqs: List[Req], what: str) -> None:
        """前提检查:这次运行里真的发生过批量 decode(总前向次数 < 生成 token 总数)。

        没有这句,一个根本不批的引擎(一条跑到底,或 LMS-7 那样逐条交替)会让这条测试空转变绿。
        """
        n_seq = sum(n for _, _, n in reqs)
        assert self.counter.n < n_seq, (
            f"[{self.test_id}] 前提没满足:总前向 {self.counter.n} 次,不少于逐条之和 {n_seq} 次 —— 这次运行根本没有"
            f"发生批量 decode,所以这条要测的「{what}」没有被测到。{self.call_summary()}\n" + explain("admission")
        )

    def call_summary(self) -> str:
        return f"每次前向处理的位置数:{[self.counter.rows(i) for i in range(self.counter.n)]}"


def _expected(model, reqs: List[Req]) -> Dict[str, List[int]]:
    return {rid: _greedy_reference(model, prompt, n) for rid, prompt, n in reqs}


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_e1_concurrent_requests_match_sequential_greedy_and_are_batched(model):
    """引擎层核心门禁:4 条不同 prompt 长度(3/11/1/25)、不同输出长度(12/6/16/9)的请求在 run() 之前
    全部入队。①每条输出和逐条贪心参照逐 token 一致;②总前向次数 <= 22,而逐条跑要 43 次。

    ▸ 22 这个上限从哪来(和你的准入策略无关):4 次 prefill + 最多 3 次「准入落后」的余量 + 最长那条
      还需要的 15 次 decode。「一轮收一条」和「一轮收全部」两种策略都在上限以内;逐条 forward 不可能。
    ▸ 为什么前向次数少就等于「批了」:每生成一个 token 都要一次前向产出它的 logits;总次数少于
      生成 token 总数,就必然有某次前向同时为 >= 2 条请求产出了 token。
    """
    reqs = [("a", _prompt(101, 3), 12), ("b", _prompt(102, 11), 6), ("c", _prompt(103, 1), 16), ("d", _prompt(104, 25), 9)]
    expected = _expected(model, reqs)
    run = EngineRun(model, "E1")
    await run.go(reqs)
    run.assert_outputs(expected)
    n_seq = sum(n for _, _, n in reqs)
    bound = len(reqs) + (len(reqs) - 1) + max(n for _, _, n in reqs) - 1
    assert run.counter.n <= bound, (
        f"[E1] 4 条并发请求一共触发了 {run.counter.n} 次前向,上限 {bound};逐条跑(不批)正好是 {n_seq} 次。"
        f"{run.call_summary()}\n" + explain("admission") + "\n" + explain("one_forward")
    )


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_e2_single_request_degenerates_correctly(model):
    """batch=1 退化(引擎层):只有一条请求时输出正确,且前向次数不超过「生成 token 数 + 1」。

    ▸ 挂了的含义:批量路径在只有一行时多做了前向(比如每步还额外空跑一次),或者单行时位置/缓存算错。
      M3 benchmark 的「顺序处理」基线就是并发 1 —— 这条不稳,基线就不可信。
    """
    reqs = [("solo", _prompt(111, 7), 10)]
    expected = _expected(model, reqs)
    run = EngineRun(model, "E2")
    await run.go(reqs)
    run.assert_outputs(expected)
    assert run.counter.n <= 10 + 1, (
        f"[E2] 单条请求生成 10 个 token 用了 {run.counter.n} 次前向(应 <= 11)。{run.call_summary()}\n"
        + explain("one_forward")
    )


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_e3_requests_joining_while_others_decode(model):
    """新请求中途加入:a(prompt 5,生成 20)、b(prompt 40,生成 18)先跑;第 4 次前向之后再来
    c(prompt 60,比在飞的都长)和 d(prompt 1)。四条输出都要正确,总前向次数 < 逐条之和的 75%。
    """
    start = [("a", _prompt(121, 5), 20), ("b", _prompt(122, 40), 18)]
    late = [(4, ("c", _prompt(123, 60), 15)), (4, ("d", _prompt(124, 1), 12))]
    everything = start + [r for _, r in late]
    expected = _expected(model, everything)
    run = EngineRun(model, "E3")
    await run.go(start, late)
    run.assert_mid_flight("c")
    run.assert_outputs(expected)
    n_seq = sum(n for _, _, n in everything)
    assert run.counter.n < 0.75 * n_seq, (
        f"[E3] 中途加入之后总前向 {run.counter.n} 次,逐条跑是 {n_seq} 次,要求 < {0.75 * n_seq:.0f}。"
        f"新来的请求没有并进正在跑的批。{run.call_summary()}\n" + explain("admission")
    )


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_e4_finished_request_leaves_the_batch(model):
    """某行先结束:三条请求生成 2 / 6 / 14 个 token。①输出都正确;②最后 8 次前向每次只处理 1 个位置
    —— 那时只剩最长的那条在飞,另外两条已经 DONE,不该还在批里陪跑。

    ▸ 为什么是「最后 8 次」(和准入策略无关,只假设按入队顺序准入):第三条最晚在第二条之后同一轮开始,
      它还需要的 decode 步数比第二条多 8。
    ▸ 这条和正确性无关:陪跑的行算出来的 token 被扔掉,输出照样对 —— 但每一步白算一行,
      请求源源不断进来时批只涨不缩。
    """
    reqs = [("short", _prompt(131, 9), 2), ("mid", _prompt(132, 3), 6), ("long", _prompt(133, 20), 14)]
    expected = _expected(model, reqs)
    run = EngineRun(model, "E4")
    await run.go(reqs)
    run.assert_outputs(expected)
    run.assert_actually_batched(reqs, "结束的行移出批")
    tail = [run.counter.rows(i) for i in range(run.counter.n - 8, run.counter.n)]
    assert all(r == 1 for r in tail), (
        f"[E4] 最后 8 次前向处理的位置数是 {tail},应当全是 1(那时只剩 long 一条在飞)。{run.call_summary()}\n"
        "  如果你是有意用固定槽位(结束的行留着占位、只 mask 掉),在 PR 里论证它在 CPU 上的代价,并和 manager 讨论这条测试。\n"
        + explain("zombie")
    )


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_e5_extreme_prompt_length_gap(model):
    """极端长度差(引擎层):prompt 1 个 token 生成 30 个,和 prompt 110 个 token 生成 18 个(正好写满 128 的上下文)同批。"""
    reqs = [("tiny", _prompt(141, 1), 30), ("huge", _prompt(142, 110), 18)]
    expected = _expected(model, reqs)
    run = EngineRun(model, "E5")
    await run.go(reqs)
    run.assert_outputs(expected)
    run.assert_actually_batched(reqs, "极端长度差的两行同批")


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_e6_staggered_arrivals_mixed_lengths(model):
    """压力混合:10 条请求,prompt 长度 1–60、生成 1–20 个 token(含「只生成 1 个、prefill 完就结束、
    根本不该进批」的请求),5 条开跑前入队,5 条在第 3/6/9/14/20 次前向之后陆续到达。
    全部输出正确,总前向次数不超过「最晚完成的那条的时间线 + 2N−1」的上限。
    """
    start = [
        ("s0", _prompt(151, 17), 20), ("s1", _prompt(152, 1), 7), ("s2", _prompt(153, 42), 1),
        ("s3", _prompt(154, 8), 13), ("s4", _prompt(155, 29), 4),
    ]
    late = [
        (3, ("l0", _prompt(156, 60), 9)), (6, ("l1", _prompt(157, 2), 1)), (9, ("l2", _prompt(158, 33), 16)),
        (14, ("l3", _prompt(159, 5), 11)), (20, ("l4", _prompt(160, 12), 3)),
    ]
    everything = start + [r for _, r in late]
    expected = _expected(model, everything)
    run = EngineRun(model, "E6")
    await run.go(start, late)
    for _, (rid, _, _) in late:
        run.assert_mid_flight(rid)
    run.assert_outputs(expected)
    N = len(everything)
    bound = max([n for _, _, n in start] + [k + n for k, (_, _, n) in late]) + 2 * N - 1
    n_seq = sum(n for _, _, n in everything)
    assert run.counter.n <= bound, (
        f"[E6] 总前向 {run.counter.n} 次,上限 {bound},逐条跑是 {n_seq} 次。{run.call_summary()}\n"
        + explain("admission")
    )


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_e7_identical_prompts_do_not_collapse(model):
    """三条请求 prompt 完全相同、生成长度不同(10 / 4 / 7):输出各自是同一条参照答案的前缀,互不串。"""
    same = _prompt(171, 8)
    reqs = [("x", same, 10), ("y", list(same), 4), ("z", list(same), 7)]
    expected = _expected(model, reqs)
    run = EngineRun(model, "E7")
    await run.go(reqs)
    run.assert_outputs(expected)
    run.assert_actually_batched(reqs, "相同 prompt 的几行在同一批里互不串")


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_e8_rejected_request_does_not_disturb_the_batch(model):
    """两条请求正在批量 decode 时,来一条「prompt 长度 == max_token_length」的坏请求(准入就该拒)
    和一条正常请求:坏的 FAILED 且一个 token 都没有,其余三条 DONE 且输出正确。
    """
    start = [("a", _prompt(181, 6), 14), ("b", _prompt(182, 15), 12)]
    late = [(3, ("bad", _prompt(183, 5), 0)), (3, ("c", _prompt(184, 4), 6))]
    good = start + [late[1][1]]
    expected = _expected(model, good)
    run = EngineRun(model, "E8")
    await run.go(start, late)
    run.assert_mid_flight("bad")
    assert run.store["bad"].status == RequestStatus.FAILED, (
        f"[E8] prompt 长度 == max_token_length 的请求状态是 {run.store['bad'].status.name},应为 FAILED。\n"
        + explain("admission")
    )
    assert run.store["bad"].generated_tokens == [], "[E8] 被拒的请求不该有任何生成 token。"
    run.assert_outputs(expected)
    run.assert_actually_batched(good, "被拒的请求不打扰正在跑的批")


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_e9_streaming_view_for_the_benchmark(model):
    """M3 benchmark 量 TTFT / ITL 靠 m2_adapters.generated_so_far:引擎每让出一次控制权,benchmark
    就读一次。这条验它的契约:①每次读到的都是上一次的前缀(只增不减);②在飞期间能看到中间状态
    (长请求至少被看到一半的中间长度);③DONE 之后等于 request_store 里的最终结果。
    """
    reqs = [("p", _prompt(191, 4), 16), ("q", _prompt(192, 21), 9)]
    expected = _expected(model, reqs)
    seen: Dict[str, List[List[int]]] = {rid: [[]] for rid, _, _ in reqs}
    run = EngineRun(model, "E9")

    def monitor() -> None:
        for rid in seen:
            if rid in run.store:
                cur = list(m2.generated_so_far(run.engine, run.store, rid))
                if cur != seen[rid][-1]:
                    seen[rid].append(cur)

    await run.go(reqs, monitor=monitor)
    run.assert_outputs(expected)
    run.assert_actually_batched(reqs, "批量路径上的流式视图")
    for rid, views in seen.items():
        for prev, cur in zip(views, views[1:]):
            assert cur[: len(prev)] == prev, (
                f"[E9] generated_so_far('{rid}') 先读到 {prev},后读到 {cur} —— 后一次不是前一次的延长。"
                f"\n  回看:{DOC['spec']}"
            )
        assert views[-1] == expected[rid], f"[E9] generated_so_far('{rid}') 最后读到 {views[-1]},最终结果是 {expected[rid]}。"
    lengths_seen = {len(v) for v in seen["p"]}
    assert len(lengths_seen) >= 16 // 2, (
        f"[E9] 长请求 p 在飞期间只被看到了 {sorted(lengths_seen)} 这几个长度 —— generated_so_far 读不到中间状态,"
        f"或者引擎很多轮才让出一次控制权。benchmark 会因此把 TTFT 量成「整条请求的完成时间」。\n  回看:{DOC['spec']}"
    )


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_e10_nondraining_shutdown_mid_batch_fails_every_row(model):
    """LMS-7 的 T7 推广到批量路径:三条请求正在批量 decode 时 shutdown(draining=False),
    三条都 FAILED,引擎停下、task 结束,join() 不挂。
    """
    reqs = [("a", _prompt(201, 5), 30), ("b", _prompt(202, 12), 30), ("c", _prompt(203, 2), 30)]
    queue: asyncio.Queue = asyncio.Queue()
    store: Dict[str, RequestData] = {}
    engine = m2.build_engine(model, queue, store, NO_EOS)
    for rid, prompt, n in reqs:
        _push(queue, store, rid, prompt, n)
    counter = HeadCounter(model)
    fire = asyncio.Event()
    counter.on_call = lambda n: fire.set() if n >= 3 + 3 else None
    try:
        task = engine.run()
        await asyncio.wait_for(fire.wait(), timeout=20)
        at_fire = {rid: store[rid].status for rid, _, _ in reqs}
        try:
            await asyncio.wait_for(engine.shutdown(draining=False), timeout=10)
            await asyncio.wait_for(task, timeout=5)
        except asyncio.TimeoutError:
            raise AssertionError(
                f"[E10] shutdown(draining=False) 10s 内没返回(或引擎 task 没结束)。此刻状态:"
                f"{ {rid: store[rid].status.name for rid, _, _ in reqs} } —— 批里还有行没被标 FAILED、"
                f"或者它们的 task_done 没调,join() 永远等不到归零。\n" + explain("shutdown")
            )
    finally:
        counter.close()
    assert all(s not in (RequestStatus.DONE, RequestStatus.FAILED, RequestStatus.PENDING) for s in at_fire.values()), (
        f"[E10] 前提没满足:开枪时状态是 { {r: s.name for r, s in at_fire.items()} },不是三条都在飞。"
    )
    for rid, _, _ in reqs:
        assert store[rid].status == RequestStatus.FAILED, (
            f"[E10] non-draining shutdown 之后 {rid} 的状态是 {store[rid].status.name},应为 FAILED(三条都在批里)。"
            f"批量路径上的「全部作废」漏了行。\n" + explain("shutdown")
        )
