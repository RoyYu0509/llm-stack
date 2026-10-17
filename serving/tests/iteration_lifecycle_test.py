"""
LMS-7(GitHub issue #11)—— request / iteration lifecycle 的正确性门禁。

这份测试文件不测「模型算得对不对」(那是 lm_engine_test.py 的事),测的是控制流:引擎
是不是真的在「每一轮循环里,把所有在飞请求各推进一步」,而不是「取一条请求跑到底」。

所有测试用一个确定性的 stub 模型(SteppingStubModel),不加载真实权重(T10 除外,它专门
用来做贪心回归)。stub 的规则极简:下一个 token = 当前序列最后一个 token 的值 + 1,用
one-hot logits 表达,argmax 正好取到这个值。因为 TransformerLMAdapter 的 recompute 路径
(use_kv_cache=False,默认)会把 prefix 原样传回下一步,而且每条测试请求的 prompt 长度都
固定为 1 个 token,所以每次真正调用 forward() 时,序列第一个位置永远是这条请求的「身份
token」,序列长度减一就是这一步是第几次调用(0 = prefill,1/2/3... = 第几次 decode)。stub
在每次 forward 里把这些信息按行记下来,还顺手记录调用那一刻 request_store 里的 status——
这样测试完全不需要真的跑一个语言模型,就能精确断言"谁在什么时候被推进了第几步、当时状态是什么"。


怎么读这份文件(第一次读的建议顺序)
====================================
不要从头往下顺着读。读这三段就够开工了:

    1. SteppingStubModel.forward()    —— 全部机关都在这 20 行里。看懂它,后面每一条断言
                                         为什么"能知道谁在第几步被推进"就都清楚了。
    2. 任意一条测试(推荐 T1)             —— 10 条测试是同一个 5 步骨架换参数,看懂一条 = 看懂十条。
    3. 出问题时再回来看具体那条的断言消息。

这份文件里反复出现的四个写法,先认一下,后面就不用再猜:

  ┌ push_stub_request(queue, store, "R1", 10, max_length=1 + 3)
  │   造一条假请求直接塞进队列,绕开真 tokenizer。"R1" 是请求名,10 是它的「身份 token」
  │   (见 SteppingStubModel),1 是 prompt 长度(本文件所有请求的 prompt 都是 1 个 token),
  │   3 是「这条请求还要生成几个 token」。所以 max_length 一律写成 `1 + N` 的形式,N 一眼可读。
  │   身份 token 是 10,生成出来的 token 就是 11, 12, 13 —— 断言里那些数字是这么来的。
  │
  ├ task = engine.run()
  │   run() 是**同步**函数,调用它就已经把引擎跑起来了,返回的是一个**正在跑的 Task**。
  │   所以这里是 `task = engine.run()`,不是 `await engine.run()`。
  │   (你 09-03 踩过这个坑,这份文件里 10 处全是这个写法,可以当对照。)
  │
  ├ await asyncio.wait_for(pending_queue.join(), timeout=5)
  │   等「队列的待办计数归零」——注意 join() 等的不是"队列空了",而是"每一个 put 进去的元素
  │   都被调用过一次 task_done()"。这两本账是分开的。测试用 join() 当「所有活都干完了」的信号。
  │
  └ calls / CallRecord
      stub 每被调用一次就往 `calls` 这个 list 里追加一条流水账(谁、第几步、当时什么状态)。
      测试和 stub 拿的是**同一个 list 对象**(引用传递),所以引擎跑完之后测试直接读它就行。
      所有关于「顺序」的断言,读的都是这个 list。
"""

import asyncio
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import pytest
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from inferenceLM.data.request import RequestData
from inferenceLM.data.request_status import RequestStatus
from inferenceLM.data.tokenized_data import TokenizedData
from inferenceLM.engine.inference_engine import InferenceEngine
from inferenceLM.engine.inference_engine_status import InferenceEngineStatus
from inferenceLM.engine.transformer_lm_adapter import TransformerLMAdapter
from inferenceLM.request_receiver.request_receiver import RequestReceiver

# ---------------------------------------------------------------------------
# T9 的裁决常量 —— D-5 只锁定了"DRAINING 期间 / STOPPED 之后停止接收新请求"这条政策本身,
# 没有锁定"拒绝这件事怎么传回调用方"——那是留给你的 ADR 决定,不是这份测试替你选的。
#   None(默认):还没做出裁决。测试只强制 policy-agnostic 的那条不变量本身(见下面
#     submit_late_and_check 的 None 分支的精确定义:要么 submit_request 直接 raise,
#     要么迟到请求在超时内离开 PENDING 并且落在 FAILED——不能是 DONE,DONE 意味着它被
#     真正接收处理了,和"停止接收"矛盾)。如果这条不变量已经满足,测试会用 pytest.fail()
#     提醒你:先在 ADR 里写清楚 "raise" 还是 "FAILED",再回来把这个常量设成对应值——
#     不能因为不变量侥幸满足就绕过这道裁决。
#   "raise" : submit_request 在 DRAINING 期间 / STOPPED 之后被调用时,直接抛异常。
#   "FAILED": submit_request 正常返回一个 request_id,但这条请求在有限时间内被标成
#             RequestStatus.FAILED,不会永远停在 PENDING。
# 两者都满足「不能永远停在 PENDING」这条底线,选哪个是你的设计决定;设成其中之一之后,
# 测试才会用你选的那条具体规则去精确比对。
EXPECTED_LATE_OUTCOME = "raise"  # 如果 引擎停止后再提交,直接 raise, 不要让他进到 engine 中


@dataclass
class CallRecord:
    """一次模型调用的流水账。stub 每处理一行输入就产生一条。

    举例:calls == [R1@0, R2@0, R1@1, R2@1, R1@2] 读作
    「先推进 R1 的 prefill,再推进 R2 的 prefill,再推进 R1 的第 1 次 decode……」。
    """

    request_id: str  # 这一行是哪条请求的(靠身份 token 反查出来的)
    step: int  # 0 = prefill;1, 2, 3... = 第几次 decode
    status: Optional[RequestStatus]  # 调用发生的**那一刻** request_store 里的 status(T4 靠它)


class SteppingStubModel(nn.Module):
    """确定性 stub:(batch, seq) -> (batch, seq, vocab)。

    每一行的规则是「下一个 token = 这一行当前最后一个 token 的值 + 1」,用 one-hot logits
    表达,argmax 之后正好取到这个值。第一个位置的 token 是这条请求的身份标记,整个生成过程中
    不变(因为 TransformerLMAdapter 的 recompute 路径把 prefix 原样向后传)。

    forward 每次被调用,不管这次调用装的是 1 行还是多行,都按行展开记录 —— 这样同一份测试
    对「每条请求各自单独调一次模型」和「拼成一个 batch 一次调」这两种实现都成立。
    """

    def __init__(
        self,
        id_to_request: Dict[int, str],
        request_store: Dict[str, RequestData],
        calls: List[CallRecord],
        vocab_size: int = 4096,
        on_call: Optional[Callable[[List[CallRecord]], None]] = None,
        raise_at: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__()
        # TransformerLMAdapter.device 读 next(self.model.parameters()).device,
        # stub 自己不需要任何可学习参数,但必须有一个,否则 parameters() 是空迭代器会 StopIteration。
        self._device_anchor = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.id_to_request = id_to_request
        self.request_store = request_store
        self.calls = calls
        self.vocab_size = vocab_size
        self.on_call = on_call
        self.raise_at = raise_at or {}

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """一次"模型调用"。真模型在这里算 attention,这个 stub 只做两件事:记账 + 编出下一个 token。

        具体例子(请求 R1,身份 token = 10,max_length = 1 + 3):

            第 1 次调用(prefill)  input_ids = [[10]]           seq=1 → step=0 → 产出 11
            第 2 次调用(decode 1) input_ids = [[10, 11]]       seq=2 → step=1 → 产出 12
            第 3 次调用(decode 2) input_ids = [[10, 11, 12]]   seq=3 → step=2 → 产出 13

        两个关键点:
          · **第 0 列永远是 10** —— 因为 TransformerLMAdapter 的 recompute 路径每次都把整段
            prefix 原样喂回来。所以 input_ids[row, 0] 就是这一行的身份,查表即知是谁。
          · **seq - 1 就是第几步** —— 因为 prompt 固定 1 个 token, 序列长了几个就是 decode 过几次。

        这两点合起来,就让 stub 在完全不跑语言模型的情况下,知道"现在正在推进谁的第几步"。
        """
        batch, seq = input_ids.shape
        logits = torch.zeros(batch, seq, self.vocab_size)
        new_records: List[CallRecord] = []

        # 按行展开:一次调用可能装 1 行(一条请求单独调),也可能装 N 行(拼成 batch 一次调)。
        # 两种实现都能通过这份测试,因为记账是按行记的,不是按调用次数记的。
        for row in range(batch):
            identity = int(input_ids[row, 0].item())  # 第 0 列 = 身份 token
            request_id = self.id_to_request.get(identity, f"<unknown:{identity}>")
            # 在**调用发生的这一刻**抓一张 status 快照。跑完再看就晚了——T4 要验的正是
            # "prefill 被调用时 status 已经是 PREFILLING 了吗",事后查看不到这个时刻。
            status = (
                self.request_store[request_id].status
                if request_id in self.request_store
                else None
            )
            step = seq - 1  # 本文件所有测试请求的 prompt 长度都固定为 1
            record = CallRecord(request_id=request_id, step=step, status=status)
            self.calls.append(record)  # 注意:self.calls 和测试里的 calls 是同一个 list
            new_records.append(record)

            # 故障注入:T5b 用 raise_at={"BAD": 2} 让 BAD 这条在第 2 次 decode 时炸掉,
            # 用来验证"一条请求挂了,别的请求还得继续跑完"。
            if self.raise_at.get(request_id) == step:
                if self.on_call is not None:
                    self.on_call(new_records)
                raise RuntimeError(
                    f"stub: 故意让 {request_id} 在第 {step} 步(0=prefill)失败"
                )

            # 造 logits:在"最后一个 token 的值 + 1"那个位置放 1.0,其余全 0。
            # LMEngine 取 logits[:, -1, :] 再 argmax,正好拿到 last_value + 1。
            last_value = int(input_ids[row, -1].item()) # 拿到最后一个 token 的值
            logits[row, -1, last_value + 1] = 1.0 # 在 logits[row, -1, last_value + 1] 放 1.0,其余全 0

        # on_call 是给测试用的钩子:让测试能在"某一步刚被推进"的瞬间做事
        # (T3 用它触发第二条请求的提交,T7/T8 用它当 shutdown 的发令枪)。
        if self.on_call is not None:
            self.on_call(new_records)
        return logits


def make_stub_adapter(
    # 把假模型装进你自己的 adapter,交出一个 LMEngine 能直接驱动的东西。
    #
    #   SteppingStubModel     ← 假模型,只认 (batch, seq) -> (batch, seq, vocab),测试文件自带
    #        ↓ 被包进
    #   TransformerLMAdapter  ← **你自己仓库里的真代码**,补上 LMEngine 依赖的 4 个 HF 约定:
    #                            model.eval() / model(input_ids, past_key_values=..., use_cache=True)
    #                            -> .logits, .past_key_values / config.n_positions / config.eos_token_id
    #        ↓ 被交给
    #   LMEngine / InferenceEngine  ← 真代码,也就是这次要测的对象
    #
    # 假模型能塞进来,正是因为 adapter 当初就是按"接任意 (batch, seq) -> logits 模块"设计的。
    #
    # 三个参数全是**引用传递** —— 测试手里的 dict/list 和 stub 手里的是同一个对象:
    #   id_to_request : {10: "R1", 20: "R2"},身份 token → 请求名。stub 在张量里只看得见数字。
    #   request_store : 让 stub 能在被调用的那一刻抓 status 快照(T4 全靠它)。
    #   calls         : stub 往里写,测试从里读。所有顺序断言的数据源。
    id_to_request: Dict[int, str],
    request_store: Dict[str, RequestData],
    calls: List[CallRecord],
    on_call: Optional[Callable[[List[CallRecord]], None]] = None,
    raise_at: Optional[Dict[str, int]] = None,
    vocab_size: int = 4096,
) -> TransformerLMAdapter:
    stub = SteppingStubModel(
        id_to_request,
        request_store,
        calls,
        vocab_size=vocab_size,
        on_call=on_call,
        raise_at=raise_at,
    )
    return TransformerLMAdapter(
        stub,
        context_length=4096,
        eos_token_id=-1,  # 永远不会被生成到的值,确保只有 max_length 能触发停止
        vocab_size=vocab_size,
    )


def make_system(model, pending_queue=None, request_store=None):
    """
    把 receiver + engine 接上同一个 queue 和同一个 store,返回四件套。

    返回 (receiver, engine, pending_queue, request_store)。多数测试写成
    `_, engine, pending_queue, request_store = make_system(...)` —— 开头那个 `_` 是
    Python 惯例,意思是"这个返回值我不要":这些测试用 push_stub_request 直接塞队列,
    绕开了真 tokenizer,所以不需要 receiver。只有 T9 要测"迟到提交",那里才写成
    `receiver, engine, ... = make_system(...)`。

    queue 和 store 既是入参又是返回值:不传就新建一个还给你,传了就用你给的那个。

    ADAPT HERE:如果你改了 RequestReceiver / InferenceEngine 的构造参数(比如给 receiver
    传一个共享的 shutdown 信号、或者把 pending_queue/request_store 包进一个新的协调对象),
    只改这个函数本身,不要去改下面每一个测试。
    """
    if pending_queue is None:
        pending_queue = asyncio.Queue()
    if request_store is None:
        request_store = {}

    engine = InferenceEngine(model, pending_queue, request_store)
    receiver = RequestReceiver("gpt2", pending_queue, request_store, engine=engine)

    return receiver, engine, pending_queue, request_store


def push_stub_request(
    pending_queue: asyncio.Queue,
    request_store: Dict[str, RequestData],
    request_id: str,
    identity_token: int,
    max_length: int,
    do_sample: bool = False,
) -> None:
    """
    绕开 RequestReceiver 的真实分词器,直接造一条身份 token 可控的请求 —— 和现有
    inference_engine_test.py::test_inference_engine_async_fetch_all_requests 里已经在用的
    手法一致(手写 RequestData + TokenizedData,直接放进 request_store / pending_queue)。

    为什么不用真 tokenizer:真 tokenizer 给出的 token id 不可控,stub 就没法靠"第 0 列是几"
    反查出这行是谁。这里手工指定 identity_token,就把这件事钉死了。

    做的两件事,和 RequestReceiver.submit_request() 的后两步一一对应:
      ① request_store[request_id] = RequestData(... status=PENDING ...)   ← 存档
      ② pending_queue.put_nowait(TokenizedData(...))                      ← 入队
    注意入队的是 TokenizedData,不是 RequestData —— 队列里流的和 store 里存的是两个对象。

    max_length 的读法:本文件一律写成 `1 + N`。1 是 prompt 长度(就是那个 identity_token),
    N 是这条请求还要生成几个 token。例:identity_token=10, max_length=1+3 → 生成 [11, 12, 13]。
    """
    request_store[request_id] = RequestData(
        request_id=request_id,
        user_id=f"user_{request_id}",
        timestamp=0.0,
        prompt_text=f"stub prompt for {request_id}",
        status=RequestStatus.PENDING,
        max_token_length=max_length,
        do_sample=do_sample,
    )
    pending_queue.put_nowait(TokenizedData(request_id=request_id, tokens=[identity_token]))


def alive_window(calls: List[CallRecord], id_a: str, id_b: str) -> List[CallRecord]:
    """从 calls 里切出"两条请求都在飞"的那一段。

    为什么需要它:开头那几步可能只有 R1 在跑(R2 还没被收进来),结尾那几步可能只剩 R1
    在收尾(R2 已经跑完了)。这两头本来就不该有"交替"可言。所以只截中间那段:
    从"两条都已经至少被推进过一次"开始,到"其中一条最后一次被推进"为止。

        calls   = [R1, R1, R2, R1, R2, R1, R1]
                           ↑ start  ↑ end   (R2 最后一次出现在这之前)
        window  = [R2, R1, R2]

    这么切之后,"同一条不能连着推进两步"这条断言就和你选哪种准入策略无关了 ——
    T2 要的正是这种策略无关的不变量,而 T1 要的是 AC 写死的那个精确顺序。两条测不同的东西。
    """
    pos_a = [i for i, c in enumerate(calls) if c.request_id == id_a]
    pos_b = [i for i, c in enumerate(calls) if c.request_id == id_b]
    assert pos_a, f"{id_a} 从未被推进过"
    assert pos_b, f"{id_b} 从未被推进过"
    start = max(pos_a[0], pos_b[0])
    end = min(pos_a[-1], pos_b[-1])
    return calls[start : end + 1]


async def wait_until(predicate: Callable[[], bool], timeout: float, description: str) -> None:
    """反复检查一个条件,直到它成立或超时。超时就报一条人话的失败消息,而不是干等到 pytest 超时。

    只在"没有更好的信号可用"时才用它(目前只有 T9)。能用事件钩子(on_call)当信号的地方
    一律用钩子——stub 快到经常在第一次采样之前就把整条请求跑完了,轮询会直接错过中间状态。
    """

    async def _poll():
        while not predicate():
            await asyncio.sleep(0.01)

    try:
        await asyncio.wait_for(_poll(), timeout=timeout)
    except asyncio.TimeoutError:
        raise AssertionError(
            f"等待『{description}』超时({timeout}s 内条件一直没有满足)"
        )


# ---------------------------------------------------------------------------
# T1 —— AC 原文的精确顺序
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_t1_ac_exact_alternation_order():
    """
    AC 原文:R1(3 token)、R2(2 token)在 run() 之前都已入队,观测到的顺序应该是
    R1_t1, R2_t1, R1_t2, R2_t2, R1_t3 —— 不是 R1 跑完才开始 R2。

    ▸ 场景:两条请求在引擎启动**之前**就都躺在队列里了,谁也没有先后可言。
    ▸ 查的是:calls 的顺序。这是全文件唯一一条写死精确顺序的测试。
    ▸ 挂了的含义:看断言消息里打出来的实际顺序。
        ["R1","R1","R1","R2","R2"]  → 还是"一条跑到底再跑下一条",循环根本没拆开。
        其他顺序                     → 循环拆开了,但你的准入策略和 AC 假设的不是同一个。
                                       这时候要做的是论证 + 走 ADR 改 AC,不是回来改这个测试。

    ⚠️ 下面这 5 步是本文件 10 条测试共用的骨架,只在这条里逐行注释,其余 9 条不再重复。
    """
    # ── 第 1 步:搭台子 ────────────────────────────────────────────────
    pending_queue = asyncio.Queue()
    request_store: Dict[str, RequestData] = {}
    calls: List[CallRecord] = []  # 空账本,待会儿 stub 往里写
    id_to_request = {10: "R1", 20: "R2"}  # 身份 token → 请求名;数字随便取,只要不撞车
    adapter = make_stub_adapter(id_to_request, request_store, calls)
    _, engine, pending_queue, request_store = make_system(adapter, pending_queue, request_store)

    # ── 第 2 步:往队列里放请求 ────────────────────────────────────────
    # R1:身份 10,要生成 3 个 token → [11, 12, 13]
    # R2:身份 20,要生成 2 个 token → [21, 22]
    push_stub_request(pending_queue, request_store, "R1", 10, max_length=1 + 3)
    push_stub_request(pending_queue, request_store, "R2", 20, max_length=1 + 2)

    # ── 第 3 步:开跑 ──────────────────────────────────────────────────
    task = engine.run()  # 同步调用,返回一个**已经在跑的** Task;不要 await 它

    # ── 第 4 步:等活干完 ──────────────────────────────────────────────
    # join() 等的是"每个 put 进去的元素都被 task_done() 过一次",不是"队列空了"。
    # wait_for 套在外面是保险丝:真卡住了 5 秒后报错,而不是让测试挂死。
    await asyncio.wait_for(pending_queue.join(), timeout=5)

    # ── 第 5 步:收摊 + 验收 ───────────────────────────────────────────
    await engine.shutdown(draining=True)
    assert task.done()  # 引擎真的停了,不是还在后台转

    # 先验"结果对不对"(这部分不管顺序,只管两条都跑完了、token 没串行)
    assert request_store["R1"].status == RequestStatus.DONE
    assert request_store["R2"].status == RequestStatus.DONE
    assert request_store["R1"].generated_tokens == [11, 12, 13]  # 身份 10 → 11,12,13
    assert request_store["R2"].generated_tokens == [21, 22]  # 身份 20 → 21,22

    # 再验"顺序对不对" —— 这条才是 LMS-7 的重点
    order = [c.request_id for c in calls]
    assert order == ["R1", "R2", "R1", "R2", "R1"], (
        "AC 原文顺序应为 R1,R2,R1,R2,R1(对应 R1_t1,R2_t1,R1_t2,R2_t2,R1_t3);"
        f"实际观测顺序 {order}。这条隐含了一个准入假设——每轮开头把队列里已有的请求尽量都"
        "收进来,且同一轮内按准入顺序推进各自一步。如果你的设计选了别的准入策略(比如每轮"
        "最多收 1 条新请求),这个顺序会变——去 handout Part A 的准入假设推演一节确认你的"
        "设计是否满足它;如果不满足,应该在 CORE gate 上论证清楚再通过 ADR 修改这条 AC,"
        "而不是回来改这个测试让它糊过去。"
    )


# ---------------------------------------------------------------------------
# T2 —— 与策略无关的交替不变量
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_t2_alternation_invariant_while_both_in_flight():
    """不依赖具体准入策略的不变量:只要 R1、R2 都已经开始跑、且都还没结束,同一条请求就
    不会连续推进两步。这条对"每轮最多收 1 条"这类策略也成立——窗口是从两者各自第一次被
    推进,到其中一条最后一次被推进为止,不包含窗口之外任何一方还没开始的部分。

    ▸ 和 T1 的分工:T1 钉死"必须是 R1,R2,R1,R2,R1 这一个顺序";T2 只要求"在两条都在飞的
      那段窗口里,不许同一条连着走两步"。你的准入策略换了,T1 可能要重谈,T2 永远该过。
    ▸ 查的是:alive_window(calls, "R1", "R2") 切出来那一段,相邻两条不能同名。
    ▸ 挂了的含义:窗口里出现了 [.., R1, R1, ..] —— 有人在另一条明明还在飞的时候,
      连着独占了两步。
    """
    pending_queue = asyncio.Queue()
    request_store: Dict[str, RequestData] = {}
    calls: List[CallRecord] = []
    id_to_request = {30: "R1", 40: "R2"}
    adapter = make_stub_adapter(id_to_request, request_store, calls)
    _, engine, pending_queue, request_store = make_system(adapter, pending_queue, request_store)

    push_stub_request(pending_queue, request_store, "R1", 30, max_length=1 + 4)
    push_stub_request(pending_queue, request_store, "R2", 40, max_length=1 + 6)

    task = engine.run()
    await asyncio.wait_for(pending_queue.join(), timeout=5)
    await engine.shutdown(draining=True)
    assert task.done()

    assert request_store["R1"].generated_tokens == [31, 32, 33, 34]
    assert request_store["R2"].generated_tokens == [41, 42, 43, 44, 45, 46]

    window = alive_window(calls, "R1", "R2")
    # 前置条件:两条必须真的同时在飞过。没有这一句,"一条跑到底再跑下一条"的实现会得到一个
    # 空窗口,下面的 for 一次都不执行,测试就假绿了(2026-09-17 在未改动的引擎上实测到过)。
    assert window, (
        "R1、R2 从来没有同时在飞过——R2 直到 R1 最后一步之后才第一次被推进,"
        f"实际顺序 {[c.request_id for c in calls]}。这还是'一条跑到底'。"
    )
    for prev, cur in zip(window, window[1:]):
        assert prev.request_id != cur.request_id, (
            "R1、R2 都还在飞的窗口内,观测到同一条请求连续推进了两步:"
            f"{[c.request_id for c in window]}"
        )


# ---------------------------------------------------------------------------
# T3 —— 飞行中准入
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_t3_admits_a_request_submitted_while_another_is_mid_flight():
    """R2 在 R1 已经开始跑之后才提交(R1 的第一步——也就是 prefill——完成才触发提交)。
    这个测试专门抓一类实现:忙时循环从头到尾不 await 任何东西,导致 receiver/producer 这类
    并发协程根本没有机会被调度——如果撞上这类实现,R2 会一直等到 R1 彻底跑完之后才第一次
    被推进,而不是在 R1 还在飞的时候就插进来。

    ▸ 和 T1/T2 的分工:那两条的请求在开跑前就都进队了。这一条的 R2 是**中途**才来的。
    ▸ 机关:on_call 钩子盯着"R1 的 prefill 刚做完"这一刻,拿它当发令枪去提交 R2 —— 用事件
      当信号,不用 sleep 赌时间。
    ▸ 查的是:R2 第一次出现的位置 < R1 最后一次出现的位置。也就是"R2 插进来的时候 R1 还在飞"。
    ▸ 挂了的含义:忙的时候整轮循环一个 await 都没有,事件循环没机会把控制权交给别的协程,
      于是新请求根本进不来。
    """
    pending_queue = asyncio.Queue()
    request_store: Dict[str, RequestData] = {}
    calls: List[CallRecord] = []
    id_to_request = {50: "R1", 60: "R2"}
    r1_first_step_done = asyncio.Event()

    def on_call(records: List[CallRecord]) -> None:
        for record in records:
            # R1 的 prefill 调用一返回，就触发 `r1_first_step_done` event
            if record.request_id == "R1" and record.step == 0:
                r1_first_step_done.set()

    adapter = make_stub_adapter(id_to_request, request_store, calls, on_call=on_call)
    _, engine, pending_queue, request_store = make_system(adapter, pending_queue, request_store)

    push_stub_request(pending_queue, request_store, "R1", 50, max_length=1 + 6)

    async def submit_r2_after_r1_starts() -> None:
        # 等 `r1_first_step_done` event 被触发之后, 立刻把 R2 推进 queue 中
        await r1_first_step_done.wait()
        push_stub_request(pending_queue, request_store, "R2", 60, max_length=1 + 2)
    producer = asyncio.create_task(submit_r2_after_r1_starts())

    # 启动 engine -> R1 会被拿 + prefill -> on_call 触发 -> R2 被推进queue中 -> engine 继续推进 R1/R2
    task = engine.run()
    await asyncio.wait_for(producer, timeout=5) # 设置一个超时，防止 producer 卡住
    await asyncio.wait_for(pending_queue.join(), timeout=5)
    await engine.shutdown(draining=True)
    await asyncio.wait_for(task, timeout=7)  # 确保 engine 任务完成
    assert task.done()

    assert request_store["R1"].status == RequestStatus.DONE
    assert request_store["R2"].status == RequestStatus.DONE

    r1_positions = [i for i, c in enumerate(calls) if c.request_id == "R1"]
    r2_positions = [i for i, c in enumerate(calls) if c.request_id == "R2"]
    assert r2_positions, "R2 应该至少被推进过一次,但 calls 里完全没有它的记录"
    assert r2_positions[0] < r1_positions[-1], (
        f"R2 直到 R1 结束后才第一次被推进(R2 首次出现在 calls[{r2_positions[0]}],"
        f"R1 最后一次出现在 calls[{r1_positions[-1]}],calls 总长度 {len(calls)})。"
    )


# ---------------------------------------------------------------------------
# T4 —— 状态转换
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_t4_prefilling_then_decoding_status_transitions():
    """prefill 那次 forward 被调用时,request_store 里的 status 必须已经是 PREFILLING;
    之后每一次 decode 被调用时,必须已经是 DECODING;结束后是 DONE。

    ▸ 背景:PREFILLING / DECODING 这两个枚举值现在定义了但全仓库零处赋值(自己 grep 一下
      RequestStatus.PREFILLING 就知道)。AC 要求这次要么写入、要么删掉,这条测试选了"写入"。
    ▸ 机关:靠 stub 在**调用那一刻**抓的 status 快照,不是事后查 —— 事后只看得到 DONE。
    ▸ 查的是:step 0 那次快照是 PREFILLING,step 1/2/3 那几次是 DECODING。
    ▸ 挂了的含义:要么这两个状态压根没写,要么写的时机不对(比如调用完才写,或者一路都写 PROCESSING)。
    ▸ 只有一条请求 —— 这条不测调度,只测状态时机。
    """
    pending_queue = asyncio.Queue()
    request_store: Dict[str, RequestData] = {}
    calls: List[CallRecord] = []
    id_to_request = {70: "R1"}
    adapter = make_stub_adapter(id_to_request, request_store, calls)
    _, engine, pending_queue, request_store = make_system(adapter, pending_queue, request_store)

    push_stub_request(pending_queue, request_store, "R1", 70, max_length=1 + 4)

    task = engine.run()
    await asyncio.wait_for(pending_queue.join(), timeout=5)
    await engine.shutdown(draining=True)
    assert task.done()

    assert request_store["R1"].status == RequestStatus.DONE
    assert request_store["R1"].generated_tokens == [71, 72, 73, 74]

    status_by_step = {c.step: c.status for c in calls if c.request_id == "R1"}
    assert status_by_step[0] == RequestStatus.PREFILLING, (
        f"prefill 被调用时 status 应为 PREFILLING,实际观测到 {status_by_step[0]}"
    )
    for step in range(1, 4):
        assert status_by_step[step] == RequestStatus.DECODING, (
            f"第 {step} 次 decode 被调用时 status 应为 DECODING,实际观测到 {status_by_step[step]}"
        )


# ---------------------------------------------------------------------------
# T5 —— 单条失败隔离
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_t5a_prefill_failure_does_not_take_down_the_other_request():
    """一条 prompt 长度 == max_length(prefill 会 raise),另一条正常。前者 FAILED,
    后者 DONE 且 token 正确,join() 能正常返回(不会因为一条失败就卡住)。

    ▸ 坏输入长什么样:max_length=1 而 prompt 就是 1 个 token —— 一个 token 都塞不下了,
      LMEngine.inference() 里那道 `if token_tensor.shape[1] == max_length: raise` 会抛。
      **注意这是现有代码已经处理对的情况,不是新要求** —— 这条测的是你重构之后还保不保得住。
    ▸ 查的是三件事:BAD 落 FAILED(不是崩掉整个引擎)/ OK 照样 DONE 且 token 完整 /
      join() 还能返回(说明失败的那条也调了 task_done(),没把队列的待办计数永远挂在那)。
    ▸ 挂了的含义:① join() 超时 → 失败路径漏了 task_done(),队列永远等不到归零;
      ② OK 也 FAILED 了 → 一条炸了把同一轮里别人也带下水;
      ③ 整个测试 error 而不是 fail → 异常根本没被接住,冒到了引擎循环外面。
    """
    pending_queue = asyncio.Queue()
    request_store: Dict[str, RequestData] = {}
    calls: List[CallRecord] = []
    id_to_request = {80: "BAD", 90: "OK"}
    adapter = make_stub_adapter(id_to_request, request_store, calls)
    _, engine, pending_queue, request_store = make_system(adapter, pending_queue, request_store)

    push_stub_request(pending_queue, request_store, "BAD", 80, max_length=1)  # prompt 长度(1) == max_length(1)
    push_stub_request(pending_queue, request_store, "OK", 90, max_length=1 + 3)

    task = engine.run()
    await asyncio.wait_for(pending_queue.join(), timeout=5)
    await engine.shutdown(draining=True)
    assert task.done()

    assert request_store["BAD"].status == RequestStatus.FAILED
    assert request_store["BAD"].generated_tokens == []
    assert request_store["OK"].status == RequestStatus.DONE
    assert request_store["OK"].generated_tokens == [91, 92, 93]


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_t5b_mid_decode_failure_does_not_take_down_the_other_request():
    """一条在 decode 中途(第 2 步)抛异常,另一条正常。前者 FAILED,后者 DONE 且 token 正确。

    ▸ 和 T5a 的分工:T5a 炸在**入口**(prefill 之前就被拦下,这条请求一步都没跑过);
      这一条炸在**半路**(已经跑了两步、在飞记录里有它、状态是 DECODING 才炸)。
      半路炸难处理得多 —— 得把它从在飞集合里摘掉、标 FAILED、还了 task_done(),
      而且这一切要在"同一轮里别人还等着被推进"的情况下做完。
    ▸ 机关:raise_at={"BAD": 2} 让 stub 在 BAD 的第 2 次 decode 时抛 RuntimeError。
    ▸ 查的是:BAD → FAILED,OK → DONE 且 token 一个不少。
    """
    pending_queue = asyncio.Queue()
    request_store: Dict[str, RequestData] = {}
    calls: List[CallRecord] = []
    id_to_request = {100: "BAD", 110: "OK"}
    adapter = make_stub_adapter(
        id_to_request, request_store, calls, raise_at={"BAD": 2}
    )
    _, engine, pending_queue, request_store = make_system(adapter, pending_queue, request_store)

    push_stub_request(pending_queue, request_store, "BAD", 100, max_length=1 + 5)
    push_stub_request(pending_queue, request_store, "OK", 110, max_length=1 + 3)

    task = engine.run()
    await asyncio.wait_for(pending_queue.join(), timeout=5)
    await engine.shutdown(draining=True)
    assert task.done()

    assert request_store["BAD"].status == RequestStatus.FAILED
    assert request_store["OK"].status == RequestStatus.DONE
    assert request_store["OK"].generated_tokens == [111, 112, 113]


# ---------------------------------------------------------------------------
# T6 —— join() 语义
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_t6_join_implies_terminal_state():
    """抓『task_done 调早了』:join() 返回的那一刻,所有已准入请求都必须已经处于终态
    (DONE 或 FAILED),不能是 join() 已经放行、但某条请求其实还卡在 PREFILLING/DECODING。

    ▸ 这条测的是 Queue 的两本账有没有对齐:
        账本一(get/put):元素还在不在队列里
        账本二(task_done/join):这个元素的活干完没有
      从"取出来"到"跑完"中间隔着很多轮,task_done() 该记在哪一轮,是这次要你定的事。
    ▸ 机关:注意这条**故意在 shutdown 之前就断言** —— 先 join(),立刻查状态,再 shutdown。
      如果放到 shutdown 之后查,shutdown 会顺手把状态收成终态,这个 bug 就被盖住了。
    ▸ 挂了的含义:某条请求 status 还是 PREFILLING/DECODING —— 你在请求刚被取出来的时候
      就调了 task_done(),于是 join() 提前放行,而请求其实还在飞。
    """
    pending_queue = asyncio.Queue()
    request_store: Dict[str, RequestData] = {}
    calls: List[CallRecord] = []
    id_to_request = {120: "R1", 130: "R2"}
    adapter = make_stub_adapter(id_to_request, request_store, calls)
    _, engine, pending_queue, request_store = make_system(adapter, pending_queue, request_store)

    push_stub_request(pending_queue, request_store, "R1", 120, max_length=1 + 4)
    push_stub_request(pending_queue, request_store, "R2", 130, max_length=1 + 6)

    task = engine.run()
    await asyncio.wait_for(pending_queue.join(), timeout=5)

    assert request_store["R1"].status in (RequestStatus.DONE, RequestStatus.FAILED), (
        f"join() 已经返回,但 R1 的 status 是 {request_store['R1'].status}"
    )
    assert request_store["R2"].status in (RequestStatus.DONE, RequestStatus.FAILED), (
        f"join() 已经返回,但 R2 的 status 是 {request_store['R2'].status}"
    )

    await engine.shutdown(draining=True)
    assert task.done()


# ---------------------------------------------------------------------------
# T7 / T8 —— shutdown 推广到 N 条在飞
# ---------------------------------------------------------------------------


def _fire_when_both_have_advanced_twice(
    seen_steps: Dict[str, int],
    both_advanced_twice: asyncio.Event,
    request_store: Dict[str, RequestData],
    status_at_fire: Dict[str, Optional[RequestStatus]],
):
    """给 on_call 用的工厂:两条请求各自都至少被推进过 2 步(prefill + 至少 1 次 decode)
    之后触发一次 Event。用调用记录本身当信号,不依赖任何 wall-clock 轮询——轮询在这份
    stub 上不可靠,因为它快到经常在测试第一次采样之前就已经把两条请求都跑完了(实测见 handout)。

    为什么要等"各走 2 步"再 shutdown:T7/T8 要检查的是"**同时有多条在飞**的时候 shutdown
    怎么办"。如果发得太早(比如刚 run() 就 shutdown),可能一条都还没进在飞集合,那就退化成
    LMS-6 已经测过的情况,什么也没验到。所以用调用记录当发令枪:两条都确认跑过 prefill +
    至少一次 decode 了,才开枪。

    但"两条都走过 2 步"不等于"两条此刻都在飞":一条跑到底的实现里,R2 走到第 2 步时
    R1 早就 DONE 了。所以开枪那一刻顺手给两条请求的 status 拍一张快照(status_at_fire),
    测试据此断言"开枪时两条都还没进终态"——否则这条测试根本没测到"多条同时在飞"
    (2026-09-17 在未改动的引擎上实测:T8 开枪时 R1 已是 DONE,测试假绿)。

        record.step + 1 = 这条请求到目前为止被推进过几次(step 从 0 起数)
        max(...)        = 取历史最大值,防止乱序回退
        都 >= 2         = 两条都至少 prefill + 1 次 decode 了 → 确实同时在飞
    """

    def on_call(records: List[CallRecord]) -> None:
        for record in records:
            seen_steps[record.request_id] = max(seen_steps[record.request_id], record.step + 1)
        if all(count >= 2 for count in seen_steps.values()) and not status_at_fire:
            status_at_fire.update({rid: request_store[rid].status for rid in seen_steps})
            both_advanced_twice.set()

    return on_call


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_t7_nondraining_shutdown_fails_all_in_flight_requests():
    """D-5 留白的裁决之一:shutdown(draining=False) 时,如果同时有 N 条在飞,是不是这
    N 条都要标 FAILED?这里 N=2,断言两条都变成 FAILED,不能只有其中一条(LMS-6 时代的
    shutdown 只假设"最多一条在飞",这条测试专门检查它有没有被真正推广)。

    ▸ draining=False 的语义:别管没跑完的活了,立刻停 —— 在飞的全部作废。
    ▸ 为什么是"推广"而不是新功能:LMS-6 的 shutdown 里 cancel 的是**一个** inferencee_task,
      因为那时候在飞的最多就一条。现在在飞可能有 N 条,那段代码必须跟着变成 N 条都处理。
      **这是最容易漏的一处** —— 它不报错,只是安静地少收拾了一条。
    ▸ 查的是:两条都 FAILED + engine.status == STOPPED + task.done()。
    """
    pending_queue = asyncio.Queue()
    request_store: Dict[str, RequestData] = {}
    calls: List[CallRecord] = []
    id_to_request = {140: "R1", 150: "R2"}
    seen_steps = {"R1": 0, "R2": 0}
    both_advanced_twice = asyncio.Event()
    status_at_fire: Dict[str, Optional[RequestStatus]] = {}
    adapter = make_stub_adapter(
        id_to_request,
        request_store,
        calls,
        on_call=_fire_when_both_have_advanced_twice(
            seen_steps, both_advanced_twice, request_store, status_at_fire
        ),
    )
    _, engine, pending_queue, request_store = make_system(adapter, pending_queue, request_store)

    push_stub_request(pending_queue, request_store, "R1", 140, max_length=1 + 6)
    push_stub_request(pending_queue, request_store, "R2", 150, max_length=1 + 6)

    task = engine.run()
    await asyncio.wait_for(both_advanced_twice.wait(), timeout=5)

    await asyncio.wait_for(engine.shutdown(draining=False), timeout=5)
    assert task.done()
    assert engine.status == InferenceEngineStatus.STOPPED

    # 前置条件:开枪那一刻两条都必须还在飞。放在 shutdown 之后才断言,是为了不管成败都先把引擎停干净。
    terminal = (RequestStatus.DONE, RequestStatus.FAILED)
    assert all(status not in terminal for status in status_at_fire.values()), (
        f"开枪(发起 shutdown)时的状态是 {status_at_fire}——至少一条已经是终态,"
        f"这次 shutdown 面对的不是'多条同时在飞'。实际顺序 {[c.request_id for c in calls]}。"
    )

    assert request_store["R1"].status == RequestStatus.FAILED
    assert request_store["R2"].status == RequestStatus.FAILED


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_t8_draining_shutdown_lets_all_in_flight_requests_finish():
    """D-5 留白的裁决之二:shutdown(draining=True) 时,N 条在飞请求都应该被放行跑完
    (DONE、token 完整),不是像 draining=False 那样直接标 FAILED。

    ▸ 和 T7 是一对:同样两条在飞、同样在"各走 2 步"之后开枪,只差 draining 这一个参数。
      draining=False → 两条都 FAILED(T7);draining=True → 两条都 DONE 且 token 一个不少(T8)。
    ▸ 注意 token 是**全量**断言([161,162,163,164] / [171,172,173]),不是"跑到哪算哪":
      draining 的承诺是"手上的活干完再停",干一半就停不算 draining。
    """
    pending_queue = asyncio.Queue()
    request_store: Dict[str, RequestData] = {}
    calls: List[CallRecord] = []
    id_to_request = {160: "R1", 170: "R2"}
    seen_steps = {"R1": 0, "R2": 0}
    both_advanced_twice = asyncio.Event()
    status_at_fire: Dict[str, Optional[RequestStatus]] = {}
    adapter = make_stub_adapter(
        id_to_request,
        request_store,
        calls,
        on_call=_fire_when_both_have_advanced_twice(
            seen_steps, both_advanced_twice, request_store, status_at_fire
        ),
    )
    _, engine, pending_queue, request_store = make_system(adapter, pending_queue, request_store)

    push_stub_request(pending_queue, request_store, "R1", 160, max_length=1 + 4)
    push_stub_request(pending_queue, request_store, "R2", 170, max_length=1 + 3)

    task = engine.run()
    await asyncio.wait_for(both_advanced_twice.wait(), timeout=5)

    await asyncio.wait_for(engine.shutdown(draining=True), timeout=5)
    assert task.done()
    assert engine.status == InferenceEngineStatus.STOPPED

    # 前置条件:开枪那一刻两条都必须还在飞。放在 shutdown 之后才断言,是为了不管成败都先把引擎停干净。
    terminal = (RequestStatus.DONE, RequestStatus.FAILED)
    assert all(status not in terminal for status in status_at_fire.values()), (
        f"开枪(发起 shutdown)时的状态是 {status_at_fire}——至少一条已经是终态,"
        f"这次 shutdown 面对的不是'多条同时在飞'。实际顺序 {[c.request_id for c in calls]}。"
    )

    assert request_store["R1"].status == RequestStatus.DONE
    assert request_store["R2"].status == RequestStatus.DONE
    assert request_store["R1"].generated_tokens == [161, 162, 163, 164]
    assert request_store["R2"].generated_tokens == [171, 172, 173]


# ---------------------------------------------------------------------------
# T9 —— D-5 留白的裁决:DRAINING 竞态窗口
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_t9_late_submission_never_stays_pending_forever():
    """D-5 留白的裁决:DRAINING 期间提交的请求、以及 shutdown 返回之后提交的请求,都不能
    永远停在 PENDING —— 要么 submit_request 直接抛异常,要么在超时内进入 FAILED(不能是
    DONE)。EXPECTED_LATE_OUTCOME 还是 None 时只强制这条政策无关的不变量本身,并用
    pytest.fail() 提醒你先在 ADR 里做出 'raise' vs 'FAILED' 的具体裁决;设成其中之一之后,
    测试才会按你选的那条具体规则精确比对。

    ▸ 这条和别的 9 条不一样:**它现在一定是红的,而且是故意的。** 文件顶上的
      EXPECTED_LATE_OUTCOME 默认是 None,哪怕你的实现完全正确,最后也会被 pytest.fail()
      拦下来,消息是"先去写 ADR 做裁决,再回来把这个常量改掉"。这是设计,不是 bug。
    ▸ 要你裁决的是什么:D-5 只锁了"DRAINING 期间 / STOPPED 之后不再收新请求"这条**政策**,
      没锁"拒绝这件事怎么告诉调用方"。两个选项:
          "raise"  —— submit_request 当场抛异常。调用方立刻知道,但要处理异常。
          "FAILED" —— submit_request 照常返回 id,这条请求随后被标成 FAILED。接口更统一,
                      但调用方得自己去查状态才知道被拒了。
      两个都满足底线("不能永远停在 PENDING"),选哪个是你的设计决定,写进 ADR。
    ▸ 两个场景:① 提交动作和 shutdown() 并发赛跑 ② shutdown 完全返回、STOPPED 之后再提交。
    ▸ 这也是唯一用到 receiver 的测试 —— 所以这里是 `receiver, engine, ... = make_system(...)`,
      不是别处那个 `_`。
    """
    pending_queue = asyncio.Queue()
    request_store: Dict[str, RequestData] = {}
    calls: List[CallRecord] = []
    id_to_request = {180: "R1"}
    adapter = make_stub_adapter(id_to_request, request_store, calls)
    receiver, engine, pending_queue, request_store = make_system(
        adapter, pending_queue, request_store
    )

    push_stub_request(pending_queue, request_store, "R1", 180, max_length=1 + 4)
    task = engine.run()

    # 先等 R1 真正开始跑,确认后面发起 shutdown 的时候引擎确实处在忙碌状态,不是在等一个空队列。
    # 注意:不在这里额外等待"engine.status == DRAINING"——stub 快到 R1 经常在下一次事件循环
    # 调度之前就已经整条跑完,DRAINING 窗口可能只存在不到 1ms,polling 大概率会直接跳过它。
    # 场景一因此故意设计成"提交动作和 shutdown() 并发赛跑",不假设自己一定落在 DRAINING 那一刻。
    await wait_until(
        lambda: request_store["R1"].status != RequestStatus.PENDING,
        timeout=5,
        description="R1 开始被处理(status 离开 PENDING)",
    )

    async def submit_late_and_check(user_id: str) -> None:
        if EXPECTED_LATE_OUTCOME == "raise":
            with pytest.raises(Exception):
                await receiver.submit_request("late prompt", user_id)
            return

        if EXPECTED_LATE_OUTCOME == "FAILED":
            late_id = await receiver.submit_request("late prompt", user_id)
            await wait_until(
                lambda: request_store[late_id].status != RequestStatus.PENDING,
                timeout=5,
                description=f"迟到请求 {late_id} 离开 PENDING",
            )
            assert request_store[late_id].status == RequestStatus.FAILED, (
                f"EXPECTED_LATE_OUTCOME='FAILED',但迟到请求 {late_id} 最终状态是 "
                f"{request_store[late_id].status},不是 FAILED。"
            )
            return

        # EXPECTED_LATE_OUTCOME 还是 None——还没有裁决,这里只强制 D-5 已经锁定的那条
        # 政策无关的不变量:D-5 锁定的是"停止接收新请求"这件事本身,不是具体传回方式,
        # 所以只要求"要么 submit_request 直接 raise,要么迟到请求在超时内离开 PENDING
        # 并且落在 FAILED"——不能是 DONE(那意味着它其实被接收并处理了,和"停止接收"矛盾),
        # 也不能永远停在 PENDING。
        raised = False
        try:
            late_id = await receiver.submit_request("late prompt", user_id)
        except Exception:
            raised = True

        if not raised:
            await wait_until(
                lambda: request_store[late_id].status != RequestStatus.PENDING,
                timeout=5,
                description=f"迟到请求 {late_id} 离开 PENDING",
            )
            assert request_store[late_id].status == RequestStatus.FAILED, (
                f"D-5 已经锁定 DRAINING 期间 / STOPPED 之后停止接收新请求——但迟到请求 "
                f"{late_id} 最终状态是 {request_store[late_id].status},不是 FAILED "
                f"(尤其不能是 DONE:那意味着它其实被接收并处理了,和'停止接收'矛盾)。"
            )


    # 场景一:提交动作和 shutdown(draining=True) 并发赛跑——不管它最终落在 DRAINING 窗口
    # 内还是 shutdown 已经返回之后,只要不是"提交成功但这条请求从此永远停在 PENDING",就算过。
    # 中间这次 sleep(0) 只是把 shutdown_task 排到的那次调度真正跑一下(create_task 只是排期,
    # 不会立刻执行),不是在赌一个 wall-clock 窗口——shutdown() 的状态切换发生在它自己第一次
    # await 之前那段同步代码里,一次 sleep(0) 足够让它跑到那里。
    shutdown_task = asyncio.create_task(engine.shutdown(draining=True))
    await asyncio.sleep(0)
    await submit_late_and_check("late_user_racing_with_shutdown")

    await asyncio.wait_for(shutdown_task, timeout=5)
    assert task.done()
    assert engine.status == InferenceEngineStatus.STOPPED

    # 场景二:shutdown 已经完全返回、engine 确定已经 STOPPED 之后再提交
    await submit_late_and_check("late_user_after_stopped")

    # 两个场景都满足了政策无关的不变量,才轮到这一条:还没裁决就不算过(AC 要求给出裁决)。
    if EXPECTED_LATE_OUTCOME is None:
        pytest.fail(
            "EXPECTED_LATE_OUTCOME 还是 None:两个场景(和 shutdown 赛跑的提交、STOPPED 之后的提交)"
            "都满足了政策无关的不变量——没有迟到请求永远停在 PENDING,也没有被接收处理成 DONE。"
            "但 D-5 只锁定了'停止接收新请求'这条政策本身,没有锁定'拒绝这件事怎么传回调用方'。"
            "先在 ADR 里做出 'raise' vs 'FAILED' 的裁决,再回来把这个常量改成你的结论对应的值。"
        )


# ---------------------------------------------------------------------------
# T10 —— 贪心回归:真实 HF gpt2,不能因为这次重构改坏输出
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gpt2_model():
    return GPT2LMHeadModel.from_pretrained("gpt2")


@pytest.fixture(scope="module")
def gpt2_tokenizer():
    return GPT2Tokenizer.from_pretrained("gpt2")


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_t10_greedy_matches_huggingface_generate_for_concurrent_requests(
    gpt2_model, gpt2_tokenizer
):
    """3 条不同长度的 prompt 并发经过 InferenceEngine,每条输出必须与
    model.generate(do_sample=False) 逐 token 一致 —— 这次重构不能改坏既有的正确性门禁。

    ▸ 唯一一条用**真 gpt2** 的测试(前 9 条全是 stub)。因此也是最慢的一条:要下载/加载权重,
      timeout 给到了 90 秒。先跑前 9 条,这条最后跑。
    ▸ 测的东西完全不同:前 9 条测"调度顺序对不对",这一条测"算出来的 token 对不对"。
      做法是拿 HuggingFace 自己的 model.generate(do_sample=False) 当标准答案逐 token 比。
    ▸ 为什么必须有它:LMS-7 改的是调度,不该动数值结果。如果只有前 9 条,你可能做出一个
      "顺序完美但把 KV / prefix 串错行"的实现 —— 交替推进多条请求时,最容易错的就是
      把 A 的状态喂给了 B。这条一比对就能抓出来。
    ▸ 挂了的含义:生成的 token 和 HF 不一致 → 多条请求之间的状态串了,或者停止条件算错了。
    """
    prompts = [
        ("Hello, how are you? My name is Yifan and ", 30),
        ("The quick brown fox ", 20),
        ("Once upon a time ", 40),
    ]
    expected_tokens = []
    for prompt, max_length in prompts:
        input_ids = gpt2_tokenizer(prompt, return_tensors="pt").input_ids
        hf_output = gpt2_model.generate(
            input_ids,
            pad_token_id=gpt2_tokenizer.eos_token_id,
            eos_token_id=gpt2_tokenizer.eos_token_id,
            do_sample=False,
            max_length=max_length,
        )
        expected_tokens.append(hf_output[0].tolist()[input_ids.shape[1] :])

    pending_queue = asyncio.Queue()
    request_store: Dict[str, RequestData] = {}
    receiver, engine, pending_queue, request_store = make_system(
        gpt2_model, pending_queue, request_store
    )

    request_ids = [
        await receiver.submit_request(prompt, "user", max_length=max_length)
        for prompt, max_length in prompts
    ]

    task = engine.run()
    await asyncio.wait_for(pending_queue.join(), timeout=75)
    await engine.shutdown(draining=True)
    assert task.done()

    for request_id, expected in zip(request_ids, expected_tokens):
        assert request_store[request_id].status == RequestStatus.DONE
        assert request_store[request_id].generated_tokens == expected


# @dataclass
# class InFlightTokenizedData(TokenizedData):
#     def __init__(self, request_id: str, tokens: List[int]):
#         super().__init__(request_id, tokens)
#         self.step = 0  # 当前 decode 步数,从 0 开始计数

# def inference(requests: Dict[int, InFlightTokenizedData]):
#     while True:
#         for request_id, in_flight_request in requests.items():
#             if in_flight_request.step == 0:
#                 # prefill -> step 1
#                 pass
#             elif in_flight_request.step > 0:
#                 # decode -> step n
#                 pass
#             else:
#                 raise ValueError(f"Invalid step {in_flight_request.step} for request {request_id}")
                