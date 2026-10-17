"""M2(LMS-10)变长批量 decode 的 adapter —— batched_decode_test.py 和你的实现之间唯一的接缝。

照 CS336 作业 `tests/adapters.py` 的做法:测试文件只调用这里的函数(外加引擎本来就公开的
run() / shutdown() / pending_queue / request_store),**不 import 你的任何内部类、不假设你的
缓存长什么样**。你把每个函数接到自己的实现上,测试就能跑;你的内部设计怎么改,测试都不用动。

每个函数的 docstring 只写**契约**(输入、输出、调用之后必须成立什么),不写怎么实现。
「每行位置从哪来 / padding 怎么不被看到 / 变长 KV 怎么拼 / 批放在哪、谁持有」全是你的设计决定。
契约的完整说明和每条测试的意图见 docs/learning/m2-m3-cb-spec.html(工作区 llm/ 下)。

测试用的模型:`cs336_basics.lm.TransformerLM` 的一个小配置(float32、eval 模式、随机初始化),
由测试文件构造好再传进来。你不需要、也不应该在这里构造模型。

「一次前向」在本文件里的精确含义:模型的输出投影 `model.head` 被调用一次。测试就是在
`model.head` 上挂 forward hook 来数前向次数、看每次处理了几行 —— 所以一次 decode step 里,
所有行必须经过同一次 `model.head` 调用。
"""

from typing import Any, Dict, Hashable, List

import asyncio
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------------------
# 模型层:一个「正在 decode 的批」的四个动作。
# 测试用它们直接验证「变长的几行合成一次前向」本身,不经过引擎和 asyncio。
# `batch` 是你自己定义的对象(测试只把它原样传回给你,从不读它的内容)。
# `key` 是测试给每一行起的名字(str),同一个批里不重复;行的身份只认 key,不认 prompt 内容
# (两行 prompt 完全相同也必须是两条独立的行)。
# ---------------------------------------------------------------------------------------


def new_decode_batch(model: nn.Module) -> Any:
    """返回一个空批。

    契约:
      - 返回值里一行都没有;之后可以用 add_to_batch 往里加行。
      - 同一个 model 可以同时存在多个互不相干的空批(测试会各建各的)。
    """
    from inferenceLM.engine.ragged_batch import RaggedDecodeBatch
    return RaggedDecodeBatch(model)


def add_to_batch(model: nn.Module, batch: Any, key: Hashable, prompt_ids: List[int]) -> torch.Tensor:
    """对一条新请求做 prefill, 并让它成为 `batch` 里名叫 `key` 的一行。

    输入:
      prompt_ids —— 这条请求的 prompt,长度 >= 1,长度和批里已有各行的当前长度可以任意不同
                   (更短、相等、更长都会测)。批里已有的行可能已经 decode 过若干步。
    返回:
      形状 (vocab_size,) 的 logits —— prompt 最后一个位置的输出,也就是「这条请求第一个生成 token」
      的分布。测试会拿它和 `model(整条 prompt)[0, -1]` 比。
    调用之后必须成立:
      - 这一行从下一次 decode_batch_step 起参与批量 decode;
      - 批里**已有**各行之后的结果不因为这一行的加入而改变。
    """
    with torch.no_grad():
        logits, past = model.forward_with_cache(torch.tensor([prompt_ids], dtype=torch.long))

    batch.add(key, past)
    return logits[0, -1, :]


def decode_batch_step(model: nn.Module, batch: Any, tokens: Dict[Hashable, int]) -> Dict[Hashable, torch.Tensor]:
    """把批里每一行各往前推进一个 token —— **只做一次前向**。

    输入:
      tokens —— {key: 这一行这一步要喂进去的 token id}。key 的集合恰好等于批里当前所有行;
                dict 的迭代顺序**不保证**和你加行的顺序一样(测试会故意打乱)。
    返回:
      {key: 形状 (vocab_size,) 的 logits},是「喂进去的这个 token 所在位置」的输出。
      对每一行,它必须和 `model(这一行到目前为止的完整序列)[0, -1]` 一致(容差见 spec)。
    调用之后必须成立:
      - 这一步 `model.head` 恰好被调用一次(测试在 head 上挂了计数 hook);
      - 每一行的缓存长了一个 token,各行互不影响。
    """
    with torch.no_grad():
        return batch.step(tokens)


def remove_from_batch(model: nn.Module, batch: Any, key: Hashable) -> None:
    """把名叫 `key` 的行移出批(这条请求结束了)。

    调用之后必须成立:
      - 下一次 decode_batch_step 的 tokens 里不再有这个 key;
      - 其余各行之后的结果不受影响(不管移走的是第一行、中间一行、还是最长的那行);
      - 行被移空之后,这个批还能继续 add_to_batch 新行并正常 decode。
    """
    batch.remove(key)
   


# ---------------------------------------------------------------------------------------
# 引擎层:测试把请求直接塞进 pending_queue / request_store(和 iteration_lifecycle_test.py 的
# push_stub_request 同一个手法),然后只通过 run() / shutdown() / pending_queue.join() 驱动引擎。
# ---------------------------------------------------------------------------------------


def build_engine(
    model: nn.Module,
    pending_queue: asyncio.Queue,
    request_store: Dict[str, Any],
    eos_token_id: int,
) -> Any:
    """返回一个还没 run() 的引擎:用 KV cache、并且把所有 DECODING 请求的下一步合成一次前向。

    输入:
      model —— 测试构造好的小 TransformerLM(eval 模式)。
      eos_token_id —— 停止 token。测试一律传 -1(永远生成不到),所以每条请求都恰好生成
                      `max_token_length - prompt 长度` 个 token,和 benchmark 的「忽略 EOS、
                      固定输出长度」是同一个设定。
    返回:
      一个对象,至少有 LMS-6/LMS-7 已经定下的公开接口:
      run() -> asyncio.Task、async shutdown(draining: bool)、status;
      读 `pending_queue` 里的 TokenizedData,把结果和状态写回 `request_store[request_id]`
      (DONE 时 generated_tokens 是全部生成的 token)。
    """

    from inferenceLM.engine.inference_engine import InferenceEngine
    from inferenceLM.engine.transformer_lm_adapter import TransformerLMAdapter

    adapter = TransformerLMAdapter(
        model, context_length=model.context_length, eos_token_id=eos_token_id,
        vocab_size=model.vocab_size, use_kv_cache=True,
    )

    return InferenceEngine(adapter, pending_queue, request_store)


def generated_so_far(engine: Any, request_store: Dict[str, Any], request_id: str) -> List[int]:
    """返回这条请求**到此刻为止**已经生成的 token(还在飞的也要能查)。

    给 M3 benchmark 量 TTFT / ITL 用:benchmark 在引擎每次让出控制权时调它一次,记下每个 token
    第一次出现的时刻。所以它必须:
      - 请求还没开始时返回 [];
      - 随时间只增不减,每次看到的都是上一次的前缀;
      - 请求 DONE 之后,等于 request_store[request_id].generated_tokens。
    这些 token 放在哪(在飞记录里、request_store 里、别处)是你的设计,这里只要求能读到。
    """
    rec = engine._in_flight.get(request_id)

    if rec is not None:
        return list(rec.generated)

    return list(request_store[request_id].generated_tokens)
