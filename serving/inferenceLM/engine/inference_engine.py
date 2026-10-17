"""Iteration-level scheduling (LMS-7) + batched decode (M2).

One round of `_keep_get_request_and_inference`:

  1. admission   : block on the queue only when nothing is in flight; otherwise drain
                   whatever is already queued (policy: admit everything that is waiting).
                   Admitted requests wait in a FIFO until they are prefilled.

  2. prefill     : take at most `max_prefills_per_round` requests from the front of that FIFO
                   (None = all of them, the original behaviour), one forward each.

  3. decode      : every request that already held a KV cache at the start of the round
                   advances one token -- ONE batched forward when the model supports it,
                   otherwise one forward per request in admission order.

  4. join batch  : requests prefilled in step 2 join the decode batch (they decode next round).

  5. yield once  : `await asyncio.sleep(0)` so producers / shutdown() can run.

With `prefill_batch_tokens` set, step 2 is ONE batched prefill forward (right-padded, at most that many padded
tokens), always followed by step 3's one decode step; what did not fit waits for the next round.

`max_prefills_per_round` (N) trades decode stall against batch size: a large N puts many
prefills in front of one decode (big batch, long ITL spike, late first token for the requests
at the back); a small N interleaves them (short stalls, but the decode batch fills up slowly).
"""
import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import torch
import transformers
from inferenceLM.data.request import RequestData
from inferenceLM.data.request_status import RequestStatus
from inferenceLM.data.tokenized_data import TokenizedData
from inferenceLM.engine.inference_engine_status import InferenceEngineStatus
from inferenceLM.engine.lm_engine import LMEngine


@dataclass
class InFlight:
    request_id: str
    prompt: List[int]
    max_length: int
    latest_token: Optional[int] = None
    past: Any = None  # per-request cache (unbatched path, or until the row joins the batch)
    generated: List[int] = field(default_factory=list)


class InferenceEngine:
    pending_queue: asyncio.Queue  # TokenizedData | None (None = shutdown sentinel)
    request_store: Dict[str, RequestData]
    status: InferenceEngineStatus

    def __init__(
        self,
        model: transformers.PreTrainedModel,
        pending_queue: asyncio.Queue,
        request_store: Dict[str, RequestData],
        max_prefills_per_round: Optional[int] = None,
    ):
        if max_prefills_per_round is not None and max_prefills_per_round < 1:
            raise ValueError(f"max_prefills_per_round must be None or >= 1, got {max_prefills_per_round}")
        self.max_prefills_per_round = max_prefills_per_round  # N; None = prefill everything admitted
        # None = one prefill forward per request (the original behaviour). A number packs the requests prefilled in
        # a round, in arrival order, into right-padded groups of at most that many PADDED tokens per forward.
        self.prefill_batch_tokens: Optional[int] = None
        self.prefill_group_sizes: List[int] = []   # requests per prefill forward, in order (for benchmark metadata)
        self._prefill_fifo: deque = deque()  # admitted, not yet prefilled (request ids, FIFO)
        self.lm_engine = LMEngine(model)
        self.pending_queue = pending_queue
        self.request_store = request_store
        self.status = InferenceEngineStatus.INIT
        self.shutdown_event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._in_flight: Dict[str, InFlight] = {}
        self._batch = None


    # ----- helpers -----
    def _finish(self, rid: str, status: RequestStatus, why: str = "") -> None:
        """Remove the request from in_flight and the inferencing _batch. Mark rid as finished (DONE or FAILED). """
        rec = self._in_flight.pop(rid, None)

        # 还在等 prefill 的请求被终止时 (shutdown / 失败), 也要从等待队列里去掉, 否则下一轮会去 prefill 一个不存在的 rid
        if rid in self._prefill_fifo:
            self._prefill_fifo.remove(rid)

        # 从现在的 inferencing batch 里移除 rid, 以便下次 decode 时不再给它生成 token
        if self._batch is not None and rid in self._batch:
            self._batch.remove(rid)
        # 重新标记 request_store 里的 status, 并把生成的 tokens 写回 request_store
        store = self.request_store[rid]
        store.status = status

        # 如果是 DONE, 就把生成的 tokens 写回 request_store
        if rec is not None and status == RequestStatus.DONE:
            store.generated_tokens = list(rec.generated)
        if why:
            print(f"Request ID {rid} failed: {why}")

        self.pending_queue.task_done()


    def _admit(self, item: Optional[TokenizedData], new_ids: List[str]) -> bool:
        """Admit a new request to in_flight. Returns True if the item is the shutdown sentinel."""
        # Returns True if `item` is the shutdown sentinel.
        if item is None:
            self.pending_queue.task_done()
            return True

        # 如果这个 item 不是一个 shutdown sentinel, 说明他是新的请求, 就把它加入 in_flight, 并在 new_ids 里记录它的 rid
        rid = item.request_id
        max_length = self.request_store[rid].max_token_length
        capacity = self.lm_engine.model.config.n_positions

        # 进行sanity check: max_length 不能大于模型的 context window, prompt 长度不能大于 max_length
        if max_length > capacity:
            self._finish(rid, RequestStatus.FAILED, f"max_length {max_length} > context {capacity}")
        elif len(item.tokens) >= max_length:
            self._finish(rid, RequestStatus.FAILED, f"prompt length {len(item.tokens)} >= max_length {max_length}")
        else:
            self._in_flight[rid] = InFlight(rid, list(item.tokens), max_length)
            new_ids.append(rid)

        return False


    def _append(self, rid: str, token: int) -> bool:
        """把生成的 token 添加到请求中。如果rid请求现在已经完成, 就返回 True, 否则返回 False."""
        rec = self._in_flight[rid]
        rec.generated.append(token)
        rec.latest_token = token

        if self.lm_engine.stopping_criteria(len(rec.prompt), rec.generated, rec.max_length):
            self._finish(rid, RequestStatus.DONE)
            return True

        return False


    def _fail_all_in_flight(self, why: str) -> None:
        """把所有现在在飞的请求通过_finish()方法, 全部标记为 FAILED. (在 shutdown 的时候会用到)"""
        for rid in list(self._in_flight):
            self._finish(rid, RequestStatus.FAILED, why)


    # ---- prefill / decode ----
    def _prefill(self, rid: str) -> None:
        """Run prefill for rid, and update in_flight and request_store accordingly."""
        # 先加入到 in_flight, 并标记 request_store 里的 status 为 PREFILLING
        rec = self._in_flight[rid]
        self.request_store[rid].status = RequestStatus.PREFILLING

        # 进行 prefill, 并把生成的 tokens 写回 request_store
        try:
            ids = torch.tensor(rec.prompt).unsqueeze(0) # (1, S) 的张量
            tok, past = self.lm_engine.prefill(ids, max_length=rec.max_length)
        # 可能会在 prefill 过程中抛出异常, 比如模型的 forward 出错, 或者输入的 prompt 太长等
        except Exception as e:
            self._finish(rid, RequestStatus.FAILED, f"prefill: {e}")
            return

        # 记录 prefill 生成的 tokens, 并更新 in_flight 的 past
        rec.past = past
        if not self._append(rid, int(tok.item())):
            self.request_store[rid].status = RequestStatus.DECODING


    def _prefill_groups(self, rids: List[str]) -> List[List[str]]:
        """Split this round's prefills into forwards: one request each (original), or arrival-order groups whose
        padded size (len(group) x longest prompt) stays within `prefill_batch_tokens`. A prompt longer than the
        budget still gets a forward of its own."""
        budget = self.prefill_batch_tokens
        if budget is None:
            return [[rid] for rid in rids]
        groups: List[List[str]] = []
        cur: List[str] = []
        cur_max = 0
        for rid in rids:
            n = len(self._in_flight[rid].prompt)
            if cur and (len(cur) + 1) * max(cur_max, n) > budget:
                groups.append(cur)
                cur, cur_max = [], 0
            cur.append(rid)
            cur_max = max(cur_max, n)
        if cur:
            groups.append(cur)
        return groups


    def _prefill_group(self, group: List[str]) -> None:
        """One forward for several requests; each ends up exactly as after `_prefill`."""
        for rid in group:
            self.request_store[rid].status = RequestStatus.PREFILLING
        try:
            tokens, pasts = self.lm_engine.prefill_batch([self._in_flight[rid].prompt for rid in group])
        except Exception:
            # one bad request (or one group too big for memory) must not fail its neighbours: redo them one by one
            for rid in group:
                self._prefill(rid)
            return
        for rid, tok, past in zip(group, tokens, pasts):
            self._in_flight[rid].past = past
            if not self._append(rid, int(tok)):
                self.request_store[rid].status = RequestStatus.DECODING


    def _decode(self, decoders: List[str]) -> None:
        """Run decode for all decoders. If the model supports batched decode, do it in one forward."""
        # 如果 self._batch 不为 None, 说明系统中有 _batch 这个 object, 可以去做 batched decode. 
        if self._batch is not None:
            # 先把每个 rid 的最新 token 取出来, 组成一个字典, 然后传给 lm_engine.decode_batch() 做 batched decode
            tokens = {rid: self._in_flight[rid].latest_token for rid in decoders}
            try:
                nxt = self.lm_engine.decode_batch(self._batch, tokens)
            except Exception as e:
                # 如果 batch decode 出错, 就需要把整个 batch 中的所有请求都标记为 FAILED, 并记录错误信息
                self._fail_all_in_flight(f"batched decode: {e}")
                return
            # 如果 batch decode 成功, 就通过 `_append()`, 把每个 rid 的生成 token 添加到它的 in_flight 中
            for rid in decoders:
                self._append(rid, nxt[rid])
            return
        
        # 如果 self._batch 是 None, 说明模型不支持 batched decode, 就只能一条一条地去做 decode
        else:
            # 逐个 rid 去做 decode, 并通过 `_append()` 把生成的 token 添加到它的 in_flight 中
            for rid in decoders:
                rec = self._in_flight[rid]
                try:
                    tok, rec.past = self.lm_engine.decode(torch.tensor([[rec.latest_token]]), rec.past)
                except Exception as e:
                    self._finish(rid, RequestStatus.FAILED, f"decode: {e}")
                    continue

                self._append(rid, int(tok.item()))


    # 主 inference loop
    async def _keep_get_request_and_inference(self):
        """Keep getting requests from the pending_queue and run inference on them."""
        # 初始状态, 标记为: 还没有看到 shutdown signal
        stop_seen = False
        while True:
            # ------- 1. 新加入进来的 requests, 以及已经在飞的 requests -------
            new_ids: List[str] = []

            # 1.1 检查非 draining 的 shutdown
            if self.status == InferenceEngineStatus.STOPPED:
                self._fail_all_in_flight("non-draining shutdown")
                break

            # 1.2 如果 inflight 里面什么都没有, 就阻塞等待 pending_queue.get() 直到有新的请求进来
            if not self._in_flight and not stop_seen:
                stop_seen = self._admit(await self.pending_queue.get(), new_ids) 

            # 1.3 如果 inflight 里面已经有了请求, 把 pending_queue 里面的请求都取出来, 直到 queue 空了为止
            while not stop_seen:
                try:
                    item = self.pending_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                # 检查拿出来的是不是 stop signal
                stop_seen = self._admit(item, new_ids)

            # 1.4 检查非 draining 的 shutdown
            if self.status == InferenceEngineStatus.STOPPED:
                self._fail_all_in_flight("non-draining shutdown")
                break

            # 1.5 如果 inflight 里面什么都没有, 并且已经看到了 stop signal, 就退出 loop
            if not self._in_flight:
                if stop_seen:
                    break
                continue

            # ------- 2. 从等待队列队头最多取 N 个 request 做 prefill (N=None: 全部) -------
            self._prefill_fifo.extend(new_ids)
            budget = self.max_prefills_per_round
            n_prefill = len(self._prefill_fifo) if budget is None else min(budget, len(self._prefill_fifo))
            prefilled = [self._prefill_fifo.popleft() for _ in range(n_prefill)]
            groups = self._prefill_groups(prefilled)
            if self.prefill_batch_tokens is not None and len(groups) > 1:
                # strictly alternate ONE prefill forward and ONE decode step: whatever did not fit in the first group
                # goes back to the FRONT of the FIFO (arrival order kept) and is prefilled next round, so the requests
                # that are already decoding are never stalled behind a long queue of prefills
                for rid in reversed([r for g in groups[1:] for r in g]):
                    self._prefill_fifo.appendleft(rid)
                groups = groups[:1]
                prefilled = list(groups[0])
            for group in groups:
                self.prefill_group_sizes.append(len(group))
                if len(group) == 1:
                    self._prefill(group[0])
                else:
                    self._prefill_group(group)

            # ------- 3. 再对"本轮开始时就已经有 KV cache"的 requests 进行 decode -------
            skip = set(prefilled) | set(self._prefill_fifo)  # 本轮刚 prefill 的 + 还在等 prefill 的, 都还没有可 decode 的状态
            decoders = [rid for rid in self._in_flight if rid not in skip]
            if decoders:
                self._decode(decoders)

            # ------- 4. 把本轮 prefill 完的 requests 的 past 加入到 batch 里, 以便下次 decode 时可以 batched decode -------
            if self._batch is not None:
                for rid in prefilled:
                    # 如果是在 inflight 中, rec 不是 None (不是 stop signal), 而且 rec.past 不是 None (已经 Prefill 完了); 那么就放进 batch 里
                    rec = self._in_flight.get(rid)
                    if rec is not None and rec.past is not None:
                        self._batch.add(rid, rec.past) # 把 new_request 和 他的 KV cache 加入 batch
                        rec.past = None # 把 rec.past 中存的 KV cache 清空, 因为已经加入 batch 里了, 下次 decode 用 batch 里的 KV cache 

            # ------ 5. yield once, so producers / shutdown() can run -------
            await asyncio.sleep(0)

        self.clean_queue()


    # --- public methods ---
    # --- run / shutdown / is_accepting / clean_queue ---
    def run(self) -> asyncio.Task:
        """启动 engine 的主循环, 并返回一个 asyncio.Task 对象, 这个 task 会在后台运行 _keep_get_request_and_inference() 协程."""
        if self.status != InferenceEngineStatus.INIT:
            raise RuntimeError(f"Inference Engine must be in INIT status to run, Not {self.status.name}")

        self.status = InferenceEngineStatus.RUNNING
        
        # 如果模型支持 batched decode, 就创建一个 RaggedDecodeBatch 对象, 并赋值给 self._batch
        if self.lm_engine.supports_batched_decode():
            self._batch = self.lm_engine.new_decode_batch()
        self._task = asyncio.create_task(self._keep_get_request_and_inference())

        return self._task


    def is_accepting(self) -> bool:
        """检查 engine 是否在接受新的请求."""
        return self.status in (InferenceEngineStatus.INIT, InferenceEngineStatus.RUNNING)


    async def shutdown(self, draining: bool = True):
        """Shut down Inference Engine"""
        # 如果 engine 不是 RUNNING 状态, 就抛出异常
        if self.status not in [InferenceEngineStatus.RUNNING]:
            raise RuntimeError(f"Inference Engine must be in RUNNING status to shutdown, Not {self.status.name}")

        # 触发 shutdown_event, 并在 pending_queue 里放一个 None, 作为 shutdown sentinel
        self.shutdown_event.set()
        self.pending_queue.put_nowait(None)

        # 判断 shutdown 是 draining 还是 non-draining
        if draining:
            self.status = InferenceEngineStatus.DRAINING
            await self.pending_queue.join() # 等待现在 queue 中的所有任务完成, 也就是等所有在飞的请求都处理完
            self.status = InferenceEngineStatus.STOPPED
        else:
            self.status = InferenceEngineStatus.STOPPED
            await self.pending_queue.join()

        # 如果 engine 的主循环 task 还在运行, 就等待它完成
        if self._task is not None and self._task is not asyncio.current_task():
            await self._task


    def clean_queue(self):
        """把 queue 中剩下的 requests 标记为 FAILED, 并清理掉."""
        while not self.pending_queue.empty():
            try:
                tokenized_data = self.pending_queue.get_nowait()
                if tokenized_data is not None:
                    self.request_store[tokenized_data.request_id].status = RequestStatus.FAILED
                    print(f"Request ID {tokenized_data.request_id} was marked as failed due to shutdown.")

                self.pending_queue.task_done()

            except asyncio.QueueEmpty:
                break


    