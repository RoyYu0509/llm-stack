"""benchmarks/serving_bench.py 的测试:先证明「测量工具」本身量得对,它量出来的数字才有资格进简历。

假引擎的每一步耗时都是写死的(time.sleep,和真实前向一样阻塞事件循环),所以 TTFT / TPOT / 排队时间
的期望值可以手算,再和 run_open_loop 实测的对。覆盖:
  · token 可见时刻 → TTFT / TPOT / E2E 的口径;
  · coordinated omission:引擎被一次长前向挡住时到达的请求,延迟必须从「调度到达」算,不是从「引擎腾出手收到」算;
  · seq 臂的闸门:引擎里同时最多一条,先到先服务;
  · 超时中止 / 请求失败 的记账;
  · 泊松到达与工作负载的确定性;
  · 真引擎 + 真模型(CPU 上的极小配置)上 ForwardProfiler 记的前向数、行数、形状。
"""

import asyncio
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import serving_bench as sb  # noqa: E402
from serving_metrics import ModelCost, Slo, e2e, summarize, tpot, ttft  # noqa: E402

from inferenceLM.data.request_status import RequestStatus  # noqa: E402


# ---------------------------------------------------------------- 假引擎:耗时写死、语义和真引擎的循环一致


class FakeEngine:
    """iteration-level 调度:每轮先 admit 所有排队的,新请求各 prefill 一次(阻塞 prefill_s),
    其余在飞的请求合成一次 decode(阻塞 decode_s),每轮末尾 yield 一次。"""

    def __init__(self, queue, store, prefill_s, decode_s, fail=()):
        self.queue, self.store, self.prefill_s, self.decode_s, self.fail = queue, store, prefill_s, decode_s, set(fail)
        self.in_flight, self.max_in_flight, self.stopped, self.task = {}, 0, False, None

    def run(self):
        self.task = asyncio.create_task(self._loop())
        return self.task

    def _admit(self, item, new) -> bool:
        if item is None:
            self.queue.task_done()
            return True
        rid = item.request_id
        if rid in self.fail:
            self.store[rid].status = RequestStatus.FAILED
            self.queue.task_done()
        else:
            n_new = self.store[rid].max_token_length - len(item.tokens)
            self.in_flight[rid] = dict(n_new=n_new, generated=[])
            new.append(rid)
        return False

    def _emit(self, rid) -> None:
        rec = self.in_flight[rid]
        rec["generated"].append(0)
        if len(rec["generated"]) == rec["n_new"]:
            self.store[rid].status = RequestStatus.DONE
            self.store[rid].generated_tokens = list(rec["generated"])
            del self.in_flight[rid]
            self.queue.task_done()

    def _abort_everything(self) -> None:
        for rid in list(self.in_flight):
            self.store[rid].status = RequestStatus.FAILED
            self.queue.task_done()
        self.in_flight.clear()
        while not self.queue.empty():
            item = self.queue.get_nowait()
            if item is not None:
                self.store[item.request_id].status = RequestStatus.FAILED
            self.queue.task_done()

    async def _loop(self):
        stop = False
        while True:
            new = []
            if not self.in_flight and not stop:
                stop = self._admit(await self.queue.get(), new)
            while not stop and not self.queue.empty():
                stop = self._admit(self.queue.get_nowait(), new)
            if self.stopped:
                self._abort_everything()
                break
            if not self.in_flight:
                if stop:
                    break
                continue
            self.max_in_flight = max(self.max_in_flight, len(self.in_flight))
            for rid in new:
                time.sleep(self.prefill_s)
                self._emit(rid)
            decoders = [r for r in self.in_flight if r not in new]
            if decoders:
                time.sleep(self.decode_s)
                for rid in decoders:
                    self._emit(rid)
            await asyncio.sleep(0)

    async def shutdown(self, draining: bool = True):
        self.queue.put_nowait(None)
        self.stopped = not draining
        await self.queue.join()
        await self.task


class FakeAdapters:
    def __init__(self, prefill_s, decode_s, fail=()):
        self.args, self.engine = (prefill_s, decode_s, fail), None

    def build_engine(self, _model, queue, store, _eos):
        self.engine = FakeEngine(queue, store, *self.args)
        return self.engine

    @staticmethod
    def generated_so_far(engine, store, rid):
        rec = engine.in_flight.get(rid)
        return list(rec["generated"]) if rec else list(store[rid].generated_tokens)


def reqs(n_new, count=1, prompt_len=4):
    return [sb.Req(f"r{i}", list(range(prompt_len)), n_new) for i in range(count)]


def run(adapters, arm, rs, arrivals, drain_s=30.0):
    return asyncio.run(sb.run_open_loop(None, adapters, arm, rs, arrivals, drain_s))


TOL = 0.03  # 睡眠超时 + 事件循环一轮的抖动


# ---------------------------------------------------------------- 口径


def test_ttft_tpot_e2e_match_hand_computed_timeline():
    """prefill 30 ms,每步 decode 10 ms,5 个 token:token 在 30 / 40 / 50 / 60 / 70 ms 可见。"""
    res = run(FakeAdapters(0.03, 0.01), "cb", reqs(5), [0.0])
    (t,) = res["traces"]
    assert t.ok and len(t.token_times) == 5
    assert ttft(t) == pytest.approx(0.03, abs=TOL)
    assert tpot(t) == pytest.approx(0.01, abs=0.005)
    assert e2e(t) == pytest.approx(0.07, abs=TOL)
    assert all(b >= a for a, b in zip(t.token_times, t.token_times[1:])), "token 可见时刻必须单调"
    assert not res["aborted"]


def test_arrival_is_the_scheduled_time_not_the_time_the_engine_got_to_it():
    """coordinated omission。req0 的 prefill 要阻塞事件循环 0.2 s;req1 在 0.05 s 就该到了,但引擎被挡着,
    客户端 0.2 s 才有机会把它交给引擎。req1 第一个 token 在 ≈0.41 s 可见,它的 TTFT 必须是 0.41−0.05≈0.36,
    不是 0.41−0.2≈0.21(那等于把被挡住的 0.15 s 从延迟里抹掉)。"""
    res = run(FakeAdapters(0.2, 0.01), "cb", reqs(2, count=2), [0.0, 0.05])
    t1 = res["traces"][1]
    assert ttft(t1) >= 0.30, f"TTFT {ttft(t1):.3f}s:arrival 取成了引擎实际收到的时刻,被前向挡住的等待丢了"
    assert ttft(t1) == pytest.approx(0.36, abs=0.06)


# ---------------------------------------------------------------- seq 臂


def test_seq_arm_serves_one_at_a_time_in_arrival_order():
    """3 条同时到,n_new=3,服务时间 = 30 + 2×10 = 50 ms。第 i 条的第一个 token 在 i×50 + 30 ms。"""
    a = FakeAdapters(0.03, 0.01)
    res = run(a, "seq", reqs(3, count=3), [0.0, 0.0, 0.0])
    assert a.engine.max_in_flight == 1, "seq 臂的闸门没起作用:引擎里同时有多条请求"
    ts = res["traces"]
    for i, t in enumerate(ts):
        assert ttft(t) == pytest.approx(i * 0.05 + 0.03, abs=0.04), f"r{i} 的排队时间不对"
    assert [t.token_times[0] for t in ts] == sorted(t.token_times[0] for t in ts), "先到先服务"


def test_cb_arm_admits_everything_that_has_arrived():
    a = FakeAdapters(0.01, 0.01)
    run(a, "cb", reqs(4, count=4), [0.0, 0.0, 0.0, 0.0])
    assert a.engine.max_in_flight > 1


# ---------------------------------------------------------------- 中止与失败的记账


def test_overload_is_aborted_not_hung_and_not_counted_as_throughput():
    """seq 臂 5 条同时到、每条服务 50 ms,但只给 0.12 s:前 2 条能完成,其余必须是 aborted。"""
    res = run(FakeAdapters(0.03, 0.01), "seq", reqs(3, count=5), [0.0] * 5, drain_s=0.12)
    ts = res["traces"]
    done = [t for t in ts if t.status == "done"]
    aborted = [t for t in ts if t.status == "aborted"]
    assert res["aborted"]
    assert 1 <= len(done) <= 3 and len(done) + len(aborted) == 5
    assert all(not t.ok for t in aborted)
    assert res["window_s"] == pytest.approx(0.12, abs=1e-6), "被中止时窗口必须正好到 deadline(最后一次到达 + drain),不是最后一条完成"
    s = summarize(ts, res["window_s"], Slo(10, 10))
    assert s["n_aborted"] == len(aborted)
    assert s["output_token_throughput"] * res["window_s"] == pytest.approx(3 * len(done)), "被中止请求的部分 token 不算吞吐"
    assert s["slo_attainment"] == pytest.approx(len(done) / 5)


def test_failed_requests_are_counted_and_excluded():
    res = run(FakeAdapters(0.01, 0.01, fail={"r1"}), "cb", reqs(3, count=3), [0.0] * 3)
    s = summarize(res["traces"], res["window_s"])
    assert (s["n_done"], s["n_failed"], s["n_aborted"]) == (2, 1, 0)
    assert [t.status for t in res["traces"]] == ["done", "failed", "done"]


# ---------------------------------------------------------------- 到达过程与工作负载


def test_poisson_arrivals():
    a = sb.poisson_arrivals(5.0, 20000, seed=3)
    assert a[0] == 0.0 and a == sorted(a)
    assert a == sb.poisson_arrivals(5.0, 20000, seed=3), "同一个种子必须给出同一串到达(seq 和 cb 才是同一份负载)"
    assert a != sb.poisson_arrivals(5.0, 20000, seed=4)
    mean_gap = a[-1] / (len(a) - 1)
    assert mean_gap == pytest.approx(1 / 5.0, rel=0.03)  # 指数分布均值 1/λ
    gaps = [y - x for x, y in zip(a, a[1:])]
    var = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
    assert var ** 0.5 == pytest.approx(mean_gap, rel=0.05), "泊松过程的间隔标准差 = 均值(比均匀到达突发得多)"


@pytest.mark.parametrize("ctx", [1024, 256, 64])  # 小 context 才会触发截断
@pytest.mark.parametrize("workload", list(sb.WORKLOADS))
def test_workloads_fit_context_and_are_deterministic(workload, ctx):
    rs = sb.make_requests(workload, 300, seed=1, vocab=100, ctx=ctx)
    assert all(1 <= len(r.prompt) and 1 <= r.n_new and len(r.prompt) + r.n_new <= ctx for r in rs)
    again = sb.make_requests(workload, 300, seed=1, vocab=100, ctx=ctx)
    assert [(r.prompt, r.n_new) for r in rs] == [(r.prompt, r.n_new) for r in again]


def test_chat_workload_is_long_tailed_and_fixed_is_not():
    chat = sb.make_requests("chat", 2000, 0, 100, 1024)
    outs = sorted(r.n_new for r in chat)
    assert outs[int(0.99 * len(outs))] > 3 * outs[len(outs) // 2], "chat 输出长度应该是长尾(p99 > 3×p50)"
    assert len({r.n_new for r in sb.make_requests("fixed", 50, 0, 100, 1024)}) == 1


# ---------------------------------------------------------------- 真引擎 + 真模型


@pytest.fixture(scope="module")
def real():
    torch = pytest.importorskip("torch")
    cfg = dict(vocab_size=64, context_length=64, num_layers=2, d_model=32, num_heads=4, d_ff=48, rope_theta=10000.0)
    model = sb.record_demo.build_model(cfg, "cpu", 0, None, sb.record_demo.get_attention("naive"))
    adapters = sb.record_demo.load_adapters(Path(__file__).parent / "m2_adapters.py")
    return torch, cfg, model, adapters


def test_profiler_on_the_real_engine(real):
    torch, cfg, model, adapters = real
    lengths = [(5, 6), (9, 8), (3, 7), (12, 5)]  # (prompt, n_new)
    rs = [sb.Req(f"r{i}", [(7 * i + j) % 64 for j in range(p)], n) for i, (p, n) in enumerate(lengths)]
    from cs336_basics.transfromer import multiheads_attention as mha

    before = mha.scaled_dot_product_attention
    out = {}
    for arm in ("seq", "cb"):
        prof = sb.ForwardProfiler(model, sync=False)
        try:
            res = asyncio.run(sb.run_open_loop(model, adapters, arm, rs, [0.0] * 4, 60.0))
        finally:
            prof.close()
        assert mha.scaled_dot_product_attention is before, "profiler 必须把被替换的注意力函数还原"
        assert [t.status for t in res["traces"]] == ["done"] * 4
        assert [len(t.token_times) for t in res["traces"]] == [n for _, n in lengths], "每条请求恰好生成 n_new 个 token"
        out[arm] = prof.records

    total_new = sum(n for _, n in lengths)
    L = cfg["num_layers"]
    seq, cb = out["seq"], out["cb"]
    assert len(seq) == total_new, "一次一条:每个 token 一次前向(prefill 的那次产出第 1 个)"
    assert sorted(r.new_tokens for r in seq if r.new_tokens > 1) == sorted(p for p, _ in lengths), "prefill 形状 = prompt 长度"
    assert all(r.rows == 1 for r in seq)
    assert all(len(r.kv_widths) == L for r in seq + cb), "每次前向每层各一次注意力"
    assert sum(r.new_tokens > 1 for r in cb) == 4, "cb 的 prefill 仍然是每条请求一次"
    assert len(cb) < len(seq), "cb 必须合批:前向次数少于一次一条"
    assert max(r.rows for r in cb if r.new_tokens == 1) > 1, "cb 的 decode 前向应该有多行"
    assert all(r.seconds > 0 for r in seq + cb)

    cost = ModelCost.from_model(model, cfg["num_heads"])
    h = sb.hw_summary(cb, cost, None, None, window_s=1.0)
    assert h["decode"]["rows_mean"] > 1 and h["prefill"]["forwards"] == 4
