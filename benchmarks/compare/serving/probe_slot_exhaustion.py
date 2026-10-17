"""Probe (CPU, tiny random model, ~2 s): what the engine does when more requests are in flight than it has KV slots,
and whether every slot / piece of state comes back after heavy churn. Uses the engine read-only.

    cd serving && ../.venv/bin/python ../benchmarks/compare/serving/probe_slot_exhaustion.py

Result on 2026-10-16: with 12 requests at once and 4 slots the engine's main task dies with RuntimeError('no free KV
slot') and every request stays DECODING forever (PreallocDecodeBatch.add is called without a guard in step 4 of
InferenceEngine._keep_get_request_and_inference, and nothing limits admission). 300 requests through 4 slots with a
caller that never exceeds 4: adds == removes == 300, all slots free, nothing left in flight.

"""
import asyncio, sys, time, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "serving" / "benchmarks"))
import torch
import record_demo as rd
from inferenceLM.data.request import RequestData
from inferenceLM.data.request_status import RequestStatus
from inferenceLM.data.tokenized_data import TokenizedData

VOCAB = 64
cfg = dict(vocab_size=VOCAB, context_length=64, num_layers=2, d_model=32, num_heads=4, d_ff=64, rope_theta=10000.0)
model = rd.build_model(cfg, "cpu", 0, None, rd.get_attention("naive"))
adapters = rd.load_adapters(rd.SERVING / "tests" / "m2_adapters.py")
g = torch.Generator().manual_seed(1)
def prompt(): return torch.randint(0, VOCAB, (12,), generator=g).tolist()

def make_engine(slots, pbt=None):
    q, store = asyncio.Queue(), {}
    eng = adapters.build_engine(model, q, store, rd.NO_EOS)
    a = eng.lm_engine.model
    a.decode_backend, a.decode_slots, a.decode_max_len = "prealloc", slots, 40
    eng.prefill_batch_tokens = pbt
    return eng, q, store

def submit(q, store, rid, n_new=6):
    p = prompt()
    store[rid] = RequestData(request_id=rid, user_id="u", timestamp=time.time(), prompt_text="", max_token_length=len(p) + n_new)
    q.put_nowait(TokenizedData(request_id=rid, tokens=p))

async def overload(slots, n, pbt):
    eng, q, store = make_engine(slots, pbt)
    task = eng.run()
    for i in range(n):
        submit(q, store, f"r{i:03d}")
    t0 = time.time()
    while time.time() - t0 < 8 and not task.done() and not all(store[r].status in (RequestStatus.DONE, RequestStatus.FAILED) for r in store):
        await asyncio.sleep(0.01)
    st = collections.Counter(s.status.name for s in store.values())
    print(f"[overload slots={slots} n={n} prefill_batch_tokens={pbt}] engine task done={task.done()}", end="")
    if task.done():
        print(f", exception={task.exception()!r}", end="")
    print(f"\n   request statuses: {dict(st)}   queue unfinished={q._unfinished_tasks}   in_flight={len(eng._in_flight)}")
    if not task.done():
        task.cancel()

async def churn(slots, n):
    eng, q, store = make_engine(slots, None)
    task = eng.run()
    sent = done = 0
    live = set()
    while done < n:
        while sent < n and len(live) < slots:       # a harness that never exceeds the slot count
            rid = f"c{sent:04d}"; submit(q, store, rid); live.add(rid); sent += 1
        await asyncio.sleep(0)
        for rid in list(live):
            if store[rid].status in (RequestStatus.DONE, RequestStatus.FAILED):
                assert store[rid].status == RequestStatus.DONE, rid
                live.discard(rid); done += 1
    await asyncio.wait_for(q.join(), 10)
    await asyncio.wait_for(eng.shutdown(draining=True), 10)
    b = eng._batch
    print(f"[churn slots={slots} n={n}] adds={b.stats['adds']} removes={b.stats['removes']} free_slots={len(b._free)}/{slots} "
          f"rows_still_in_batch={len(b)} in_flight={len(eng._in_flight)} prefill_fifo={len(eng._prefill_fifo)} "
          f"all_DONE={all(s.status == RequestStatus.DONE for s in store.values())} request_store_entries={len(store)}")

async def main():
    await overload(slots=4, n=12, pbt=None)
    await overload(slots=4, n=12, pbt=256)
    await overload(slots=4, n=4, pbt=None)
    await churn(slots=4, n=300)
asyncio.run(main())
