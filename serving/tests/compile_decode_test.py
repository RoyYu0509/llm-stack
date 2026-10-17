"""The decode forward as ONE torch.compile'd graph (PreallocDecodeBatch(compile_decode=True)).

On CPU the backend is `aot_eager`: Dynamo traces the whole forward with `fullgraph=True` (a graph break is an
error, not a silent fallback to eager) and AOTAutograd functionalizes the in-place cache writes, but no kernels are
generated -- so what is checked here is that the compiled graph (a) exists without breaks, (b) computes the same
numbers, (c) still writes the KV cache IN PLACE (the property the whole design rests on), (d) does not recompile
for every batch size. Whether Inductor then fuses the kernels, and whether the cache writes stay in place in the
generated code, is a GPU question (benchmarks/compare/remote/run_stage_c.sh checks both).
"""
import asyncio
import time
from typing import Dict, List

import pytest
import torch

TransformerLM = pytest.importorskip("cs336_basics.lm", reason="needs the training package").TransformerLM

from inferenceLM.data.request import RequestData  # noqa: E402
from inferenceLM.data.request_status import RequestStatus  # noqa: E402
from inferenceLM.data.tokenized_data import TokenizedData  # noqa: E402
from inferenceLM.engine.inference_engine import InferenceEngine  # noqa: E402
from inferenceLM.engine.prealloc_batch import PreallocDecodeBatch  # noqa: E402
from inferenceLM.engine.transformer_lm_adapter import TransformerLMAdapter  # noqa: E402

VOCAB, CTX = 64, 64
ATOL = 1e-5


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = TransformerLM(vocab_size=VOCAB, context_length=CTX, num_layers=2, d_model=32, heads_num=4, d_ff=64,
                      theta=10000.0, device="cpu", dtype=torch.float32)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@pytest.fixture(autouse=True)
def _fresh_dynamo():
    import torch._dynamo as dyn

    dyn.reset()
    yield
    dyn.reset()


def _prompt(n: int, seed: int) -> List[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (n,), generator=g).tolist()


def _batch(model, compiled: bool, slots=5):
    return PreallocDecodeBatch(model, n_slots=slots, max_len=48, dtype=torch.float32, device="cpu",
                               compile_decode=compiled, compile_backend="aot_eager")


def _fill(model, batch, lengths):
    seqs = {}
    for i, n in enumerate(lengths):
        p = _prompt(n, seed=50 + i)
        logits, past = model.forward_with_cache(torch.tensor([p]))
        seqs[f"r{i}"] = p + [int(logits[0, -1].argmax())]
        batch.add(f"r{i}", past)
    return seqs


def test_compiled_decode_matches_eager_step_for_step_and_keeps_the_cache_in_place(model):
    eager, comp = _batch(model, False), _batch(model, True)
    ptrs = [(k.data_ptr(), v.data_ptr()) for k, v in zip(comp.k, comp.v)]
    seqs_e, seqs_c = _fill(model, eager, (5, 9, 3)), _fill(model, comp, (5, 9, 3))
    for step in range(8):
        out_e = eager.step({r: s[-1] for r, s in seqs_e.items()})
        out_c = comp.step({r: s[-1] for r, s in seqs_c.items()})
        for r in out_e:
            err = float((out_e[r] - out_c[r]).abs().max())
            assert err < ATOL, f"step {step} row {r}: compiled vs eager max|logit diff| {err:.2e}"
            seqs_e[r].append(int(out_e[r].argmax()))
            seqs_c[r].append(int(out_c[r].argmax()))
    assert ptrs == [(k.data_ptr(), v.data_ptr()) for k, v in zip(comp.k, comp.v)], "compiled path re-created the cache"
    for l in range(len(comp.k)):
        assert torch.equal(comp.k[l], eager.k[l]) and torch.equal(comp.v[l], eager.v[l]), (
            f"layer {l}: the compiled step wrote different K/V into the cache than the eager step")


def test_whole_forward_is_one_graph_and_batch_size_changes_do_not_recompile_each_time(model):
    from torch._dynamo.utils import counters

    counters.clear()
    comp = _batch(model, True, slots=6)                                  # warm_compile already ran B=1 and B=2
    seqs = _fill(model, comp, (4, 6, 8, 5, 7))
    for _ in range(3):
        comp.step({r: s[-1] for r, s in seqs.items()})
        for r in list(seqs):
            seqs[r].append(0)
    comp.remove("r0"); comp.remove("r1"); comp.remove("r2"); comp.remove("r3")      # down to a single row: B == 1
    comp.step({"r4": seqs["r4"][-1]})
    assert counters["graph_break"] == {}, f"graph breaks: {dict(counters['graph_break'])}"
    assert counters["stats"]["unique_graphs"] <= 3, (
        f"{counters['stats']['unique_graphs']} graphs for batch sizes 1..5: expected one for B==1 and one symbolic for B>=2")


def test_the_attention_call_is_a_registered_op_that_declares_the_cache_writes():
    op = torch.ops.llmstack.kvcache_attention.default
    mutated = {a.name for a in op._schema.arguments if a.alias_info is not None and a.alias_info.is_write}
    assert mutated == {"k_cache", "v_cache"}, f"the op must declare exactly the two caches as mutated, got {mutated}"


def test_eager_and_op_paths_agree_on_the_attention_itself():
    from cs336_basics.transfromer.kv_cache_attention import kvcache_attention_reference

    g = torch.Generator().manual_seed(3)
    B, H, D, S, ML = 3, 2, 8, 4, 16
    kc, vc = torch.randn(S, ML, H, D, generator=g), torch.randn(S, ML, H, D, generator=g)
    q, kn, vn = (torch.randn(B, 1, H, D, generator=g) for _ in range(3))
    seqlens = torch.tensor([5, 0, 9], dtype=torch.int32)
    idx = torch.tensor([2, 0, 3], dtype=torch.int32)
    kc2, vc2 = kc.clone(), vc.clone()
    ref = kvcache_attention_reference(q, kc, vc, kn, vn, seqlens, idx)
    got = torch.ops.llmstack.kvcache_attention(q, kc2, vc2, kn, vn, seqlens, idx, "reference")
    assert torch.equal(got, ref) and torch.equal(kc, kc2) and torch.equal(vc, vc2)


async def _engine_outputs(model, compiled: bool, prompts: Dict[str, List[int]], new_tokens: int):
    queue: asyncio.Queue = asyncio.Queue()
    store: Dict[str, RequestData] = {}
    adapter = TransformerLMAdapter(model, context_length=CTX, eos_token_id=-1, vocab_size=VOCAB, use_kv_cache=True,
                                   decode_backend="prealloc", decode_slots=8, decode_max_len=CTX)
    adapter.compile_decode, adapter.compile_backend = compiled, "aot_eager"
    engine = InferenceEngine(adapter, queue, store)
    for rid, p in prompts.items():
        store[rid] = RequestData(request_id=rid, user_id="u", timestamp=time.time(), prompt_text="",
                                 max_token_length=len(p) + new_tokens)
        queue.put_nowait(TokenizedData(request_id=rid, tokens=list(p)))
    engine.run()
    await asyncio.wait_for(engine.shutdown(draining=True), timeout=120)
    assert all(store[r].status == RequestStatus.DONE for r in prompts)
    return {r: list(store[r].generated_tokens) for r in prompts}


@pytest.mark.asyncio
async def test_engine_with_compiled_decode_generates_the_same_tokens(model):
    prompts = {f"r{i}": _prompt(n, seed=60 + i) for i, n in enumerate((4, 11, 7, 2, 9))}
    base = await _engine_outputs(model, False, prompts, new_tokens=8)
    comp = await _engine_outputs(model, True, prompts, new_tokens=8)
    assert comp == base
