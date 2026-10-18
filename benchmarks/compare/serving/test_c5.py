"""CPU tests for C5 (engine vs HF greedy). Tiny random model, so no checkpoint and no GPU are needed.

The point of the mutation test: C5 is only evidence if it can fail. A batched-decode bug that ignores the padding
mask must be caught by the `cb` path while the single-sequence `seq` path stays exact -- that is exactly the kind
of bug the continuous-batching engine could have.
"""
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c5_greedy_equal as c5  # noqa: E402

pytest.importorskip("cs336_basics.lm")
pytest.importorskip("transformers")
import convert_190m_to_hf as c3  # noqa: E402
import record_demo as rd  # noqa: E402

CFG = dict(vocab_size=97, context_length=64, num_layers=2, d_model=64, num_heads=4, d_ff=128, rope_theta=10000.0)


def _models(attn_fn, seed=0):
    from cs336_basics.lm import TransformerLM

    torch.manual_seed(seed)
    ref = TransformerLM(vocab_size=CFG["vocab_size"], context_length=CFG["context_length"], num_layers=CFG["num_layers"],
                        d_model=CFG["d_model"], heads_num=CFG["num_heads"], d_ff=CFG["d_ff"], theta=CFG["rope_theta"],
                        device="cpu", dtype=torch.float32, attention_fn=attn_fn).eval()
    with torch.no_grad():
        for n, p in ref.named_parameters():
            if n.endswith(".gain"):
                p.add_(0.1 * torch.randn_like(p))
    for p in ref.parameters():
        p.requires_grad_(False)
    sd = dict(ref.state_dict())
    cfg = c3.infer_config(sd, n_heads=CFG["num_heads"], theta=CFG["rope_theta"])
    hf = c3.build_hf_model(cfg, c3.convert_state_dict(sd, cfg)).eval()
    return ref, hf


def _prompts():
    g = torch.Generator().manual_seed(1)
    lens = [3, 17, 9, 25, 5, 31, 12, 20]            # very different lengths -> ragged, left-padded decode batches
    return [torch.randint(0, CFG["vocab_size"], (n,), generator=g).tolist() for n in lens]


@pytest.fixture(scope="module")
def adapters():
    return rd.load_adapters(rd.SERVING / "tests" / "m2_adapters.py")


def test_first_divergence_helper():
    assert c5.first_divergence([1, 2, 3], [1, 2, 3]) is None
    assert c5.first_divergence([1, 2, 3], [1, 9, 3]) == 1
    assert c5.first_divergence([1, 2], [1, 2, 3]) == 2, "a shorter output is a divergence at the point it stops"


def test_engine_matches_hf_greedy_in_fp32_seq_and_cb(adapters):
    torch.set_float32_matmul_precision("highest")
    engine, hf = _models(rd.get_attention("naive"))
    rows = c5.compare_all(rd, adapters, engine, hf, _prompts(), 12, ["seq", "cb"], 4, "cpu", strict=True)
    for r in rows:
        assert r["exact"] == r["n_prompts"], (
            f"{r['mode']}: engine and HF differ (first divergence per prompt {r['first_divergence']}). "
            "seq=single-sequence KV cache, cb=left-padded batched decode -- see serving/inferenceLM/engine")
        assert r["teacher_forced_mean"] == pytest.approx(1.0)


def test_cb_only_bug_is_caught_and_seq_stays_exact(adapters):
    """Mutant: the attention wrapper ignores the padding mask. Single-sequence decode has no padding, so `seq`
    must stay exact; batched decode attends to pad slots, so `cb` must NOT be exact."""
    torch.set_float32_matmul_precision("highest")
    naive = rd.get_attention("naive")

    def ignores_mask(q, k, v, is_causal=False, attn_mask=None):
        return naive(q, k, v, is_causal)

    engine, hf = _models(ignores_mask)
    rows = {r["mode"]: r for r in c5.compare_all(rd, adapters, engine, hf, _prompts(), 12, ["seq", "cb"], 4, "cpu",
                                                  strict=True)}
    assert rows["seq"]["exact"] == rows["seq"]["n_prompts"]
    assert rows["cb"]["exact"] < rows["cb"]["n_prompts"], "C5 cannot see a batched-decode mask bug"
    assert rows["cb"]["passed"] is False


def test_non_fp32_runs_are_report_only(adapters):
    engine, hf = _models(rd.get_attention("naive"))
    rows = c5.compare_all(rd, adapters, engine, hf, _prompts()[:2], 4, ["seq"], 1, "cpu", strict=False)
    assert rows[0]["passed"] is None


# ---------------------------------------------------------------------------------------------------------------
# the OPTIMISED cb path: pre-allocated KV, batched prefill, last-position logits, torch.compile (aot_eager on CPU)
# ---------------------------------------------------------------------------------------------------------------
@pytest.fixture
def fresh_dynamo():
    import torch._dynamo as dyn

    dyn.reset()
    yield
    dyn.reset()


def _opts(**kw):
    base = dict(decode_backend="prealloc", last_logits=True, prefill_batch_tokens=256, compile_decode=False,
                compile_backend="aot_eager")
    return {**base, **kw}


def test_optimised_path_matches_hf_in_fp32_prealloc_batched_prefill_and_compiled(adapters, fresh_dynamo):
    torch.set_float32_matmul_precision("highest")
    engine, hf = _models(rd.get_attention("naive"))
    for compiled in (False, True):
        rows = c5.compare_all(rd, adapters, engine, hf, _prompts(), 12, ["cb"], 8, "cpu", strict=True,
                              opts=_opts(compile_decode=compiled))
        r = rows[0]
        assert r["exact"] == r["n_prompts"], (
            f"optimised cb path (compile={compiled}) differs from HF: first divergence per prompt {r['first_divergence']}. "
            "Pre-allocated KV / right-padded batched prefill / last-position logits / compiled decode: one of them computes a different function")
        assert r["teacher_forced_mean"] == pytest.approx(1.0)


def test_a_wrong_prompt_length_in_the_preallocated_cache_is_caught(adapters, monkeypatch):
    """Mutant: the slot's length is one short, so every decode step forgets the last prompt token."""
    from inferenceLM.engine.prealloc_batch import PreallocDecodeBatch

    torch.set_float32_matmul_precision("highest")
    real_add = PreallocDecodeBatch.add

    def add_short(self, key, row_past):
        real_add(self, key, row_past)
        slot = self._slot[key]
        self.lens[slot] -= 1
        self._lens_cpu[slot] -= 1

    monkeypatch.setattr(PreallocDecodeBatch, "add", add_short)
    engine, hf = _models(rd.get_attention("naive"))
    r = c5.compare_all(rd, adapters, engine, hf, _prompts(), 12, ["cb"], 8, "cpu", strict=True, opts=_opts())[0]
    assert r["exact"] < r["n_prompts"] and r["passed"] is False, "C5 cannot see a corrupted pre-allocated cache"


def test_engine_opts_validation():
    assert c5.engine_opts("fa2")["decode_backend"] == "prealloc" and c5.engine_opts("naive")["decode_backend"] == "ragged"
    with pytest.raises(SystemExit, match="prealloc"):
        c5.engine_opts("naive", compile_decode=True)
    with pytest.raises(SystemExit, match="--modes cb"):
        c5.engine_opts("fa2", compile_decode=True, modes=("seq", "cb"))
    with pytest.raises(SystemExit, match="--modes cb"):
        c5.engine_opts("fa2", prefill_batch_tokens=4096, modes=("seq",))


def test_a_run_aborts_when_the_labelled_path_never_ran():
    ok = dict(kv_stats=dict(steps=5, compiled=1), prefill_group_sizes=[8, 3])
    c5.check_exercised(ok, _opts(compile_decode=True))
    with pytest.raises(SystemExit, match="never ran a step"):
        c5.check_exercised(dict(kv_stats=dict(steps=0, compiled=0), prefill_group_sizes=[2]), _opts())
    with pytest.raises(SystemExit, match="not compiled"):
        c5.check_exercised(dict(kv_stats=dict(steps=5, compiled=0), prefill_group_sizes=[2]), _opts(compile_decode=True))
    with pytest.raises(SystemExit, match="more than one prompt"):
        c5.check_exercised(dict(kv_stats=dict(steps=5, compiled=0), prefill_group_sizes=[1, 1, 1]), _opts())
    c5.check_exercised(dict(kv_stats=None, prefill_group_sizes=[]), dict(decode_backend="ragged"))   # the original path asks for nothing


def _cli_fixture(tmp_path):
    """A 12-head toy checkpoint (main() assumes the 190M model's 12 heads) plus a prompt workload."""
    import json

    from cs336_basics.lm import TransformerLM

    torch.manual_seed(0)
    m = TransformerLM(vocab_size=97, context_length=64, num_layers=2, d_model=96, heads_num=12, d_ff=128,
                      theta=10000.0, device="cpu", dtype=torch.float32).eval()
    ck = tmp_path / "model_only.pt"
    torch.save({"model": m.state_dict()}, ck)
    wl = tmp_path / "wl.json"
    wl.write_text(json.dumps(dict(prompts=_prompts())))
    return ck, wl


def test_cli_runs_the_optimised_path_and_names_it_in_the_csv(tmp_path, fresh_dynamo):
    import csv

    ck, wl = _cli_fixture(tmp_path)
    out = tmp_path / "c5.csv"
    rc = c5.main(["--checkpoint", str(ck), "--workload", str(wl), "--n-prompts", "8", "--new-tokens", "8", "--dtype", "float32",
                  "--device", "cpu", "--attn", "naive", "--decode-backend", "prealloc", "--prefill-last-logits",
                  "--prefill-batch-tokens", "256", "--compile-decode", "--compile-backend", "aot_eager",
                  "--label", "fa2_compiled", "--concurrency", "8", "--out", str(out)])
    assert rc == 0
    rows = list(csv.DictReader(open(out)))
    assert {r["system"] for r in rows} == {"engine_fa2_compiled_cb_vs_hf_generate"}, "optimised switches default to the cb mode only"
    assert all("compile=True" in r["variant"] and "prefill_batch_tokens=256" in r["variant"] and "decode=prealloc" in r["variant"]
               for r in rows)
    exact = next(r for r in rows if r["metric"] == "greedy_exact_match_prompts")
    assert float(exact["value"]) == 8 and "PASS" in exact["notes"]


def test_cli_original_path_keeps_its_old_csv_name_and_both_modes(tmp_path):
    import csv

    ck, wl = _cli_fixture(tmp_path)
    out = tmp_path / "c5.csv"
    assert c5.main(["--checkpoint", str(ck), "--workload", str(wl), "--n-prompts", "4", "--new-tokens", "6", "--dtype", "float32",
                    "--device", "cpu", "--concurrency", "4", "--out", str(out)]) == 0
    assert {r["system"] for r in csv.DictReader(open(out))} == {"engine_naive_seq_vs_hf_generate", "engine_naive_cb_vs_hf_generate"}


def test_cli_rejects_fa2_in_fp32_and_a_wrong_mode_combination(tmp_path):
    ck, wl = _cli_fixture(tmp_path)
    base = ["--checkpoint", str(ck), "--workload", str(wl), "--device", "cpu", "--out", str(tmp_path / "x.csv")]
    with pytest.raises(SystemExit, match="no fp32 kernel"):
        c5.main(base + ["--attn", "fa2", "--dtype", "float32"])
    with pytest.raises(SystemExit, match="--modes cb"):
        c5.main(base + ["--dtype", "float32", "--prefill-batch-tokens", "256", "--modes", "seq", "cb"])
