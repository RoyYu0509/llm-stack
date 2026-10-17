"""The large random-init presets and --context-length that the KV-cache size sweep (benchmarks/compare/remote/run_kv_sweep.sh)
depends on.

What must hold before the GPU is paid for: (1) a preset is the size its name says (a '6.9b' that is really 4B would make the whole
sweep mislabelled), (2) weights + KV cache for the biggest planned cell fit the 24 GB card on paper, (3) --context-length really
changes the model and the recording says so. Shapes are built on the meta device: no memory, no time.
"""
import functools
import json
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("cs336_basics.lm", reason="needs the training package")
SERVING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVING / "benchmarks"))
import record_demo as rd  # noqa: E402

NOMINAL = {"190m": 190e6, "1.4b": 1.4e9, "2.8b": 2.8e9, "6.9b": 6.9e9}
GIB = 2**30


@functools.lru_cache(maxsize=None)
def n_params(name):
    m = rd.build_model(rd.PRESETS[name], "meta", 0, None, rd.get_attention("naive"))
    return sum(p.numel() for p in m.parameters())


@pytest.mark.parametrize("name", list(NOMINAL))
def test_preset_has_the_size_its_name_says(name):
    got = n_params(name)
    assert got == pytest.approx(NOMINAL[name], rel=0.05), f"preset {name!r} has {got / 1e9:.2f}B parameters"


@pytest.mark.parametrize("name", ["1.4b", "2.8b", "6.9b"])
def test_large_presets_use_head_dim_128_like_the_flash_kernels_expect(name):
    c = rd.PRESETS[name]
    assert c["d_model"] % c["num_heads"] == 0 and c["d_model"] // c["num_heads"] == 128


def kv_bytes_per_token(c):  # K and V, every layer, bf16, multi-head (no GQA)
    return 2 * c["num_layers"] * c["d_model"] * 2


def test_biggest_planned_cell_fits_the_card_on_paper():
    """6.9b, 12288-token prompt: bf16 weights + the KV cache of the whole sequence must leave room for activations on 24 GB.
    (16384 tokens was tried on the card first: weights 12.8 + KV 8.0 GiB + prefill activations ran out of memory at 22 of 23.55 GiB.)"""
    c = rd.PRESETS["6.9b"]
    resident = n_params("6.9b") * 2 + kv_bytes_per_token(c) * (12288 + 8)
    assert kv_bytes_per_token(c) == 512 * 1024
    assert resident < 20 * GIB, f"{resident / GIB:.1f} GiB resident before any activation"
    assert resident > 17 * GIB, "the cell no longer stresses the card; the sweep's point was a model that fills 24 GB"


def run_record(tmp_path, *extra):
    out = tmp_path / "t.json"
    argv = ["record_demo.py", "--preset", "small", "--synthetic", "--prompt-len", "16", "16", "--new-tokens", "4", "--attn", "naive",
            "--mode", "seq", "--dtype", "float32", "--device", "cpu", "--n-requests", "2", "--concurrency", "1",
            "--warmup-requests", "0", "--reps", "1", "--arm-name", "t", "--out", str(out), *extra]
    old, sys.argv = sys.argv, argv
    try:
        rd.main()
    finally:
        sys.argv = old
    return json.loads(out.read_text())   # one repetition: the file is the --out path itself


def test_context_length_override_reaches_the_model_and_the_recording(tmp_path):
    rec = run_record(tmp_path, "--context-length", "128", "--no-kv-cache")
    assert rec["meta"]["model_cfg"]["context_length"] == 128
    assert all(r["prompt_len"] == 16 and len(r["tokens"]) == 4 for r in rec["requests"])


def test_without_the_override_the_preset_keeps_its_own_context_length(tmp_path):
    assert run_record(tmp_path)["meta"]["model_cfg"]["context_length"] == rd.PRESETS["small"]["context_length"]


def test_context_length_cannot_be_combined_with_a_checkpoint(tmp_path):
    with pytest.raises(SystemExit):
        run_record(tmp_path, "--context-length", "128", "--checkpoint", str(tmp_path / "x.pt"))


# ---- sdpa on the model's real tensor layout -------------------------------------------------------------------------------------

def _model_layout(S=8, H=2, D=4, seed=0):
    """q/k/v as the model's einsum projections hand them to the attention kernel: (H, S, D) views whose LAST dim has stride > 1."""
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(S, D, H, generator=g).permute(2, 0, 1) for _ in range(3)]


def test_sdpa_wrapper_hands_the_fused_kernels_a_last_dim_stride_of_one_on_the_causal_call(monkeypatch):
    """Found on the GPU: with stride-32 q/k and stride-16384 v, SDPA fell back to math attention and asked for a 32 GiB score matrix."""
    import torch.nn.functional as F

    seen = []
    real = F.scaled_dot_product_attention
    monkeypatch.setattr(F, "scaled_dot_product_attention", lambda q, k, v, **kw: (seen.append((q.stride(-1), k.stride(-1), v.stride(-1))), real(q, k, v, **kw))[1])
    q, k, v = _model_layout()
    assert q.stride(-1) != 1 and v.stride(-1) != 1
    out = rd.sdpa_kernel_fn(q, k, v, is_causal=True)
    assert seen == [(1, 1, 1)]
    ref = real(q.contiguous().unsqueeze(0), k.contiguous().unsqueeze(0), v.contiguous().unsqueeze(0), is_causal=True).squeeze(0)
    assert torch.allclose(out, ref)


def test_sdpa_wrapper_leaves_the_decode_call_alone(monkeypatch):
    """Decode (is_causal False, one query): no copy of the whole KV cache per layer per step -- that would be a cost of the harness, not of the cache."""
    import torch.nn.functional as F

    seen = []
    real = F.scaled_dot_product_attention
    monkeypatch.setattr(F, "scaled_dot_product_attention", lambda q, k, v, **kw: (seen.append((q.stride(-1), v.stride(-1))), real(q, k, v, **kw))[1])
    q, k, v = _model_layout()
    rd.sdpa_kernel_fn(q[:, :1], k, v, is_causal=False)
    assert seen[0][0] == q.stride(-1) and seen[0][1] == v.stride(-1), "decode inputs were copied"


# ---- the vectorised RoPE table ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("d_k,seq,theta", [(8, 33, 10000.0), (64, 100, 10000.0), (7, 20, 500.0)])
def test_vectorised_rope_table_equals_the_original_per_entry_loop(d_k, seq, theta):
    from cs336_basics.transfromer.positionalNencoding import PosEncod

    ref = PosEncod(theta, d_k, seq)
    fast = PosEncod.__new__(PosEncod)
    rd._vectorised_posencod_init(fast, theta, d_k, seq)
    assert fast.Rs.shape == ref.Rs.shape == (seq, (d_k + d_k % 2) // 2, 2, 2) and fast.Rs.dtype == ref.Rs.dtype
    assert torch.allclose(fast.Rs, ref.Rs, rtol=0, atol=1e-5), float((fast.Rs - ref.Rs).abs().max())
    x = torch.randn(2, 3, seq, d_k)
    pos = torch.arange(seq)[None, :].expand(2, seq)   # the module wants (batch, seq) positions for (batch, heads, seq, d_k) input
    assert torch.allclose(fast(x, pos), ref(x, pos), atol=1e-4)


def test_fast_rope_init_changes_nothing_but_the_table_build_and_restores_the_class():
    from cs336_basics.transfromer.positionalNencoding import PosEncod

    cfg = dict(vocab_size=64, context_length=40, num_layers=2, d_model=32, num_heads=4, d_ff=64, rope_theta=10000.0)
    before = PosEncod.__init__
    slow = rd.build_model(cfg, "cpu", 0, None, rd.get_attention("naive"))
    fast = rd.build_model(cfg, "cpu", 0, None, rd.get_attention("naive"), fast_rope_init=True)
    assert PosEncod.__init__ is before, "the patch must not leak out of build_model"
    for (n, a), (_, b) in zip(slow.state_dict().items(), fast.state_dict().items()):
        assert torch.equal(a, b), f"weights differ at {n}: the RoPE shortcut must not touch the random stream"
    ids = torch.randint(0, 64, (1, 12))
    assert torch.allclose(slow(ids), fast(ids), atol=1e-4)
