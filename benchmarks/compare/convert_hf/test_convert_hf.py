"""C3 tests. The tiny-model tests run anywhere on CPU; the real-checkpoint test is skipped
when the 190M weights are not on this machine.

Error messages name the component that is most likely wrong, so a red test points at a fix.
"""
import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import convert_190m_to_hf as c3  # noqa: E402

cs336 = pytest.importorskip("cs336_basics.lm", reason="needs the repo venv (cs336_basics importable)")
pytest.importorskip("transformers")

TINY = dict(vocab_size=97, d_model=64, n_layers=2, d_ff=128, n_heads=4, context_length=32,
            theta=10000.0, eps=1e-5)

REAL_CKPT = Path(os.environ.get(
    "LLM190M_CKPT",
    Path(__file__).resolve().parents[3] / "training" / "ckpt_190m_fwe" / "model_only.pt",
))


def _tiny_reference(seed=0):
    torch.manual_seed(seed)
    from cs336_basics.lm import TransformerLM

    m = TransformerLM(
        vocab_size=TINY["vocab_size"], context_length=TINY["context_length"],
        num_layers=TINY["n_layers"], d_model=TINY["d_model"], heads_num=TINY["n_heads"],
        d_ff=TINY["d_ff"], theta=TINY["theta"], eps=TINY["eps"], device="cpu", dtype=torch.float32,
    ).eval()
    # random init is already non-trivial (trunc-normal); perturb gains so RMSNorm weights matter
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith(".gain"):
                p.add_(0.1 * torch.randn_like(p))
    return m


def _convert(m):
    sd = {k: v for k, v in m.state_dict().items()}
    cfg = c3.infer_config(sd, n_heads=TINY["n_heads"], theta=TINY["theta"], eps=TINY["eps"])
    hf = c3.build_hf_model(cfg, c3.convert_state_dict(sd, cfg))
    return sd, cfg, hf


def test_infer_config_reads_shapes_correctly():
    sd, cfg, _ = _convert(_tiny_reference())
    for k in ("vocab_size", "d_model", "n_layers", "d_ff", "context_length"):
        assert cfg[k] == TINY[k], f"infer_config.{k}: got {cfg[k]}, expected {TINY[k]}"


def test_deinterleave_is_a_permutation_inside_each_head():
    w = torch.arange(8 * 3, dtype=torch.float32).view(8, 3)          # 2 heads x dk=4
    out = c3.deinterleave_rows(w, n_heads=2)
    # head 0 rows [0,1,2,3] -> [0,2,1,3]; head 1 rows [4,5,6,7] -> [4,6,5,7]
    assert out[:, 0].tolist() == [0, 6, 3, 9, 12, 18, 15, 21], (
        "deinterleave_rows must reorder rows as [0,2,..,1,3,..] inside each head only "
        "(RoPE interleaved -> half-split)"
    )


def test_tiny_model_logits_match_hf():
    ref = _tiny_reference()
    _, cfg, hf = _convert(ref)
    res = c3.compare_logits(ref, hf, TINY["vocab_size"], seq=24, batch=3)
    assert res["max_abs_diff"] < 1e-4, (
        f"logits differ by {res['max_abs_diff']:.3g}. Check in order: (1) RoPE pairing -- "
        f"q/k rows must be de-interleaved per head; (2) W1/W3 -> gate/up not swapped; "
        f"(3) rms eps; (4) tie_word_embeddings must be False. {res}"
    )
    assert res["argmax_agree"] == 1.0


def test_skipping_the_rope_permutation_is_detected():
    """Mutation check: if q/k are NOT permuted, the logits must differ. Otherwise the test
    above could not tell a wrong conversion from a right one (RoPE bug would be invisible)."""
    ref = _tiny_reference()
    sd = {k: v for k, v in ref.state_dict().items()}
    cfg = c3.infer_config(sd, n_heads=TINY["n_heads"], theta=TINY["theta"], eps=TINY["eps"])
    hf_sd = c3.convert_state_dict(sd, cfg)
    for i in range(cfg["n_layers"]):
        hf_sd[f"model.layers.{i}.self_attn.q_proj.weight"] = sd[f"tf_layers.{i}.MHA.W_Q"].clone()
        hf_sd[f"model.layers.{i}.self_attn.k_proj.weight"] = sd[f"tf_layers.{i}.MHA.W_K"].clone()
    hf = c3.build_hf_model(cfg, hf_sd)
    res = c3.compare_logits(ref, hf, TINY["vocab_size"], seq=24, batch=3)
    assert res["max_abs_diff"] > 1e-3, "un-permuted q/k should NOT match -- the check has no teeth"


def test_swapping_gate_and_up_is_detected():
    ref = _tiny_reference()
    sd = {k: v for k, v in ref.state_dict().items()}
    cfg = c3.infer_config(sd, n_heads=TINY["n_heads"], theta=TINY["theta"], eps=TINY["eps"])
    hf_sd = c3.convert_state_dict(sd, cfg)
    for i in range(cfg["n_layers"]):
        a, b = f"model.layers.{i}.mlp.gate_proj.weight", f"model.layers.{i}.mlp.up_proj.weight"
        hf_sd[a], hf_sd[b] = hf_sd[b], hf_sd[a]
    hf = c3.build_hf_model(cfg, hf_sd)
    res = c3.compare_logits(ref, hf, TINY["vocab_size"], seq=24, batch=3)
    assert res["max_abs_diff"] > 1e-3, "swapped gate/up should NOT match"


@pytest.mark.skipif(not REAL_CKPT.exists(), reason=f"190M checkpoint not found at {REAL_CKPT}")
def test_real_190m_checkpoint_matches_hf():
    sd = c3.load_state_dict(REAL_CKPT)
    cfg = c3.infer_config(sd)
    assert (cfg["vocab_size"], cfg["d_model"], cfg["n_layers"], cfg["d_ff"]) == (50257, 768, 16, 2048)
    hf = c3.build_hf_model(cfg, c3.convert_state_dict(sd, cfg))
    ref = c3.build_reference_model(sd, cfg)
    res = c3.compare_logits(ref, hf, cfg["vocab_size"], seq=48, batch=2)
    assert res["max_abs_diff"] < 1e-3, f"C3 acceptance criterion (fp32 max diff < 1e-3) failed: {res}"
    assert res["argmax_agree"] == 1.0, f"argmax disagrees somewhere: {res}"


def test_random_init_writes_marked_model_with_the_requested_architecture(tmp_path):
    out = tmp_path / "rand"
    rc = c3.main(["--random-init", "--out", str(out), "--vocab-size", "64", "--d-model", "32",
                  "--n-layers", "2", "--d-ff", "64", "--ctx", "32", "--n-heads", "4"])
    assert rc == 0
    assert (out / "RANDOM_INIT.txt").exists(), "random weights must be marked so nobody mistakes them"
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(out)
    assert (m.config.vocab_size, m.config.hidden_size, m.config.num_hidden_layers,
            m.config.intermediate_size) == (64, 32, 2, 64)
    assert m.config.tie_word_embeddings is False


def test_ckpt_is_required_without_random_init(tmp_path, capsys):
    with pytest.raises(SystemExit):
        c3.main(["--out", str(tmp_path / "x")])
