"""C3 -- convert the self-written TransformerLM checkpoint to a HuggingFace `LlamaForCausalLM`
and check that both models produce the same logits.

Why this exists: HF generate / HF Trainer / SGLang can only load an HF-format model, so every
cross-framework comparison in this directory is gated on this conversion being numerically right.

What has to match (verified against the model source, see README "C3"):

  self-written key                         HF key                                         transform
  ---------------------------------------  ---------------------------------------------  --------------------------
  in_embedding.embed_mat      (V, d)       model.embed_tokens.weight                      copy
  tf_layers.i.RMSN1.gain      (d,)         model.layers.i.input_layernorm.weight          copy
  tf_layers.i.MHA.W_Q  (h*dk, d)           model.layers.i.self_attn.q_proj.weight         per-head de-interleave
  tf_layers.i.MHA.W_K  (h*dk, d)           model.layers.i.self_attn.k_proj.weight         per-head de-interleave
  tf_layers.i.MHA.W_V  (h*dk, d)           model.layers.i.self_attn.v_proj.weight         copy
  tf_layers.i.MHA.W_O  (d, h*dk)           model.layers.i.self_attn.o_proj.weight         copy
  tf_layers.i.RMSN2.gain      (d,)         model.layers.i.post_attention_layernorm.weight copy
  tf_layers.i.FNN.W1   (ff, d)             model.layers.i.mlp.gate_proj.weight            copy   (SiLU branch)
  tf_layers.i.FNN.W3   (ff, d)             model.layers.i.mlp.up_proj.weight              copy   (linear branch)
  tf_layers.i.FNN.W2   (d, ff)             model.layers.i.mlp.down_proj.weight            copy
  norm.gain                   (d,)         model.norm.weight                              copy
  head.weightMat              (V, d)       lm_head.weight                                 copy  (NOT tied)
  token_positions                          (dropped: buffer)

RoPE: the self-written PosEncod rotates *adjacent* pairs (x0,x1),(x2,x3)... (interleaved);
HF Llama's `rotate_half` pairs (x_j, x_{j+d/2}) (half-split). The two are the same rotation
if, inside every head, the q/k output rows are permuted [0,2,4,...,1,3,5,...]. q.k dot products
are invariant to applying the same permutation to q and k; v is untouched.

Usage:
  python convert_190m_to_hf.py --ckpt ../../../training/ckpt_190m_fwe/model_only.pt --out /tmp/llm190m-hf
  python convert_190m_to_hf.py --ckpt ... --out ... --tokenizer gpt2     # also saves the GPT-2 tokenizer (network)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch

N_HEADS_DEFAULT = 12
THETA_DEFAULT = 10000.0
EPS_DEFAULT = 1e-5


def _strip_prefixes(sd: dict) -> dict:
    """torch.compile / DDP wrappers add `_orig_mod.` / `module.` / `model.` prefixes."""
    out = {}
    for k, v in sd.items():
        changed = True
        while changed:
            changed = False
            for p in ("_orig_mod.", "module.", "model."):
                if k.startswith(p):
                    k, changed = k[len(p):], True
        out[k] = v
    return out


def load_state_dict(path: str | Path) -> dict:
    obj = torch.load(path, map_location="cpu", weights_only=True)
    sd = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    return _strip_prefixes(sd)


def infer_config(sd: dict, n_heads: int = N_HEADS_DEFAULT, theta: float = THETA_DEFAULT,
                 eps: float = EPS_DEFAULT) -> dict:
    """Everything except n_heads / theta / eps can be read off the tensor shapes."""
    vocab, d_model = sd["in_embedding.embed_mat"].shape
    n_layers = 1 + max(int(m.group(1)) for k in sd
                       if (m := re.match(r"tf_layers\.(\d+)\.", k)))
    d_ff = sd["tf_layers.0.FNN.W1"].shape[0]
    ctx = sd["token_positions"].shape[0] if "token_positions" in sd else 1024
    assert d_model % n_heads == 0, f"d_model {d_model} not divisible by n_heads {n_heads}"
    return dict(vocab_size=vocab, d_model=d_model, n_layers=n_layers, d_ff=d_ff,
                n_heads=n_heads, context_length=ctx, theta=theta, eps=eps)


def deinterleave_rows(w: torch.Tensor, n_heads: int) -> torch.Tensor:
    """(h*dk, d) -> same shape; inside each head, rows [0,1,2,3,...] -> [0,2,4,...,1,3,5,...]."""
    out_dim, in_dim = w.shape
    dk = out_dim // n_heads
    perm = torch.cat([torch.arange(0, dk, 2), torch.arange(1, dk, 2)])
    return w.view(n_heads, dk, in_dim).index_select(1, perm).reshape(out_dim, in_dim).contiguous()


def convert_state_dict(sd: dict, cfg: dict) -> dict:
    h = cfg["n_heads"]
    out = {
        "model.embed_tokens.weight": sd["in_embedding.embed_mat"],
        "model.norm.weight": sd["norm.gain"],
        "lm_head.weight": sd["head.weightMat"],
    }
    for i in range(cfg["n_layers"]):
        s, d = f"tf_layers.{i}.", f"model.layers.{i}."
        out[d + "input_layernorm.weight"] = sd[s + "RMSN1.gain"]
        out[d + "post_attention_layernorm.weight"] = sd[s + "RMSN2.gain"]
        out[d + "self_attn.q_proj.weight"] = deinterleave_rows(sd[s + "MHA.W_Q"], h)
        out[d + "self_attn.k_proj.weight"] = deinterleave_rows(sd[s + "MHA.W_K"], h)
        out[d + "self_attn.v_proj.weight"] = sd[s + "MHA.W_V"]
        out[d + "self_attn.o_proj.weight"] = sd[s + "MHA.W_O"]
        out[d + "mlp.gate_proj.weight"] = sd[s + "FNN.W1"]   # SiLU branch
        out[d + "mlp.up_proj.weight"] = sd[s + "FNN.W3"]     # linear branch
        out[d + "mlp.down_proj.weight"] = sd[s + "FNN.W2"]
    return {k: v.detach().clone().contiguous() for k, v in out.items()}


def build_hf_model(cfg: dict, hf_sd: dict):
    from transformers import LlamaConfig, LlamaForCausalLM

    hf_cfg = LlamaConfig(
        vocab_size=cfg["vocab_size"],
        hidden_size=cfg["d_model"],
        intermediate_size=cfg["d_ff"],
        num_hidden_layers=cfg["n_layers"],
        num_attention_heads=cfg["n_heads"],
        num_key_value_heads=cfg["n_heads"],      # MHA
        hidden_act="silu",
        max_position_embeddings=cfg["context_length"],
        rms_norm_eps=cfg["eps"],
        rope_theta=cfg["theta"],                 # works on transformers 4.x and 5.x (-> rope_parameters)
        tie_word_embeddings=False,               # embedding and head are independent here
        attention_bias=False,
        mlp_bias=False,
        use_cache=True,
        bos_token_id=50256 if cfg["vocab_size"] == 50257 else None,   # GPT-2 <|endoftext|>
        eos_token_id=50256 if cfg["vocab_size"] == 50257 else None,
    )
    model = LlamaForCausalLM(hf_cfg)
    if hf_sd is None:          # --random-init: architecture only, values are the HF initialisation
        return model.eval()
    missing, unexpected = model.load_state_dict(hf_sd, strict=False)
    # rotary_emb.inv_freq is a non-persistent buffer in HF, so it may legitimately be "missing".
    real_missing = [k for k in missing if "rotary_emb" not in k]
    if real_missing or unexpected:
        raise RuntimeError(f"state_dict mismatch: missing={real_missing[:5]} unexpected={list(unexpected)[:5]}")
    return model.eval()


def build_reference_model(sd: dict, cfg: dict):
    """The self-written TransformerLM, loaded from the same state dict (fp32, CPU)."""
    from cs336_basics.lm import TransformerLM

    m = TransformerLM(
        vocab_size=cfg["vocab_size"], context_length=cfg["context_length"],
        num_layers=cfg["n_layers"], d_model=cfg["d_model"], heads_num=cfg["n_heads"],
        d_ff=cfg["d_ff"], theta=cfg["theta"], eps=cfg["eps"], device="cpu", dtype=torch.float32,
    )
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if [k for k in missing if k != "token_positions"] or unexpected:
        raise RuntimeError(f"reference state_dict mismatch: missing={missing[:5]} unexpected={list(unexpected)[:5]}")
    return m.eval()


@torch.no_grad()
def compare_logits(ref, hf, vocab: int, seq: int = 48, batch: int = 2, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (batch, seq), generator=g)
    a = ref(ids).float()
    b = hf(input_ids=ids).logits.float()
    diff = (a - b).abs()
    return dict(
        max_abs_diff=float(diff.max()),
        mean_abs_diff=float(diff.mean()),
        argmax_agree=float((a.argmax(-1) == b.argmax(-1)).float().mean()),
        logit_scale=float(a.abs().mean()),
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", help="model_only.pt or a full checkpoint with a 'model' key")
    ap.add_argument("--out", required=True, help="output dir for the HF model")
    ap.add_argument("--random-init", action="store_true",
                    help="no checkpoint: build the 190M architecture with RANDOM weights. For speed "
                         "benchmarks only (throughput/latency do not depend on weight values); "
                         "correctness (C3) must be checked with real weights. Writes RANDOM_INIT.txt.")
    ap.add_argument("--vocab-size", type=int, default=50257)
    ap.add_argument("--d-model", type=int, default=768)
    ap.add_argument("--n-layers", type=int, default=16)
    ap.add_argument("--d-ff", type=int, default=2048)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-heads", type=int, default=N_HEADS_DEFAULT)
    ap.add_argument("--theta", type=float, default=THETA_DEFAULT)
    ap.add_argument("--eps", type=float, default=EPS_DEFAULT)
    ap.add_argument("--tol", type=float, default=1e-3, help="pass criterion on max |logit diff| (fp32)")
    ap.add_argument("--tokenizer", default=None, help="e.g. 'gpt2' -> also save that HF tokenizer (needs network)")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args(argv)

    if args.random_init:
        torch.manual_seed(args.seed)
        cfg = dict(vocab_size=args.vocab_size, d_model=args.d_model, n_layers=args.n_layers,
                   d_ff=args.d_ff, n_heads=args.n_heads, context_length=args.ctx,
                   theta=args.theta, eps=args.eps)
        print("config:", cfg)
        print("WARNING: RANDOM weights. Valid for throughput / latency only; says nothing about "
              "correctness or output quality.", file=sys.stderr)
        hf = build_hf_model(cfg, None)
        args.no_verify = True
        sd = None
    else:
        if not args.ckpt:
            ap.error("--ckpt is required unless --random-init is given")
        sd = load_state_dict(args.ckpt)
        cfg = infer_config(sd, args.n_heads, args.theta, args.eps)
        print("config:", cfg)
        hf = build_hf_model(cfg, convert_state_dict(sd, cfg))

    if not args.no_verify:
        ref = build_reference_model(sd, cfg)
        res = compare_logits(ref, hf, cfg["vocab_size"])
        print("C3 logits check:", res)
        if res["max_abs_diff"] >= args.tol:
            print(f"FAIL: max_abs_diff {res['max_abs_diff']:.3g} >= tol {args.tol}", file=sys.stderr)
            return 1
        print(f"PASS: max_abs_diff {res['max_abs_diff']:.3g} < {args.tol}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hf.save_pretrained(out, safe_serialization=True)
    if args.random_init:
        (out / "RANDOM_INIT.txt").write_text(
            "Random-initialised weights (seed %d). Architecture-only: use for speed benchmarks, "
            "never for correctness or quality claims.\n" % args.seed)
    if args.tokenizer:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(args.tokenizer).save_pretrained(out)
    print("saved HF model to", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
