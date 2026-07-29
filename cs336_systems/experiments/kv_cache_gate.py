"""Correctness gate for the inference KV cache.

The cached path is an optimization, so the only acceptable outcome is that it changes
speed and nothing else. This checks that against the un-cached path that is already
trusted (`TransformerLM.forward`, the same code the model was trained with):

  1. logits agree within fp32 tolerance at every position, and
  2. greedy decoding produces byte-identical token sequences.

(2) is the one that actually matters -- tiny logit differences are expected from different
reduction orders, but they must never change an argmax.

Exit code 0 means the cache is safe to use. Anything else means it is not, and the caller
should not enable it.
"""

import argparse
import json
import sys

import torch

from cs336_basics.lm import TransformerLM


def build_small_model(device, seed=0):
    torch.manual_seed(seed)
    return TransformerLM(
        vocab_size=257, context_length=128, num_layers=4, d_model=64,
        heads_num=4, d_ff=176, theta=10000.0, device=str(device), dtype=torch.float32,
    ).eval()


@torch.no_grad()
def check_logits_match(model, device, prompt_len=16, tol=2e-4) -> tuple[bool, float]:
    """Prefill through the cache must reproduce the plain forward exactly."""
    vocab = model.head.weightMat.shape[0]
    ids = torch.randint(0, vocab, (1, prompt_len), device=device)
    ref = model(ids)
    cached, _ = model.forward_with_cache(ids, past_kv=None)
    delta = (ref - cached).abs().max().item()
    return delta < tol, delta


@torch.no_grad()
def check_greedy_match(model, device, prompt_len=12, n_new=24) -> tuple[bool, list, list]:
    """The real test: decode the same continuation with and without the cache."""
    vocab = model.head.weightMat.shape[0]
    torch.manual_seed(1234)
    prompt = torch.randint(0, vocab, (1, prompt_len), device=device)

    # Un-cached reference: re-run the whole prefix every step.
    ids = prompt.clone()
    recompute = []
    for _ in range(n_new):
        nxt = torch.argmax(model(ids)[:, -1, :], dim=-1, keepdim=True)
        recompute.append(int(nxt.item()))
        ids = torch.cat([ids, nxt], dim=1)

    # Cached: prefill once, then one token at a time.
    logits, past = model.forward_with_cache(prompt, past_kv=None)
    nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    cached = [int(nxt.item())]
    for _ in range(n_new - 1):
        logits, past = model.forward_with_cache(nxt, past_kv=past)
        nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        cached.append(int(nxt.item()))

    return recompute == cached, recompute, cached


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", help="Also gate against the real trained model, if given.")
    p.add_argument("--checkpoint")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    failures = []

    print("=== gate 1: small random model ===")
    model = build_small_model(device)
    ok, delta = check_logits_match(model, device)
    print(f"  prefill logits max|delta| = {delta:.3e}  -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append("small/logits")
    ok, ref, cac = check_greedy_match(model, device)
    print(f"  greedy tokens identical    -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append("small/greedy")
        print(f"    recompute: {ref}\n    cached:    {cac}")

    if args.config and args.checkpoint:
        print("\n=== gate 2: real trained model ===")
        cfg = json.load(open(args.config))
        m = cfg["model"]
        real = TransformerLM(
            vocab_size=m["vocab_size"], context_length=m["context_length"],
            num_layers=m["num_layers"], d_model=m["d_model"], heads_num=m["num_heads"],
            d_ff=m["d_ff"], theta=m.get("rope_theta", 10_000.0),
            device=str(device), dtype=torch.float32,
        )
        ckpt = torch.load(args.checkpoint, map_location=device)
        real.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()})
        real.to(device).eval()

        ok, delta = check_logits_match(real, device, prompt_len=16)
        print(f"  prefill logits max|delta| = {delta:.3e}  -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append("real/logits")
        ok, ref, cac = check_greedy_match(real, device, prompt_len=12, n_new=16)
        print(f"  greedy tokens identical    -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append("real/greedy")
            print(f"    recompute: {ref}\n    cached:    {cac}")

    if failures:
        print(f"\nGATE FAILED: {failures}. KV cache must NOT be enabled.")
        sys.exit(1)
    print("\nGATE PASSED: cached decode is token-identical to recompute.")


if __name__ == "__main__":
    main()
