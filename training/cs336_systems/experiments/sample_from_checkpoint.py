"""Qualitative sampling from a pretraining checkpoint.

`cs336_basics/text_gen.py` can't be used for this run: it rebuilds the tokenizer with
`Tokenizer.from_files(vocab.pkl, merges.pkl)`, and this corpus was encoded with tiktoken's
pretrained gpt2 encoding instead of a custom-trained BPE (see real_pretrain_config.json's
_NOTE_tokenizer_swap) -- those two .pkl files are placeholders with no real merges in them.
It also re-encodes the whole prompt string on every generated token, which is both O(n^2)
and lossy across BPE boundaries.

This script reads the same JSON config the training run used (so the model dims can never
drift out of sync with the checkpoint), decodes with tiktoken, and carries token ids
forward instead of re-encoding text.

Usage:
    uv run python cs336_systems/experiments/sample_from_checkpoint.py \
        --config cs336_systems/experiments/real_pretrain_config.json \
        --checkpoint checkpoints_pretrain_v1/checkpoint_epoch_0100.pt
"""

import argparse
import json

import tiktoken
import torch

from cs336_basics.lm import TransformerLM

DEFAULT_PROMPTS = [
    # web / general prose -- the 70% of the mix
    "The history of the internet began in the",
    "In recent years, researchers have found that",
    # wikipedia-style -- the 20%
    "Paris is the capital and most populous city of",
    # instruction format -- the 10%, rendered exactly as build_mixed_corpus.py wrote it
    "### Instruction:\nExplain what a neural network is.\n### Response:\n",
]


def top_p_filter(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """Return a probability distribution over the *full* vocab with everything outside the
    nucleus zeroed out. Operating in log-space via softmax (not raw exp) so a cold model
    with large-magnitude logits doesn't overflow to inf/nan."""
    probs = torch.softmax(logits / temperature, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    # Keep every token up to and including the one that crosses top_p, so the nucleus is
    # never empty (the naive `cumulative <= top_p` mask drops everything when the top
    # token alone already exceeds top_p).
    keep = cumulative - sorted_probs < top_p
    keep[0] = True
    filtered = torch.zeros_like(probs)
    filtered[sorted_idx[keep]] = probs[sorted_idx[keep]]
    return filtered / filtered.sum()


@torch.no_grad()
def generate(model, enc, prompt, max_new_tokens, temperature, top_p, context_length, device, eot_id):
    ids = enc.encode(prompt, allowed_special={"<|endoftext|>"})
    generated = []
    for _ in range(max_new_tokens):
        window = ids[-context_length:]
        x = torch.tensor(window, dtype=torch.long, device=device).unsqueeze(0)
        logits = model(x)[0, -1, :].float()
        if temperature <= 0:
            next_id = int(torch.argmax(logits).item())
        else:
            next_id = int(torch.multinomial(top_p_filter(logits, temperature, top_p), 1).item())
        if next_id == eot_id:
            break
        ids.append(next_id)
        generated.append(next_id)
    return enc.decode(generated)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="The same JSON config the run trained with.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--samples-per-prompt", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--prompt", action="append", dest="prompts",
                   help="Repeatable; replaces the built-in prompt set.")
    args = p.parse_args()

    cfg = json.load(open(args.config))
    m = cfg["model"]
    device = torch.device(args.device)

    model = TransformerLM(
        vocab_size=m["vocab_size"],
        context_length=m["context_length"],
        num_layers=m["num_layers"],
        d_model=m["d_model"],
        heads_num=m["num_heads"],
        d_ff=m["d_ff"],
        theta=m.get("rope_theta", 10_000.0),
        device=str(device),
        dtype=torch.float32,
    )
    ckpt = torch.load(args.checkpoint, map_location=device)
    # Training runs with compile=true, and torch.compile wraps the model in an
    # OptimizedModule whose state_dict prefixes every key with "_orig_mod." -- so a
    # checkpoint from a compiled run will not load into a plain TransformerLM.
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"[sample] loaded {args.checkpoint} (iter={ckpt.get('iter')}) onto {device}")

    enc = tiktoken.get_encoding("gpt2")
    eot_id = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

    for prompt in (args.prompts or DEFAULT_PROMPTS):
        print("\n" + "=" * 70)
        print(f"PROMPT: {prompt!r}")
        print("=" * 70)
        for i in range(args.samples_per_prompt):
            out = generate(model, enc, prompt, args.max_new_tokens, args.temperature,
                           args.top_p, m["context_length"], device, eot_id)
            print(f"--- sample {i} ---\n{prompt}{out}")


if __name__ == "__main__":
    main()
