"""Interactive REPL for talking to the locally-trained model.

Two modes, because the model was only ever trained on one of them:

  * `instruct` (default) -- wraps input in the exact `### Instruction:\\n...\\n### Response:\\n`
    template the Alpaca slice of pretraining and the whole fine-tune used. This is the
    format the model has actually seen.
  * `raw` -- feeds your text in verbatim and continues it. Useful for "the history of the
    internet began in the..." style prompts.

There is deliberately no multi-turn conversation state. The model has never seen a dialogue
transcript, so accumulating turns into a context would be inventing a format it cannot
follow and would mostly produce drift. Each input is an independent request -- which is
also exactly how `LMEngine` treats it.

Uses the KV cache (measured ~1.24x faster than recompute on this model), so it is the same
code path the serving benchmark exercises.

    uv run python cs336_systems/experiments/chat.py \\
        --config configs_sft/lr1e5_replay.json \\
        --checkpoint checkpoints_sft_lr1e5_replay/checkpoint_epoch_0100.pt
"""

import argparse
import json
import sys
import time

import tiktoken
import torch

from cs336_basics.lm import TransformerLM

INSTRUCT_TEMPLATE = "### Instruction:\n{}\n### Response:\n"
BANNER = """
  Local 0.19B model -- trained from scratch on 492M tokens of C4/Wikipedia/Alpaca.
  It is heavily under-trained (2.6 tokens/param vs Chinchilla's ~20), so expect
  fluent-sounding text that is frequently wrong and prone to repeating itself.

  /raw       switch to raw continuation mode (no instruction template)
  /instruct  switch back to instruction mode (default)
  /temp X    sampling temperature (current: {temp})
  /topp X    nucleus threshold (current: {top_p})
  /tokens N  max new tokens (current: {max_new})
  /quit      exit
"""


def load_model(config_path, checkpoint, device):
    cfg = json.load(open(config_path))
    m = cfg["model"]
    model = TransformerLM(
        vocab_size=m["vocab_size"], context_length=m["context_length"],
        num_layers=m["num_layers"], d_model=m["d_model"], heads_num=m["num_heads"],
        d_ff=m["d_ff"], theta=m.get("rope_theta", 10_000.0),
        device=str(device), dtype=torch.float32,
    )
    ckpt = torch.load(checkpoint, map_location=device)
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    return model.to(device).eval(), m


def sample_next(logits, temperature, top_p):
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    probs = torch.softmax(logits / temperature, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    # Keep through the token that crosses top_p so the nucleus is never empty.
    keep = cumulative - sorted_probs < top_p
    keep[0] = True
    idx, p = sorted_idx[keep], sorted_probs[keep]
    return int(idx[torch.multinomial(p / p.sum(), 1)].item())


@torch.no_grad()
def stream_generate(model, enc, prompt, max_new, temperature, top_p, ctx_len, device, eot):
    ids = enc.encode(prompt, allowed_special={"<|endoftext|>"})
    if len(ids) >= ctx_len:
        print(f"[prompt is {len(ids)} tokens, over the {ctx_len} context limit]")
        return 0, 0.0

    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    logits, past = model.forward_with_cache(x, past_kv=None)
    t0 = time.perf_counter()
    n = 0
    pending = []  # buffer bytes until they decode cleanly (multi-token characters)
    for _ in range(max_new):
        nxt = sample_next(logits[0, -1, :].float(), temperature, top_p)
        if nxt == eot:
            break
        pending.append(nxt)
        text = enc.decode(pending)
        if "�" not in text:            # complete character(s) -- safe to print
            print(text, end="", flush=True)
            pending = []
        n += 1
        if len(ids) + n >= ctx_len:
            print("\n[hit context limit]", end="")
            break
        logits, past = model.forward_with_cache(
            torch.tensor([[nxt]], dtype=torch.long, device=device), past_kv=past)
    if pending:
        print(enc.decode(pending), end="", flush=True)
    return n, time.perf_counter() - t0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs_sft/lr1e5_replay.json")
    p.add_argument("--checkpoint",
                   default="checkpoints_sft_lr1e5_replay/checkpoint_epoch_0100.pt")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.92)
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--mode", choices=["instruct", "raw"], default="instruct")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"loading {args.checkpoint} onto {device} ...")
    model, mcfg = load_model(args.config, args.checkpoint, device)
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
    params = sum(t.numel() for t in model.parameters()) / 1e9
    print(f"ready: {params:.2f}B params, context {mcfg['context_length']}, mode={args.mode}")

    temp, top_p, max_new, mode = args.temperature, args.top_p, args.max_new_tokens, args.mode
    print(BANNER.format(temp=temp, top_p=top_p, max_new=max_new))

    while True:
        try:
            line = input("\n\033[1myou >\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith("/"):
            cmd, _, arg = line.partition(" ")
            if cmd in ("/quit", "/exit", "/q"):
                break
            elif cmd == "/raw":
                mode = "raw"; print("mode: raw continuation")
            elif cmd == "/instruct":
                mode = "instruct"; print("mode: instruction")
            elif cmd == "/temp" and arg:
                temp = float(arg); print(f"temperature = {temp}")
            elif cmd == "/topp" and arg:
                top_p = float(arg); print(f"top_p = {top_p}")
            elif cmd == "/tokens" and arg:
                max_new = int(arg); print(f"max_new_tokens = {max_new}")
            else:
                print("commands: /raw /instruct /temp X /topp X /tokens N /quit")
            continue

        prompt = INSTRUCT_TEMPLATE.format(line) if mode == "instruct" else line
        print("\033[90mmodel >\033[0m ", end="", flush=True)
        if mode == "raw":
            print(f"\033[90m{line}\033[0m", end="", flush=True)
        n, secs = stream_generate(model, enc, prompt, max_new, temp, top_p,
                                  mcfg["context_length"], device, eot)
        if n:
            print(f"\n\033[90m[{n} tokens, {n/secs:.1f} tok/s]\033[0m")


if __name__ == "__main__":
    main()
