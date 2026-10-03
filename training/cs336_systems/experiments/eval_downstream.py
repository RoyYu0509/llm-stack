"""Zero-shot multiple-choice accuracy (HellaSwag, ARC-Easy) for a pretraining checkpoint.

Why this exists: two models trained on DIFFERENT corpora cannot be ranked by their own
validation losses -- each val set comes from its own training distribution. A downstream
task that neither corpus was built for is a neutral yardstick (together with a neutral
held-out stream such as Wikipedia, scored with eval_checkpoint.py).

Scoring = the lm-evaluation-harness recipe for these two tasks:
  * each choice is scored as sum log p(" " + choice | context) under the model;
  * acc      = argmax of that sum;
  * acc_norm = argmax of the sum divided by the choice's length in UTF-8 bytes
               (removes the bias toward short answers);
  * HellaSwag context = preprocess(activity_label + ": " + ctx_a + " " + ctx_b.capitalize()),
    ARC context = "Question: {question}\\nAnswer:".
Chance level is 25% for both (a few ARC items have 3 or 5 choices).
At ~0.2B params and <1B tokens expect numbers close to chance; the comparison between
two runs is what matters, with the printed standard error as the noise floor.

Usage:
    uv run python cs336_systems/experiments/eval_downstream.py \\
        --config configs_190m/fwe.json --checkpoint checkpoints_190m_fwe/checkpoint_epoch_0001.pt \\
        --tasks hellaswag,arc_easy --json-out artifacts/190m/fwe_downstream.json
"""

import argparse
import json
import math
import re
import time

import tiktoken
import torch
import torch.nn.functional as F
from datasets import load_dataset

from cs336_basics.lm import TransformerLM


def _hs_pre(text: str) -> str:
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    return text.replace("  ", " ")


def load_task(name: str, limit: int | None):
    """Return a list of (context, [choices], gold_index)."""
    items = []
    if name == "hellaswag":
        for d in load_dataset("Rowan/hellaswag", split="validation"):
            ctx = d["ctx_a"] + " " + d["ctx_b"].capitalize()
            items.append((_hs_pre(d["activity_label"] + ": " + ctx),
                          [_hs_pre(e) for e in d["endings"]], int(d["label"])))
    elif name == "arc_easy":
        for d in load_dataset("allenai/ai2_arc", "ARC-Easy", split="test"):
            labels = d["choices"]["label"]
            items.append((f"Question: {d['question']}\nAnswer:", list(d["choices"]["text"]),
                          labels.index(d["answerKey"])))
    else:
        raise ValueError(f"unknown task {name}")
    return items[:limit] if limit else items


@torch.no_grad()
def score_item(model, enc, context: str, choices: list[str], ctx_len: int, device) -> list[float]:
    """Sum of log-probs of each continuation given the context (one padded batch per item)."""
    c_ids = enc.encode_ordinary(context)
    seqs, spans = [], []
    for ch in choices:
        e_ids = enc.encode_ordinary(" " + ch)
        ids = (c_ids + e_ids)[-(ctx_len + 1):]  # left-truncate; keep the whole continuation
        n_e = min(len(e_ids), len(ids) - 1)
        seqs.append(ids)
        spans.append(n_e)
    L = max(len(s) for s in seqs) - 1
    x = torch.zeros(len(seqs), L, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, : len(s) - 1] = torch.tensor(s[:-1])  # right-pad; padded positions are never read
    logp = F.log_softmax(model(x.to(device)).float(), dim=-1)
    out = []
    for i, (s, n_e) in enumerate(zip(seqs, spans)):
        tgt = torch.tensor(s[-n_e:], device=device)
        pos = torch.arange(len(s) - 1 - n_e, len(s) - 1, device=device)  # logits at t predict t+1
        out.append(logp[i, pos, tgt].sum().item())
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tasks", default="hellaswag,arc_easy")
    p.add_argument("--limit", type=int, default=None, help="first N items per task (smoke tests)")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    cfg = json.load(open(args.config))
    m = cfg["model"]
    device = torch.device(args.device)
    model = TransformerLM(
        vocab_size=m["vocab_size"], context_length=m["context_length"],
        num_layers=m["num_layers"], d_model=m["d_model"], heads_num=m["num_heads"],
        d_ff=m["d_ff"], theta=m.get("rope_theta", 10_000.0),
        device=str(device), dtype=torch.float32,
    )
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()})
    model.to(device).eval()
    enc = tiktoken.get_encoding("gpt2")

    results = {"checkpoint": args.checkpoint, "global_step": ckpt.get("global_step"), "tasks": {}}
    for task in args.tasks.split(","):
        items = load_task(task, args.limit)
        t0 = time.time()
        acc = acc_norm = 0
        for context, choices, gold in items:
            lls = score_item(model, enc, context, choices, m["context_length"], device)
            norm = [ll / max(len((" " + c).encode("utf-8")), 1) for ll, c in zip(lls, choices)]
            acc += int(max(range(len(lls)), key=lls.__getitem__) == gold)
            acc_norm += int(max(range(len(norm)), key=norm.__getitem__) == gold)
        n = len(items)
        r = {"n": n, "acc": acc / n, "acc_norm": acc_norm / n,
             "stderr": math.sqrt((acc_norm / n) * (1 - acc_norm / n) / n), "seconds": round(time.time() - t0, 1)}
        results["tasks"][task] = r
        print(f"{task:<10} n={n:<6} acc={r['acc']:.4f}  acc_norm={r['acc_norm']:.4f}  "
              f"(±{r['stderr']:.4f} 1 s.e.)  {r['seconds']}s", flush=True)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[eval] wrote {args.json_out}")


if __name__ == "__main__":
    main()
