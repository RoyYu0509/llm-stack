"""Evaluate a checkpoint's loss/perplexity on one or more tokenized streams.

The training loop only ever evaluates against the single validation set wired into its
config. That is not enough to answer the question the fine-tuning arms exist to answer:
fine-tuning on instruction data is expected to improve instruction loss and degrade
general-language loss, and you only see the trade-off if you score the same checkpoint
on both streams. This runs a checkpoint against any number of .npy token streams and
prints one table.

Usage:
    uv run python cs336_systems/experiments/eval_checkpoint.py \
        --config cs336_systems/experiments/real_pretrain_config.json \
        --checkpoint checkpoints_pretrain_v1/checkpoint_epoch_0100.pt \
        --stream general=data/tokenized/mixed_valid.npy \
        --stream instruction=data/tokenized/alpaca_valid.npy
"""

import argparse
import json
import math

import torch
from torch.utils.data import DataLoader

from cs336_basics.lm import TransformerLM
from cs336_basics.train.loss import cross_entropy
from cs336_systems.Parallelization.DDP.stream_dataset import TokenStreamDataset


@torch.no_grad()
def eval_stream(model, path: str, context_length: int, batch_size: int, n_batches: int,
                device, seed: int) -> tuple[float, int]:
    ds = TokenStreamDataset(path, context_length)
    # Fixed-seed shuffle: the streams have very different lengths, so scoring the first
    # N windows of each would compare "start of C4" against "start of alpaca" rather
    # than a representative sample of either.
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g, num_workers=2)
    total, seen = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        total += cross_entropy(model(x), y).item()
        seen += 1
        if seen >= n_batches:
            break
    return total / max(seen, 1), seen


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--stream", action="append", required=True,
                   help="Repeatable, as name=path/to/tokens.npy")
    p.add_argument("--batches", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
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
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.to(device).eval()

    results = {"checkpoint": args.checkpoint, "iter": ckpt.get("iter"), "streams": {}}
    print(f"\n=== {args.checkpoint} (iter={ckpt.get('iter')}) ===")
    print(f"{'stream':<16}{'loss':>10}{'ppl':>12}{'batches':>10}")
    for spec in args.stream:
        name, _, path = spec.partition("=")
        loss, seen = eval_stream(model, path, m["context_length"], args.batch_size,
                                 args.batches, device, args.seed)
        ppl = math.exp(min(loss, 20))  # cap so a diverged checkpoint prints instead of overflowing
        results["streams"][name] = {"path": path, "loss": loss, "ppl": ppl, "batches": seen}
        print(f"{name:<16}{loss:>10.4f}{ppl:>12.2f}{seen:>10}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[eval] wrote {args.json_out}")


if __name__ == "__main__":
    main()
