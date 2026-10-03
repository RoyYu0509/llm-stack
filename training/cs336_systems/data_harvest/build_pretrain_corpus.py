"""Stream one Hugging Face text dataset into tiktoken-gpt2 token files for pretraining.

Output (in --out-dir):
    {prefix}_train.npy     uint16, exactly --train-tokens tokens (skipped if 0)
    {prefix}_valid.npy     uint16, exactly --val-tokens tokens
    {prefix}_manifest.json what was built: source, seed, doc/token counts, dump histogram, timing

Format decisions (identical for train, valid and every corpus, so losses are comparable):
  * each document = encode_ordinary(text) + [EOT=50256]; no other separators;
  * uint16 is enough for the 50257-token gpt2 vocabulary and halves the file vs int32;
  * the doc that crosses the token budget is truncated, so file length == budget exactly.

Held-out validation:
  * default: the FIRST --val-tokens of the shuffled stream go to valid, everything after to
    train, so the two are disjoint by construction (no document in both);
  * --val-split NAME: valid comes from that official split instead (e.g. C4 "validation").

Shuffle: the stream is read through `IterableDataset.shuffle(seed, buffer_size)`. This matters:
FineWeb-Edu sample-10BT is stored in dump order (its first 3000 docs are all CC-MAIN-2013-20),
so taking a prefix without shuffling would train on a single 2013 crawl.

Build on the GPU instance, not the laptop: it needs a few GB of download and many CPU threads.

Examples:
    uv run python cs336_systems/data_harvest/build_pretrain_corpus.py \\
        --dataset HuggingFaceFW/fineweb-edu --config sample-10BT \\
        --train-tokens 800000000 --val-tokens 5000000 --out-dir data/pretrain190m --prefix fwe
    uv run python cs336_systems/data_harvest/build_pretrain_corpus.py \\
        --dataset allenai/c4 --config en --val-split validation \\
        --train-tokens 800000000 --val-tokens 5000000 --out-dir data/pretrain190m --prefix c4
    uv run python cs336_systems/data_harvest/build_pretrain_corpus.py \\
        --dataset wikimedia/wikipedia --config 20231101.en \\
        --train-tokens 0 --val-tokens 3000000 --out-dir data/pretrain190m --prefix wiki
"""

import argparse
import collections
import json
import os
import time

import numpy as np
import tiktoken
from datasets import load_dataset

EOT = 50256


def _stream(dataset: str, config: str | None, split: str, seed: int, buffer: int):
    ds = load_dataset(dataset, config, split=split, streaming=True)
    return iter(ds.shuffle(seed=seed, buffer_size=buffer))


def _fill(path: str, n_tokens: int, docs, enc, batch_docs: int, threads: int, label: str) -> dict:
    """Pull documents from `docs` until exactly n_tokens are written to a uint16 .npy at path."""
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint16, shape=(n_tokens,))
    pos, n_docs, truncated = 0, 0, False
    dumps = collections.Counter()
    t0, next_report = time.time(), 0
    while pos < n_tokens:
        batch = []
        for rec in docs:
            batch.append(rec)
            if len(batch) >= batch_docs:
                break
        if not batch:
            break  # stream exhausted
        encoded = enc.encode_ordinary_batch([r["text"] for r in batch], num_threads=threads)
        for rec, ids in zip(batch, encoded):
            ids.append(EOT)
            take = min(len(ids), n_tokens - pos)
            out[pos:pos + take] = ids[:take]
            pos += take
            n_docs += 1
            if "dump" in rec:
                dumps[rec["dump"]] += 1
            if take < len(ids):
                truncated = True
            if pos >= n_tokens:
                break
        if pos >= next_report:
            rate = pos / max(time.time() - t0, 1e-9)
            print(f"[{label}] {pos:,}/{n_tokens:,} tokens  {n_docs:,} docs  {rate / 1e6:.2f}M tok/s", flush=True)
            next_report = pos + 20_000_000
    out.flush()
    del out
    if pos < n_tokens:
        raise RuntimeError(f"[{label}] stream ran out at {pos:,} tokens (< {n_tokens:,}); lower the budget")
    return dict(path=path, tokens=pos, docs=n_docs, mean_tokens_per_doc=round(pos / max(n_docs, 1), 1),
                last_doc_truncated=truncated, seconds=round(time.time() - t0, 1),
                dumps=dict(dumps.most_common()) if dumps else None)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--val-split", default=None, help="take valid from this split instead of a held-out prefix")
    p.add_argument("--train-tokens", type=int, required=True)
    p.add_argument("--val-tokens", type=int, required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shuffle-buffer", type=int, default=10_000)
    p.add_argument("--batch-docs", type=int, default=1_000)
    p.add_argument("--threads", type=int, default=os.cpu_count() or 8)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    assert enc.n_vocab == 50257 and enc.eot_token == EOT
    manifest = dict(dataset=args.dataset, config=args.config, split=args.split, val_split=args.val_split,
                    seed=args.seed, shuffle_buffer=args.shuffle_buffer, tokenizer="tiktoken gpt2",
                    tiktoken_version=tiktoken.__version__, eot=EOT, dtype="uint16",
                    doc_format="encode_ordinary(text) + [EOT]")

    main_docs = _stream(args.dataset, args.config, args.split, args.seed, args.shuffle_buffer)
    val_docs = (_stream(args.dataset, args.config, args.val_split, args.seed, args.shuffle_buffer)
                if args.val_split else main_docs)  # same iterator => valid is the held-out prefix

    if args.val_tokens > 0:
        manifest["valid"] = _fill(os.path.join(args.out_dir, f"{args.prefix}_valid.npy"), args.val_tokens,
                                  val_docs, enc, args.batch_docs, args.threads, f"{args.prefix}/valid")
    if args.train_tokens > 0:
        manifest["train"] = _fill(os.path.join(args.out_dir, f"{args.prefix}_train.npy"), args.train_tokens,
                                  main_docs, enc, args.batch_docs, args.threads, f"{args.prefix}/train")

    mpath = os.path.join(args.out_dir, f"{args.prefix}_manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("train", "valid")}, indent=1))
    for part in ("valid", "train"):
        if part in manifest:
            m = manifest[part]
            print(f"{part}: {m['tokens']:,} tokens / {m['docs']:,} docs "
                  f"({m['mean_tokens_per_doc']} tok/doc) in {m['seconds']}s -> {m['path']}")
    print(f"manifest -> {mpath}", flush=True)


if __name__ == "__main__":
    main()
    # Everything is flushed and closed above. datasets' streaming readers can leave background
    # HTTP retry threads alive that keep the interpreter from exiting for many minutes (seen on
    # 2026-09-30), so exit hard instead of waiting for them.
    os._exit(0)
