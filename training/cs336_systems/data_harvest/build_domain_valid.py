"""Build per-domain validation streams (web / wikipedia / instruction).

The pretraining corpus mixes three domains at 70/20/10, but the validation set is a single
concatenated stream -- so a single val loss cannot say whether the mixture ratio actually
did anything per domain. That leaves the central "LLaMA-ratio-inspired data mixing" claim
with no evidence attached to it.

This writes one .npy per domain, drawn from the same sources and held-out slices that
build_mixed_corpus.py used for its validation split, so a checkpoint can be scored on each
domain separately and the mixture's effect becomes a table instead of an assertion.

Streams are deliberately small and equal-sized: they are used for *comparison between
checkpoints* on identical batches, not for an absolute quality number, so a few hundred
thousand tokens each is plenty and keeps scoring to well under a minute per checkpoint.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset

from cs336_systems.data_harvest.build_mixed_corpus import _load_alpaca_pools

OUT_DIR = Path("data/tokenized")
EOT = "<|endoftext|>"
# Matches build_mixed_corpus.py's validation slices: C4 has a real `validation` split, and
# wikipedia (train-only) is skipped past the region the training corpus consumed.
WIKI_VALID_SKIP = 200_000


def _encode(enc, docs: list[str]) -> np.ndarray:
    text = EOT.join(docs) + EOT
    return np.asarray(enc.encode(text, allowed_special={EOT}), dtype=np.int32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target_chars", type=int, default=1_200_000,
                   help="Per-domain character budget (equal across domains on purpose).")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    t0 = time.time()

    def take(stream, field: str) -> list[str]:
        docs, chars = [], 0
        for row in stream:
            text = (row.get(field) or "").strip()
            if not text:
                continue
            docs.append(text)
            chars += len(text)
            if chars >= args.target_chars:
                break
        return docs

    print("[domain-valid] web (allenai/c4 validation split) ...")
    web = take(load_dataset("allenai/c4", "en", split="validation", streaming=True), "text")

    print("[domain-valid] wikipedia (train split, skipped past the training region) ...")
    wiki_stream = load_dataset(
        "wikimedia/wikipedia", "20231101.en", split="train", streaming=True
    ).skip(WIKI_VALID_SKIP)
    wiki = take(wiki_stream, "text")

    print("[domain-valid] instruction (alpaca held-out pool) ...")
    _, instr_pool = _load_alpaca_pools()
    instr, chars = [], 0
    for doc in instr_pool:
        instr.append(doc)
        chars += len(doc)
        if chars >= args.target_chars:
            break

    for name, docs in [("web", web), ("wiki", wiki), ("instruction", instr)]:
        arr = _encode(enc, docs)
        out = OUT_DIR / f"domain_valid_{name}.npy"
        np.save(str(out), arr)
        print(f"[domain-valid] {name}: {len(docs):,} docs -> {len(arr):,} tokens -> {out}")

    print(f"[domain-valid] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
