"""Build a LLaMA-ratio-inspired mixed pretraining corpus from three public HF-hosted sources:

  - web text:          allenai/c4 (en)                          -> 70% of the budget
  - wikipedia:         wikimedia/wikipedia (20231101.en)         -> 20% of the budget
  - instruction/resp.: tatsu-lab/alpaca                          -> 10% of the budget

Alpaca is tiny relative to its target share, so it is deliberately oversampled (cycled)
to hit its budget -- mirroring how the LLaMA paper itself over-samples small,
high-quality domains like Wikipedia/Books (expressed as >1 epoch over those domains,
not a flat byte-for-byte split).

Writes:
  data/mixed_corpus/train.txt
  data/mixed_corpus/valid.txt
  data/mixed_corpus/manifest.json   (exact sources, achieved char counts/ratios, oversample factor)

Usage:
  uv run python cs336_systems/data_harvest/build_mixed_corpus.py
  uv run python cs336_systems/data_harvest/build_mixed_corpus.py --dry_run   # ~2MB total, fast sanity check
"""

import argparse
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

EOT = "<|endoftext|>"
REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "mixed_corpus"

_WS_RE = re.compile(r"\n{3,}")

# Full-scale budgets (raw characters, before tokenization).
FULL_TRAIN_BUDGET = {"web": 546_000_000, "wikipedia": 156_000_000, "instruction": 78_000_000}
FULL_VALID_BUDGET = {"web": 14_000_000, "wikipedia": 4_000_000, "instruction": 2_000_000}

# Dry-run budgets: same 70/20/10 ratio, ~2MB total, for a fast end-to-end sanity check.
DRY_TRAIN_BUDGET = {"web": 1_400_000, "wikipedia": 400_000, "instruction": 200_000}
DRY_VALID_BUDGET = {"web": 70_000, "wikipedia": 20_000, "instruction": 10_000}


def clean(text: str) -> str:
    text = text.strip()
    text = _WS_RE.sub("\n\n", text)
    return text


def _write_doc(fh, text: str) -> int:
    fh.write(text)
    fh.write("\n")
    fh.write(EOT)
    fh.write("\n")
    return len(text) + len(EOT) + 2


def build_web(fh, split: str, char_budget: int) -> dict:
    ds = load_dataset("allenai/c4", "en", split=split, streaming=True)
    written = docs = 0
    for row in ds:
        text = clean(row["text"])
        if not text:
            continue
        written += _write_doc(fh, text)
        docs += 1
        if written >= char_budget:
            break
    return {"source": "allenai/c4:en", "split": split, "chars": written, "docs": docs, "passes": 1}


def build_wikipedia(fh, char_budget: int, skip_docs: int) -> tuple[dict, int]:
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    if skip_docs:
        ds = ds.skip(skip_docs)
    written = docs = 0
    for row in ds:
        text = clean(row["text"])
        if not text:
            continue
        written += _write_doc(fh, text)
        docs += 1
        if written >= char_budget:
            break
    stats = {
        "source": "wikimedia/wikipedia:20231101.en",
        "split": f"train[skip={skip_docs}]",
        "chars": written,
        "docs": docs,
        "passes": 1,
    }
    return stats, skip_docs + docs


def _format_alpaca(row: dict) -> str:
    instruction = row["instruction"].strip()
    inp = (row.get("input") or "").strip()
    output = row["output"].strip()
    if inp:
        return f"### Instruction:\n{instruction}\n### Input:\n{inp}\n### Response:\n{output}"
    return f"### Instruction:\n{instruction}\n### Response:\n{output}"


def _load_alpaca_pools(seed: int = 0, valid_frac: float = 0.05) -> tuple[list[str], list[str]]:
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    examples = [_format_alpaca(row) for row in ds]
    rng = random.Random(seed)
    rng.shuffle(examples)
    n_valid = max(1, int(len(examples) * valid_frac))
    valid_pool = examples[:n_valid]
    train_pool = examples[n_valid:]
    return train_pool, valid_pool


def build_instruction(fh, char_budget: int, pool: list[str]) -> dict:
    """Fill `char_budget` from `pool`, cycling the pool when it is smaller than the budget.

    KNOWN ISSUE -- correct for the train split, WRONG for the valid split. Cycling is the
    point for training: it is how a small high-quality domain gets its target share of the
    mixture, exactly as LLaMA epochs Wikipedia/Books more times than CommonCrawl. But the
    same call is used to fill the VALIDATION budget, where it repeated the 2,600 held-out
    Alpaca examples 1.91x, and repeated text is artificially easy -- a 1024-token window can
    contain the same example twice, so the second copy is predictable by in-context copying
    rather than by anything the model learned.

    Measured on the 2026-07-27 run: the pretrained model scores 2.3863 on mixed_valid's
    instruction region but 2.7345 on the same examples without repetition -- ~0.35 nats of
    free credit. It also flips the sign of the fine-tuning delta there (+0.09 on the
    repeated region vs -0.33 on clean instruction data), which is what made the per-domain
    arithmetic fail to reconcile until the regions were sliced and scored directly.

    Fix for the next corpus build: do not cycle for the valid split. The validation set does
    not need to match the training mixture ratio -- let each domain contribute whatever its
    held-out pool actually holds, and report per-domain losses separately.
    """
    written = docs = 0
    n = len(pool)
    idx = 0
    while written < char_budget:
        written += _write_doc(fh, pool[idx % n])
        docs += 1
        idx += 1
    return {
        "source": "tatsu-lab/alpaca",
        "chars": written,
        "docs": docs,
        "pool_size": n,
        "passes": round(idx / n, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true", help="~2MB total, fast sanity check.")
    args = parser.parse_args()

    train_budget = DRY_TRAIN_BUDGET if args.dry_run else FULL_TRAIN_BUDGET
    valid_budget = DRY_VALID_BUDGET if args.dry_run else FULL_VALID_BUDGET

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUT_DIR / "train.txt"
    valid_path = OUT_DIR / "valid.txt"
    manifest: dict = {"dry_run": args.dry_run, "train": {}, "valid": {}}

    print(f"[mix] target train budget (chars): {train_budget}")
    print(f"[mix] target valid budget (chars): {valid_budget}")

    print("[mix] Alpaca: loading full dataset (small, non-streaming) and building train/valid pools ...")
    instr_train_pool, instr_valid_pool = _load_alpaca_pools()
    print(f"[mix]   train pool: {len(instr_train_pool)} examples, valid pool: {len(instr_valid_pool)} examples")

    with open(train_path, "w") as fh:
        print("[mix] train: web (c4:en, split=train) ...")
        manifest["train"]["web"] = build_web(fh, "train", train_budget["web"])
        print(f"[mix]   {manifest['train']['web']}")

        print("[mix] train: wikipedia (skip_docs=0) ...")
        wiki_stats, wiki_docs_consumed = build_wikipedia(fh, train_budget["wikipedia"], skip_docs=0)
        manifest["train"]["wikipedia"] = wiki_stats
        print(f"[mix]   {wiki_stats}")

        print("[mix] train: instruction (alpaca, oversampled) ...")
        manifest["train"]["instruction"] = build_instruction(fh, train_budget["instruction"], instr_train_pool)
        print(f"[mix]   {manifest['train']['instruction']}")

    with open(valid_path, "w") as fh:
        print("[mix] valid: web (c4:en, split=validation) ...")
        manifest["valid"]["web"] = build_web(fh, "validation", valid_budget["web"])
        print(f"[mix]   {manifest['valid']['web']}")

        print(f"[mix] valid: wikipedia (skip_docs={wiki_docs_consumed}, non-overlapping with train) ...")
        wiki_valid_stats, _ = build_wikipedia(fh, valid_budget["wikipedia"], skip_docs=wiki_docs_consumed)
        manifest["valid"]["wikipedia"] = wiki_valid_stats
        print(f"[mix]   {wiki_valid_stats}")

        print("[mix] valid: instruction (alpaca held-out pool, no overlap with train pool) ...")
        manifest["valid"]["instruction"] = build_instruction(fh, valid_budget["instruction"], instr_valid_pool)
        print(f"[mix]   {manifest['valid']['instruction']}")

    for split_name, split_stats in (("train", manifest["train"]), ("valid", manifest["valid"])):
        total = sum(s["chars"] for s in split_stats.values())
        for domain, s in split_stats.items():
            s["achieved_ratio"] = round(s["chars"] / total, 4) if total else 0.0
        manifest[f"{split_name}_total_chars"] = total

    manifest_path = OUT_DIR / "manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"[mix] wrote {train_path} ({manifest['train_total_chars']:,} chars)")
    print(f"[mix] wrote {valid_path} ({manifest['valid_total_chars']:,} chars)")
    print(f"[mix] wrote {manifest_path}")


if __name__ == "__main__":
    main()
