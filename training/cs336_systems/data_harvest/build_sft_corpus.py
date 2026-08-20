"""Build the instruction-tuning corpus for the post-pretraining fine-tune.

Two arms, so the fine-tune is a comparison rather than a single run:

  * `alpaca`        -- instruction data only. The straightforward fine-tune.
  * `alpaca_replay` -- instruction data plus a slice of the original pretraining mix
                       (~25% of the token budget), interleaved. This is the standard
                       guard against catastrophic forgetting: fine-tuning a small model
                       on one narrow distribution usually costs a lot of general
                       language ability, and replaying pretraining data is the cheapest
                       way to keep it. Having both arms lets us actually *measure* the
                       forgetting instead of asserting it.

Formatting is imported from build_mixed_corpus so the fine-tune sees byte-identical
prompt structure to what the 10% instruction slice of pretraining already used --
otherwise the model would be adapting to a format change at the same time as the task.

Note on what this is and is not: cs336_basics' cross_entropy has no loss masking, so
loss is computed over the whole `### Instruction: ... ### Response: ...` document, not
only the response tokens. That makes this domain-adaptive continued pretraining on
instruction data, not textbook SFT. Called out here rather than papered over.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import tiktoken

from cs336_systems.data_harvest.build_mixed_corpus import _load_alpaca_pools

OUT_DIR = Path("data/tokenized")
TEXT_DIR = Path("data/sft_corpus")
PRETRAIN_TRAIN_NPY = OUT_DIR / "mixed_train.npy"
EOT = "<|endoftext|>"


def _write_pool(path: Path, pool: list[str]) -> int:
    chars = 0
    with open(path, "w", encoding="utf-8") as f:
        for doc in pool:
            f.write(doc)
            f.write(EOT)
            chars += len(doc) + len(EOT)
    return chars


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay_frac", type=float, default=0.25,
                   help="Pretraining tokens to mix into the replay arm, as a fraction of "
                        "the total replay-arm token count.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")

    t0 = time.time()
    train_pool, valid_pool = _load_alpaca_pools(seed=args.seed)
    print(f"[sft] alpaca pools: {len(train_pool):,} train / {len(valid_pool):,} valid examples")

    train_txt = TEXT_DIR / "alpaca_train.txt"
    valid_txt = TEXT_DIR / "alpaca_valid.txt"
    tr_chars = _write_pool(train_txt, train_pool)
    va_chars = _write_pool(valid_txt, valid_pool)
    print(f"[sft] wrote {tr_chars:,} train chars / {va_chars:,} valid chars")

    encoded = {}
    for name, txt in [("train", train_txt), ("valid", valid_txt)]:
        text = open(txt, encoding="utf-8").read()
        ids = enc.encode(text, allowed_special={EOT})
        arr = np.asarray(ids, dtype=np.int32)
        out = OUT_DIR / f"alpaca_{name}.npy"
        np.save(str(out), arr)
        encoded[name] = arr
        print(f"[sft] {name}: {len(text):,} chars -> {len(arr):,} tokens -> {out}")

    # Replay arm: alpaca tokens + a contiguous slice of the pretraining stream, sized so
    # pretraining data is `replay_frac` of the result. Interleaving happens for free --
    # TokenStreamDataset draws sliding windows and DistributedSampler shuffles them, so
    # concatenating is enough; no need to shuffle documents on disk.
    alpaca_train = encoded["train"]
    n_alpaca = len(alpaca_train)
    n_replay = int(n_alpaca * args.replay_frac / (1 - args.replay_frac))
    pretrain = np.load(str(PRETRAIN_TRAIN_NPY), mmap_mode="r")
    if n_replay > len(pretrain):
        raise SystemExit(f"replay slice {n_replay:,} exceeds pretraining stream {len(pretrain):,}")
    replay = np.concatenate([alpaca_train, np.asarray(pretrain[:n_replay], dtype=np.int32)])
    out = OUT_DIR / "alpaca_replay_train.npy"
    np.save(str(out), replay)
    print(f"[sft] replay arm: {n_alpaca:,} alpaca + {n_replay:,} pretrain "
          f"= {len(replay):,} tokens ({n_replay / len(replay):.1%} pretrain) -> {out}")

    print(f"[sft] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
