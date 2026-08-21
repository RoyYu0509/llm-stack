"""
Utility script to convert raw text files into NumPy arrays of token IDs.

Example:
    uv run python src/train/build_dataset.py \
        --train-text data/TinyStoriesV2-GPT4-train.txt \
        --val-text data/TinyStoriesV2-GPT4-valid.txt \
        --vocab-path src/bpe_tokenizer/vocab_id2b_dict.pkl \
        --merges-path src/bpe_tokenizer/merges_seq.pkl \
        --out-dir data/tokenized
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from cs336_basics.bpe_tokenizer.tokenizer import Tokenizer

_WORKER_TOKENIZER = None  # lazily initialized in worker processes

def _worker_init(vocab_path: str, merges_path: str):
    global _WORKER_TOKENIZER
    if _WORKER_TOKENIZER is None:
        _WORKER_TOKENIZER = Tokenizer.from_files(
            vocab_path,
            merges_path,
            special_tokens=["<|endoftext|>"],
        )


def _encode_chunk_worker(lines: list[str]) -> list[int]:
    assert _WORKER_TOKENIZER is not None, "Tokenizer must be initialized in worker."
    tokens: list[int] = []
    for line in lines:
        tokens.extend(_WORKER_TOKENIZER.encode(line))
    return tokens


def _encode_file(text_path: Path, vocab_path: Path, merges_path: Path, max_size: int | None, num_workers: int) -> np.ndarray:
    """
    Return a 1D tokenized text stream from the `text_path`.

    The function reads row by row from a training text files, and serialized it
    into one gaint token stream.
    """
    if num_workers <= 1:
        tokenizer = _load_tokenizer(vocab_path, merges_path)
        tokens = []
        line_count = 0
        with text_path.open("r", encoding="utf-8") as f:
            for line in f:
                tokens.extend(tokenizer.encode(line))
                line_count += 1
                if max_size is not None and line_count >= max_size:
                    break
                if line_count % 1000 == 0:
                    print(f"Processed {line_count} lines...")
        return np.asarray(tokens, dtype=np.int32)

    chunk_size = 2000  # lines per worker chunk
    tokens: list[int] = []
    line_count = 0
    chunk: list[str] = []

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_worker_init,
        initargs=(str(vocab_path), str(merges_path)),
    ) as pool:
        futures = []
        with text_path.open("r", encoding="utf-8") as f:
            for line in f:
                chunk.append(line)
                line_count += 1
                if len(chunk) == chunk_size:
                    futures.append(pool.submit(_encode_chunk_worker, chunk))
                    chunk = []
                if max_size is not None and line_count >= max_size:
                    break
        if chunk:
            futures.append(pool.submit(_encode_chunk_worker, chunk))

        for future in tqdm(futures, desc=f"Encoding {text_path.name}", unit="chunk"):
            tokens.extend(future.result())

    return np.asarray(tokens, dtype=np.int32)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize a text file into a .npy buffer.")
    parser.add_argument("--size", type=int, required=False, help="Rows of data to process (optional).")
    parser.add_argument("--text-path", type=Path, required=True, help="Path to text file.")
    parser.add_argument("--vocab-path", type=Path, required=True, help="Tokenizer vocab pickle/json.")
    parser.add_argument("--merges-path", type=Path, required=True, help="Tokenizer merges pickle/json.")
    parser.add_argument("--out", type=Path, required=True, help="Output .npy file path.")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel workers to tokenize.")
    return parser.parse_args()


def _load_tokenizer(vocab_path: Path, merges_path: Path, eos_token: str = "<|endoftext|>") -> Tokenizer:
    """Load a tokenizer directly from its serialized files."""
    return Tokenizer.from_files(
        str(vocab_path),
        str(merges_path),
        special_tokens=[eos_token],
    )


def main() -> None:
    args = parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Encoding {args.text_path}...")
    tokens = _encode_file(args.text_path, args.vocab_path, args.merges_path, args.size, args.num_workers)
    np.save(args.out, tokens)
    print(f"Wrote {args.out} ({tokens.shape[0]} tokens)")


if __name__ == "__main__":
    main()
