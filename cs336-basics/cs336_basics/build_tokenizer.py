import argparse
import pickle
import random
from pathlib import Path

from cs336_basics.bpe_tokenizer.tokenizer import Tokenizer


def train_tokenizer(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 3,
    split_special_token: bytes = b"<|endoftext|>",
):
    bpe_tk = Tokenizer(vocab=vocab_size, special_tokens=special_tokens)
    vocab, merges = bpe_tk.train(
        input_path,
        vocab_size,
        num_processes,
        split_special_token,
    )
    return vocab, merges

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and serialize a BPE tokenizer.")
    parser.add_argument("--input", type=Path, required=True, help="Path to the raw text corpus.")
    parser.add_argument("--vocab-size", type=int, required=True, help="Maximum tokenizer vocab size.")
    parser.add_argument(
        "--special-tokens",
        type=str,
        default="<|endoftext|>",
        help="Comma separated list of special tokens (default: <|endoftext|>).",
    )
    parser.add_argument("--num-processes", type=int, default=3, help="Parallel workers for BPE training.")
    parser.add_argument(
        "--vocab-output",
        type=Path,
        default=Path("vocab_id2b_dict.pkl"),
        help="Output path for vocab pickle.",
    )
    parser.add_argument(
        "--merges-output",
        type=Path,
        default=Path("merges_seq.pkl"),
        help="Output path for merges pickle.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    special_tokens = [tok.strip() for tok in args.special_tokens.split(",") if tok.strip()]
    vocab, merges = train_tokenizer(
        input_path=str(args.input),
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        num_processes=args.num_processes,
        split_special_token=special_tokens[0].encode("utf-8") if special_tokens else b"",
    )

    args.vocab_output.parent.mkdir(parents=True, exist_ok=True)
    args.merges_output.parent.mkdir(parents=True, exist_ok=True)

    with args.vocab_output.open("wb") as f:
        pickle.dump(vocab, f)
    with args.merges_output.open("wb") as f:
        pickle.dump(merges, f)

    print(f"Saved vocab to {args.vocab_output}")
    print(f"Saved merges to {args.merges_output}")


if __name__ == "__main__":
    main()
