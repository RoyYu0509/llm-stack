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


# def train_encoder():
#     ECHO = True
#     special_tokens = ['<|endoftext|>', ]
#     max_vocab_size = 10000
#     fp = "/Users/yifanyu/Desktop/CS336 LLM/CS336 A1/data/TinyStoriesV2-GPT4-train.txt"
#     num_processes = 10
#     sp_tok = "<|endoftext|>".encode("utf-8")

#     bpe_tk = Tokenizer(vocab=max_vocab_size, merges=None, special_tokens=special_tokens)
#     vocab_id2b_dict, merges_seq = bpe_tk.train(fp, max_vocab_size,
#                                  num_processes, sp_tok)

#     # Save vocab and merges
#     with open("vocab_id2b_dict.pkl", "wb") as f:
#         pickle.dump(vocab_id2b_dict, f)

#     with open("merges_seq.pkl", "wb") as f:
#         pickle.dump(merges_seq, f)


def small_vocab_size_encode_decode_sample():
    with open("vocab_id2b_dict.pkl", "rb") as f:
        vocab_id2b_dict = pickle.load(f)

    with open("merges_seq.pkl", "rb") as f:
        merges_seq = pickle.load(f)

    # Create tokenizer
    special_tokens = ['<|endoftext|>', ]
    bpe_tk = Tokenizer(vocab_id2b_dict, merges_seq, special_tokens)

    # Sample a few lines from text corp

    def sample_lines(filename, k=5):
        samples = []
        with open(filename, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                if i <= k:
                    samples.append(line)
                else:
                    j = random.randint(1, i)
                    if j <= k:
                        samples[j - 1] = line
        return [s.strip() for s in samples]

    samples = sample_lines("/Users/yifanyu/Desktop/CS336 LLM/CS336 A1/data/TinyStoriesV2-GPT4-valid.txt", k=5)
    for i, s in enumerate(samples, 1):
        print(f"--- Raw Sample {i} ---\n{s}\n")

        encoded = bpe_tk.encode(s)
        decoded = bpe_tk.decode(encoded)

        # --- Compute statistics ---
        n_bytes = len(s.encode("utf-8"))  # raw text bytes
        n_tokens = len(encoded)  # number of BPE tokens
        ratio = n_bytes / n_tokens if n_tokens > 0 else float("inf")

        print(f"Encoded IDs: {encoded}")
        print(f"Decoded txt: {decoded}\n")
        print(f"Bytes: {n_bytes}, Tokens: {n_tokens}, Bytes/Token: {ratio:.2f}\n")


def large_vocab_size_encode_decode_sample():
    with open("vocab_id2b_dict.pkl", "rb") as f:
        vocab_id2b_dict = pickle.load(f)

    with open("merges_seq.pkl", "rb") as f:
        merges_seq = pickle.load(f)

    # Create tokenizer
    special_tokens = ['<|endoftext|>', ]
    bpe_tk = Tokenizer(vocab_id2b_dict, merges_seq, special_tokens)

    # Sample a few lines from text corp

    def sample_lines(filename, k=5):
        samples = []
        with open(filename, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                if i <= k:
                    samples.append(line)
                else:
                    j = random.randint(1, i)
                    if j <= k:
                        samples[j - 1] = line
        return [s.strip() for s in samples]

    samples = sample_lines("/Users/yifanyu/Desktop/CS336 LLM/CS336 A1/data/TinyStoriesV2-GPT4-valid.txt", k=5)
    for i, s in enumerate(samples, 1):
        print(f"--- Raw Sample {i} ---\n{s}\n")

        encoded = bpe_tk.encode(s)
        decoded = bpe_tk.decode(encoded)

        # --- Compute statistics ---
        n_bytes = len(s.encode("utf-8"))  # raw text bytes
        n_tokens = len(encoded)  # number of BPE tokens
        ratio = n_bytes / n_tokens if n_tokens > 0 else float("inf")

        print(f"Encoded IDs: {encoded}")
        print(f"Decoded txt: {decoded}\n")
        print(f"Bytes: {n_bytes}, Tokens: {n_tokens}, Bytes/Token: {ratio:.2f}\n")


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
