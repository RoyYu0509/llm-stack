"""
Python BPE Tokenizer Benchmark
用法: python benchmark_python.py <input_file> <vocab_size> [num_processes]
"""
import sys
import time
from cs336_basics.bpe_tokenizer.tokenizer import Tokenizer


def run_benchmark(input_file: str, vocab_size: int, num_processes: int):
    special_tokens = ["<|endoftext|>"]
    split_token = "<|endoftext|>".encode("utf-8")

    # --- Training ---
    print(f"  Training with {num_processes} process(es)...")
    tokenizer = Tokenizer(vocab=vocab_size, special_tokens=special_tokens)

    t0 = time.time()
    vocab, merges = tokenizer.train(input_file, vocab_size, num_processes, split_token)
    t1 = time.time()
    train_time = t1 - t0

    # --- Encode / Decode roundtrip ---
    test_texts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "<|endoftext|>Once upon a time there was a little girl<|endoftext|>",
    ]

    all_ok = True
    for text in test_texts:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        if text != decoded:
            all_ok = False
            print(f"  FAIL: '{text}' != '{decoded}'")

    # --- Encode throughput ---
    long_text = "Once upon a time there was a little girl named Lily. She loved to play in the garden. " * 100
    e0 = time.time()
    for _ in range(10):
        tokenizer.encode(long_text)
    e1 = time.time()
    encode_time = e1 - e0
    encode_speed = (10 * len(long_text.encode("utf-8")) / 1024) / encode_time

    print(f"  Encode speed: {encode_speed:.1f} KB/s")

    return {
        "label": f"Python ({num_processes} proc)",
        "train_time": train_time,
        "vocab_size": len(vocab),
        "num_merges": len(merges),
        "correct": all_ok,
    }


def main():
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <input_file> <vocab_size> [num_processes]")
        sys.exit(1)

    input_file = sys.argv[1]
    vocab_size = int(sys.argv[2])
    num_processes = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    print(f"╔══════════════════════════════════════╗")
    print(f"║   Python BPE Benchmark               ║")
    print(f"╠══════════════════════════════════════╣")
    print(f"║ File:       {input_file:<25s}║")
    print(f"║ Vocab size: {vocab_size:<25d}║")
    print(f"╚══════════════════════════════════════╝")

    results = []

    # Serial
    print(f"\n[1/2] Python serial (1 process)...")
    results.append(run_benchmark(input_file, vocab_size, 1))

    # Parallel
    print(f"\n[2/2] Python parallel ({num_processes} processes)...")
    results.append(run_benchmark(input_file, vocab_size, num_processes))

    # Results
    print(f"\n╔══════════════════════════╦════════════╦════════╦════════╦═════════╗")
    print(f"║ Configuration            ║ Train (s)  ║ Vocab  ║ Merges ║ Correct ║")
    print(f"╠══════════════════════════╬════════════╬════════╬════════╬═════════╣")
    for r in results:
        ok = "YES" if r["correct"] else "NO "
        print(f"║ {r['label']:<25s}║ {r['train_time']:>9.3f}s ║ {r['vocab_size']:>5d} ║ {r['num_merges']:>5d}  ║   {ok}   ║")
    print(f"╚══════════════════════════╩════════════╩════════╩════════╩═════════╝")

    # Output for bash script to parse
    with open("/tmp/python_benchmark_results.txt", "w") as f:
        for r in results:
            f.write(f"{r['label']}|{r['train_time']:.3f}|{r['vocab_size']}|{r['num_merges']}|{r['correct']}\n")


if __name__ == "__main__":
    main()
