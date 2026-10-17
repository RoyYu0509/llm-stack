import argparse
import os

import numpy as np
import tiktoken

EOT = 50256

# 1. Stream texts from a dataset, 并且进行 shuffle
def stream_texts(dataset, config, split, seed, buffer):
    """
    返回的是一个 generator, 每次 next() 会返回一条记录的 text 字段, 也就是一篇网页的文本内容。
    
    Args:
        dataset: 是一个 huggingface datasets 的 dataset name, 例如 "allenai/c4"
        config: 是 dataset 的 config name, 例如 "en"
        split: 是 dataset 的 split name, 例如 "train" 或 "validation"
        seed: 是一个 int, 用来控制 shuffle 的随机种子
        buffer: 是一个 int, 用来控制 shuffle 的 buffer size, 越大越随机, 但是越占用内存

    Yields:
        rec[“text”]: 是一个 str, 也就是一篇网页的文本内容。

    Variables:
        rec: 自己是一篇网页,一个 dict
            eg.    {'text': 'The Independent Jane\nFor all the love, romance and scandal in Jane Austen’s book...',
                    'id': '<urn:uuid:0d8a...>',
                    'dump': 'CC-MAIN-2013-20',        # 爬取批次 = 哪一年哪一周爬的
                    'url': 'http://austenauthors.net/the-independent-jane',
                    'language': 'en', 'language_score': 0.97,
                    'token_count': 845,
                    'score': 2.75, 'int_score': 3}    # 教育价值打分
    """
    from datasets import load_dataset
    # 创建一个 iterable 用来 stream 数据集, 并且进行 shuffle
    ds = load_dataset(dataset, config, split=split, streaming=True)
    for rec in ds.shuffle(seed=seed, buffer_size=buffer):
        yield rec["text"] # yield 每一条记录的 text 字段, 作为一个 generator

def fill(path, n_tokens, texts, enc):
    """
    enc: 是一个 tiktoken.Encoding object
    """
    # 用 numpy memmap, 这样数据还在 disk 上, 内存里面只会有一小部分, 方便处理大数据集
    # 并且用 np.uint16 dtype 来表示所有的 token ids, 因为 tiktoken 的 vocab size < 65536, 所以 uint16 足够了
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint16, shape=(n_tokens,))
    pos = n_docs = 0
    for text in texts:
        ids = enc.encode_ordinary(text) + [EOT] # 用 encoder 将 str → list[int]
        take = min(len(ids), n_tokens - pos)
        out[pos:pos + take] = ids[:take]
        pos += take
        n_docs += 1
        if n_docs % 5000 == 0:
            print(f"{path}: {pos:,}/{n_tokens:,} tokens", flush=True)
        if pos >= n_tokens:
            break
    else:
        raise RuntimeError(f"stream ended at {pos:,} < {n_tokens:,} tokens")
    out.flush()
    return n_docs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--val-split", default=None)
    p.add_argument("--train-tokens", type=int, required=True)
    p.add_argument("--val-tokens", type=int, required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shuffle-buffer", type=int, default=10_000)
    a = p.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    train_texts = stream_texts(a.dataset, a.config, a.split, a.seed, a.shuffle_buffer)
    val_texts = (stream_texts(a.dataset, a.config, a.val_split, a.seed, a.shuffle_buffer)
                 if a.val_split else train_texts)
    if a.val_tokens > 0:
        n = fill(f"{a.out_dir}/{a.prefix}_valid.npy", a.val_tokens, val_texts, enc)
        print(f"valid: {a.val_tokens:,} tokens, {n:,} docs")
    if a.train_tokens > 0:
        n = fill(f"{a.out_dir}/{a.prefix}_train.npy", a.train_tokens, train_texts, enc)
        print(f"train: {a.train_tokens:,} tokens, {n:,} docs")


if __name__ == "__main__":
    main()
    os._exit(0)
