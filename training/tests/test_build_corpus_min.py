"""Acceptance tests for the hand-written corpus builder (2026-10-03).

Target: cs336_systems/data_harvest/build_pretrain_corpus.py
Required interface:
    fill(path, n_tokens, texts, enc) -> int     # writes tokens of `texts` into a uint16 .npy; returns docs consumed
    CLI flags used by scripts_190m/run_arm.sh   # see test_cli_flags
Each failure message names the broken part and what to re-read.
"""

import importlib.util
import pathlib
import subprocess
import sys

import numpy as np
import pytest
import tiktoken

# Load by file path (not `import cs336_systems...`): the package __init__ needs the installed
# "llm-training" metadata, which would force a full torch/triton sync just to test a data script.
SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "cs336_systems/data_harvest/build_pretrain_corpus.py"
_spec = importlib.util.spec_from_file_location("build_pretrain_corpus", SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
fill = _mod.fill

EOT = 50256
ENC = tiktoken.get_encoding("gpt2")


def _docs(n):
    return [f"document number {i} says hello world {i * 7}" for i in range(n)]


def _expected(docs):
    out = []
    for d in docs:
        out += ENC.encode_ordinary(d) + [EOT]
    return out


def test_dtype_and_exact_length(tmp_path):
    docs = _docs(50)
    n = 200
    p = str(tmp_path / "a.npy")
    fill(p, n, iter(docs), ENC)
    arr = np.load(p)
    assert arr.dtype == np.uint16, f"dtype is {arr.dtype}: 词表 50257 < 65536,必须 uint16(int32 会让文件翻倍)。回看:token 文件格式"
    assert arr.shape == (n,), f"length is {arr.shape}, must be exactly n_tokens={n}(跨过预算的那篇要截断,不能多写也不能少写)"


def test_every_doc_ends_with_eot(tmp_path):
    docs = _docs(10)
    exp = _expected(docs)
    p = str(tmp_path / "b.npy")
    fill(p, len(exp), iter(docs), ENC)  # budget == whole stream: no truncation
    assert np.load(p).tolist() == exp, "文件内容 != 逐篇 encode_ordinary(text)+[50256] 拼接。检查:每篇文档后是否追加 EOT、篇与篇之间有无多余分隔符"


def test_truncates_last_doc(tmp_path):
    docs = _docs(10)
    exp = _expected(docs)
    n = len(exp) - 3
    p = str(tmp_path / "c.npy")
    fill(p, n, iter(docs), ENC)
    assert np.load(p).tolist() == exp[:n], "预算在最后一篇中间用完时,应截断到恰好 n_tokens(前缀必须和完整流一致)"


def test_stream_too_short_raises(tmp_path):
    docs = _docs(3)
    with pytest.raises(RuntimeError):
        fill(str(tmp_path / "d.npy"), len(_expected(docs)) + 1, iter(docs), ENC)
    # 流提前耗尽却静默返回,文件尾部会是一串 0 —— 训练时 token 0 ('!') 会被当成真数据


def test_literal_special_token_in_text_is_plain_text(tmp_path):
    docs = ["before <|endoftext|> after", "second doc"]
    exp = _expected(docs)
    p = str(tmp_path / "e.npy")
    fill(p, len(exp), iter(docs), ENC)
    got = np.load(p).tolist()
    assert got == exp and got.count(EOT) == 2, (
        "网页文本里的字面量 <|endoftext|> 必须当普通文本切成多个 token;"
        "用 enc.encode 会报错或把它变成真 EOT,应该用 encode_ordinary。回看:A1 BPE 里 special token 的处理")


def test_val_and_train_share_one_stream_disjoint_docs(tmp_path):
    """main() 把同一个迭代器先给 val 再给 train;fill 不能多吞、也不能回头重读。"""
    docs = _docs(200)
    it = iter(docs)
    v, t = str(tmp_path / "v.npy"), str(tmp_path / "t.npy")
    fill(v, 300, it, ENC)
    fill(t, 300, it, ENC)

    def split_docs(path):
        text = ENC.decode(np.load(path).tolist())
        return [s for s in text.split("<|endoftext|>") if s]

    vd, td = split_docs(v), split_docs(t)
    # drop the possibly-truncated last piece of each file before comparing
    assert not (set(vd[:-1]) & set(td[:-1])), "同一篇文档同时出现在 val 和 train:验证集泄漏。val 和 train 必须从同一个迭代器顺序取,不能各开一个新迭代器"
    first_train_doc = td[0]
    assert first_train_doc not in vd, "train 的第一篇和 val 重复:fill 结束后不应把已经消费的文档放回迭代器"


def test_cli_flags():
    r = subprocess.run([sys.executable, str(SCRIPT), "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"--help 失败(顶层 import 了装不上的东西?load_dataset 应该放在用到它的函数里):\n{r.stderr[-500:]}"
    for flag in ["--dataset", "--config", "--split", "--val-split", "--train-tokens", "--val-tokens",
                 "--out-dir", "--prefix", "--seed", "--shuffle-buffer"]:
        assert flag in r.stdout, f"缺少命令行参数 {flag}:scripts_190m/run_arm.sh 会用到"
