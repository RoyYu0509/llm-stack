"""CPU checks for the pieces added for the 190M rerun (2026-10-01).

Each failure message names the piece that is wrong:
  * TokenStreamDataset(stride=...)      -> cs336_systems/Parallelization/DDP/stream_dataset.py
  * _prune_old_checkpoints ordering     -> cs336_systems/Parallelization/FlashDDP/FlashDDP_runner.py
  * eval_downstream.score_item          -> cs336_systems/experiments/eval_downstream.py
"""

import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from cs336_systems.Parallelization.DDP.stream_dataset import TokenStreamDataset


@pytest.fixture
def stream(tmp_path):
    path = tmp_path / "toks.npy"
    np.save(path, np.arange(1000, dtype=np.uint16))
    return str(path)


def test_stride1_is_old_sliding_window(stream):
    ds = TokenStreamDataset(stream, 16)
    assert len(ds) == 1000 - 16 - 1, "stride=1 must keep the old len() (N - ctx - 1)"
    x, y = ds[5]
    assert x.tolist() == list(range(5, 21)) and y.tolist() == list(range(6, 22)), "stride=1 sample i must start at i"


def test_stride_ctx_gives_disjoint_chunks_covering_the_stream(stream):
    ctx = 16
    ds = TokenStreamDataset(stream, ctx, stride=ctx)
    assert len(ds) == (1000 - ctx - 1) // ctx + 1, "stride=ctx: one sample per start 0, ctx, 2ctx, ... with room for y"
    starts = [ds[i][0][0].item() for i in range(len(ds))]
    assert starts == [i * ctx for i in range(len(ds))], "stride=ctx: sample i must start at i*ctx (non-overlapping)"
    x, y = ds[len(ds) - 1]
    assert y[-1].item() <= 999, "last sample's target runs past the end of the stream"
    assert x.dtype == torch.long and y.dtype == torch.long, "uint16 tokens must come out as int64 for the embedding"


def test_prune_keeps_numerically_newest(tmp_path):
    from cs336_systems.Parallelization.FlashDDP.FlashDDP_runner import _prune_old_checkpoints

    for step in (5000, 7500, 10000, 12500):
        (tmp_path / f"checkpoint_step_{step:04d}.pt").write_bytes(b"x")
    (tmp_path / "checkpoint_epoch_0001.pt").write_bytes(b"x")
    _prune_old_checkpoints(str(tmp_path), "step", keep_last=2)
    left = sorted(os.listdir(tmp_path))
    assert left == ["checkpoint_epoch_0001.pt", "checkpoint_step_10000.pt", "checkpoint_step_12500.pt"], (
        f"pruning must keep the 2 highest STEP numbers (string sort keeps step_7500); got {left}")


def test_score_item_matches_unpadded_logprobs():
    import tiktoken
    from cs336_basics.lm import TransformerLM
    from cs336_systems.experiments.eval_downstream import score_item

    torch.manual_seed(0)
    model = TransformerLM(vocab_size=50257, context_length=64, num_layers=2, d_model=32, heads_num=4,
                          d_ff=64, theta=10000.0, device="cpu", dtype=torch.float32).eval()
    enc = tiktoken.get_encoding("gpt2")
    context, choices = "Question: What is water made of?\nAnswer:", ["hydrogen and oxygen", "sand", "a very long wrong answer here"]
    got = score_item(model, enc, context, choices, 64, torch.device("cpu"))
    for ch, g in zip(choices, got):
        c, e = enc.encode_ordinary(context), enc.encode_ordinary(" " + ch)
        ids = torch.tensor([c + e])
        with torch.no_grad():
            logp = F.log_softmax(model(ids[:, :-1]), dim=-1)[0]
        want = sum(logp[len(c) - 1 + j, e[j]].item() for j in range(len(e)))
        assert abs(g - want) < 1e-3, (
            f"score_item({ch!r}) = {g:.4f} but unpadded sum log p = {want:.4f}: padding or the "
            "continuation positions are off by one")
