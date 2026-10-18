"""How does the self-written DDP actually bucket the 190M model? (CPU only, no process group.)

Reuses the real `Bucket` class from FlashDDP.py and replays the loop of
`DDPOverlapBucketed._build_bucket` (reverse parameter order; close a bucket once its running size
>= threshold; the tensor that crosses the threshold stays in that bucket). Parameters are
`meta` tensors built from the checkpoint's shapes, so nothing is allocated.

Two facts this makes visible, both relevant to "why is scaling efficiency 8x% and not 95%":

  1. A single tensor can exceed the threshold: the output head (V x d) and the embedding (V x d)
     are ~154 MB each in fp32. The head closes a bucket by itself; the embedding lands in the
     final bucket together with the layer-0 tensors (~164 MB at a 25 MB threshold).
  2. Gradients become ready from the output layer to the input layer, so the LAST bucket to be
     ready is the one holding the embedding -- there is no remaining backward compute to hide its
     all-reduce behind.

Units: FlashDDP's `_get_tensor_size` divides by 1e6 (decimal MB). torch DDP's `bucket_cap_mb` is
MiB (bucket_bytes_cap = int(mb * 1024 * 1024)) and its *first* bucket is capped at 1 MiB
(`torch.distributed._DEFAULT_FIRST_BUCKET_BYTES`). "25" is therefore not the same byte count.

Usage:
  python bucket_plan.py --ckpt ../../../training/ckpt_190m_fwe/model_only.pt --bucket-mb 25
  python bucket_plan.py --shapes-from-config 50257 768 16 2048 --bucket-mb 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def param_shapes_from_ckpt(path: str) -> list[tuple[str, tuple[int, ...]]]:
    obj = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    sd = obj["model"] if "model" in obj else obj
    # state_dict order == parameter registration order (buffers such as token_positions are skipped)
    return [(k, tuple(v.shape)) for k, v in sd.items()
            if v.dtype.is_floating_point and k != "token_positions"]


def param_shapes_from_config(vocab: int, d: int, layers: int, d_ff: int) -> list[tuple[str, tuple[int, ...]]]:
    out = [("in_embedding.embed_mat", (vocab, d))]
    for i in range(layers):
        p = f"tf_layers.{i}."
        out += [(p + "RMSN1.gain", (d,)), (p + "MHA.W_Q", (d, d)), (p + "MHA.W_K", (d, d)),
                (p + "MHA.W_V", (d, d)), (p + "MHA.W_O", (d, d)), (p + "RMSN2.gain", (d,)),
                (p + "FNN.W1", (d_ff, d)), (p + "FNN.W3", (d_ff, d)), (p + "FNN.W2", (d, d_ff))]
    out += [("norm.gain", (d,)), ("head.weightMat", (vocab, d))]
    return out


def plan(named_shapes, bucket_mb: float, dtype=torch.float32):
    """Return a list of buckets: dict(idx, size_mb, n_params, first, last, biggest_mb)."""
    sys.path.insert(0, str(REPO / "training"))
    sys.path.insert(0, str(REPO / "training" / "cs336-basics"))
    from cs336_systems.Parallelization.FlashDDP.FlashDDP import Bucket  # the real class

    names = {}
    paras = []
    for n, s in named_shapes:
        p = torch.nn.Parameter(torch.empty(s, dtype=dtype, device="meta"))
        names[id(p)] = n
        paras.append(p)

    buckets, cur = [], Bucket(bucket_mb, offsets=0)
    for p in reversed(paras):                       # same order as _build_bucket
        cur.add_tensor(p)
        if cur.is_full():
            buckets.append(cur)
            cur = Bucket(bucket_mb, offsets=0)
    if cur.para_list:
        buckets.append(cur)

    out = []
    for i, b in enumerate(buckets):
        sizes = [q.numel() * q.element_size() / 1e6 for q in b.para_list]
        out.append(dict(idx=i, size_mb=sum(sizes), n_params=len(b.para_list),
                        first=names[id(b.para_list[0])], last=names[id(b.para_list[-1])],
                        biggest_mb=max(sizes)))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt")
    src.add_argument("--shapes-from-config", nargs=4, type=int, metavar=("VOCAB", "D", "LAYERS", "D_FF"))
    ap.add_argument("--bucket-mb", type=float, default=25.0)
    ap.add_argument("--busbw-gbps", type=float, default=None,
                    help="if given, print the ideal all-reduce time of each bucket at this bus bandwidth (GB/s)")
    args = ap.parse_args(argv)

    shapes = (param_shapes_from_ckpt(args.ckpt) if args.ckpt
              else param_shapes_from_config(*args.shapes_from_config))
    b = plan(shapes, args.bucket_mb)
    total = sum(x["size_mb"] for x in b)
    print(f"{len(shapes)} parameter tensors, {total:.1f} MB (fp32 grads), threshold {args.bucket_mb} MB "
          f"-> {len(b)} buckets  (ready order: 0 = output side first, {len(b) - 1} = embedding last)")
    for x in b:
        t = "" if not args.busbw_gbps else f"  ideal allreduce {1e3 * x['size_mb'] / 1e3 / args.busbw_gbps:6.1f} ms"
        print(f"  #{x['idx']:2d} {x['size_mb']:7.1f} MB  {x['n_params']:2d} tensors  "
              f"{x['first']} .. {x['last']}{t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
