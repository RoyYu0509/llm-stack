"""bucket_plan tests (CPU). They pin the facts the README / guide rely on."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
pytest.importorskip("cs336_systems.Parallelization.FlashDDP.FlashDDP", reason="needs the repo venv")
import bucket_plan as bp  # noqa: E402

SHAPES = bp.param_shapes_from_config(50257, 768, 16, 2048)   # the 190M configuration


def test_total_gradient_bytes_is_the_190m_model():
    total_mb = sum(x["size_mb"] for x in bp.plan(SHAPES, 25.0))
    n = sum(a * (b if len(s) == 2 else 1) for _, s in SHAPES for a, b in [(s[0], s[1] if len(s) == 2 else 1)])
    assert abs(total_mb - n * 4 / 1e6) < 1e-6, "buckets must cover every parameter exactly once"
    assert 760 < total_mb < 765, f"190M fp32 gradients should be ~762 MB, got {total_mb}"


def test_every_parameter_is_in_exactly_one_bucket():
    plan = bp.plan(SHAPES, 25.0)
    assert sum(x["n_params"] for x in plan) == len(SHAPES)


def test_head_and_embedding_are_single_tensor_buckets_bigger_than_threshold():
    plan = bp.plan(SHAPES, 25.0)
    assert plan[0]["first"] == "head.weightMat" and plan[0]["n_params"] == 1, (
        "reverse order starts at the output head; it alone exceeds 25 MB so it closes its bucket"
    )
    assert plan[0]["size_mb"] > 150
    assert plan[-1]["last"] == "in_embedding.embed_mat" and plan[-1]["size_mb"] > 150, (
        "the embedding gradient is ready LAST -> its bucket cannot overlap with any remaining compute"
    )


def test_bucket_count_matches_threshold_semantics():
    # a bucket closes when the running size is >= threshold, so a bigger threshold -> fewer buckets
    counts = [len(bp.plan(SHAPES, mb)) for mb in (1, 5, 25, 100)]
    assert counts == sorted(counts, reverse=True), counts
    assert counts[0] > counts[-1]
