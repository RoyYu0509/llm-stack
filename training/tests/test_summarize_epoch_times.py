"""summarize_epoch_times.py: the one hop between raw per-epoch times and the numbers on screen.

Every assertion message names the component that is wrong, because a silently wrong number
here ends up narrated in a video.
"""

import importlib.util
import json
import statistics as st
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "cs336_systems" / "experiments" / "summarize_epoch_times.py"
_spec = importlib.util.spec_from_file_location("summarize_epoch_times", SCRIPT)
sm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sm)


def row(kernel, wrapper, times, *, status="ok", samples=64, gpus=1):
    return {
        "kernel": kernel, "ddp": wrapper, "gpus": gpus, "status": status,
        "epoch_times_s": times, "samples_per_epoch": samples, "steps_per_epoch": 16,
        "context_length": 2048, "local_batch_size": 2, "tokens_per_sec": 1000.0, "peak_gpu_mb": 1.0,
        "meta": {"gpu_name": "fake"},
    }


NAIVE, FLASH = sm.NAIVE_ATTN, sm.FLASH_ATTN


def test_mean_and_std_are_sample_statistics():
    times = [10.0, 10.5, 9.5, 10.2, 9.8]
    c = sm.config_stats(row(NAIVE, sm.LOCAL, times))
    assert c["mean_s"] == pytest.approx(st.mean(times)), "config_stats: mean_s is not the arithmetic mean"
    assert c["std_s"] == pytest.approx(st.stdev(times)), \
        "config_stats: std_s must be the SAMPLE std (n-1); pstdev under-states the error bars in the video"
    assert c["cv_pct"] == pytest.approx(100 * st.stdev(times) / st.mean(times))


def test_failed_config_is_reported_as_failed_not_as_a_number():
    c = sm.config_stats(row(NAIVE, sm.NAIVE_DDP, [], status="oom"))
    assert c["status"] == "oom"
    assert "mean_s" not in c, "config_stats: an OOM config must not carry a mean (a 0 would look like a result)"


def test_single_epoch_cannot_give_a_std():
    c = sm.config_stats(row(NAIVE, sm.LOCAL, [10.0]))
    assert c["status"] == "too_few_epochs" and "std_s" not in c, \
        "config_stats: one epoch has no variance; must not report std"


def test_comparison_with_failed_side_is_unavailable_and_does_not_crash():
    rows = [row(NAIVE, sm.LOCAL, [10, 10.1, 9.9]), row(NAIVE, sm.NAIVE_DDP, [], status="oom")]
    facts = sm.summarize(rows)
    cmp_ = next(c for c in facts["comparisons"] if c["name"] == "ddp_naive_vs_single_gpu")
    assert "speedup" not in cmp_ and cmp_["unavailable"] == {"b": "oom"}, \
        "summarize: a comparison involving an OOM config must be 'unavailable', not a speedup"


def test_mixed_samples_per_epoch_is_refused():
    rows = [row(NAIVE, sm.LOCAL, [10, 10.1, 9.9], samples=64),
            row(NAIVE, sm.NAIVE_DDP, [6, 6.1, 5.9], samples=128, gpus=2)]
    with pytest.raises(SystemExit):
        sm.summarize(rows)  # epochs of different data sizes are not comparable


def test_verdict_resolved_vs_within_noise():
    a = sm.config_stats(row(NAIVE, sm.LOCAL, [10.0, 10.1, 9.9, 10.0, 10.05]))
    b_fast = sm.config_stats(row(NAIVE, sm.NAIVE_DDP, [6.0, 6.1, 5.9, 6.0, 6.05]))
    b_same = sm.config_stats(row(NAIVE, sm.NAIVE_DDP, [10.3, 9.6, 10.4, 9.7, 10.0]))
    fast, same = sm.compare(a, b_fast), sm.compare(a, b_same)
    assert fast["verdict"] == "resolved" and fast["speedup"] == pytest.approx(10.01 / 6.01, rel=1e-2), \
        "compare: a 40% faster config with tiny spread must be 'resolved'"
    assert same["verdict"] == "within_noise", \
        "compare: a 0.5% mean difference under +-0.4s spread must be 'within_noise' -- the video may not claim it"


def test_json_round_trip_and_cli(tmp_path):
    for r in (row(NAIVE, sm.LOCAL, [10, 10.1, 9.9]), row(FLASH, sm.LOCAL, [8, 8.1, 7.9])):
        (tmp_path / f"epoch_times__{r['kernel']}__{r['ddp'].replace(' ', '_')}.json").write_text(json.dumps(r))
    facts = sm.summarize(sm.load_rows(str(tmp_path)))
    cmp_ = next(c for c in facts["comparisons"] if c["name"] == "flash_vs_naive_attn_single_gpu")
    assert cmp_["speedup"] == pytest.approx(10.0 / 8.0), "summarize: speedup must be mean_naive / mean_flash"
    assert facts["samples_per_epoch"] == 64
