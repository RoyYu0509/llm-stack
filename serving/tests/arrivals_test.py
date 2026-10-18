"""arrival_times: the schedule an open-loop load test is only as honest as. Pure Python, no torch."""
import statistics
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from arrivals import arrival_times, n_for_duration  # noqa: E402


def _gaps(ts):
    return [b - a for a, b in zip([0.0] + ts[:-1], ts)]


def test_increasing_and_deterministic_per_seed():
    a = arrival_times(20.0, 500, seed=3)
    assert all(y > x for x, y in zip(a, a[1:])) and a[0] > 0
    assert a == arrival_times(20.0, 500, seed=3), "same seed must give the same schedule for every system"
    assert a != arrival_times(20.0, 500, seed=4)


def test_mean_rate_matches_for_every_burstiness():
    for cv in (0.5, 1.0, 3.0):
        ts = arrival_times(25.0, 20000, seed=1, cv=cv)
        assert 20000 / ts[-1] == pytest.approx(25.0, rel=0.05), f"cv={cv}: the offered load drifted from the requested rate"


def test_cv_is_the_coefficient_of_variation_of_the_gaps():
    for cv in (0.5, 1.0, 3.0):
        g = _gaps(arrival_times(25.0, 20000, seed=2, cv=cv))
        assert statistics.pstdev(g) / statistics.fmean(g) == pytest.approx(cv, rel=0.08), cv


def test_cv1_is_poisson_so_a_second_holds_rate_requests_on_average_with_poisson_spread():
    ts = arrival_times(10.0, 20000, seed=5, cv=1.0)
    per_second = [0] * int(ts[-1])
    for t in ts:
        if int(t) < len(per_second):
            per_second[int(t)] += 1
    assert statistics.fmean(per_second) == pytest.approx(10.0, rel=0.05)
    assert statistics.pvariance(per_second) == pytest.approx(10.0, rel=0.15), "Poisson counts have variance == mean"


def test_bad_arguments_are_errors():
    for kw in (dict(rate=0, n=3), dict(rate=-1, n=3), dict(rate=1, n=3, cv=0), dict(rate=1, n=-1)):
        with pytest.raises(ValueError):
            arrival_times(**kw)
    assert arrival_times(1.0, 0) == []


def test_n_for_duration():
    assert n_for_duration(20, 30) == 600 and n_for_duration(0.1, 1) == 1 and n_for_duration(2.5, 3) == 8
