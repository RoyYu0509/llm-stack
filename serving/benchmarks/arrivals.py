"""Request arrival schedules for OPEN-loop load tests. Standard library only.

A closed loop (C users, each sends the next request when its last one is answered) can never overload a server:
the faster the server, the faster the load. An open loop sends requests on a clock that does not care how the
server is doing, so a server that is too slow builds a queue and its TTFT grows -- which is what the load test is
for. Every system in a comparison must get the SAME schedule (same rate, cv, seed, n), so prompt i arrives at
the same instant for the engine, SGLang, vLLM and HF.

"""
from __future__ import annotations

import math
import random
from typing import List


def arrival_times(rate: float, n: int, seed: int = 0, cv: float = 1.0) -> List[float]:
    """Arrival offsets (seconds from the start of the run) of `n` requests, in increasing order.

    The gaps between requests are Gamma(shape=1/cv^2, scale=cv^2/rate): mean 1/rate, coefficient of variation `cv`.
    cv = 1 is exponential gaps, i.e. a Poisson process (the usual model of independent users); cv > 1 is burstier
    (requests arrive in clumps with long silences); cv < 1 is more regular than Poisson.
    """
    if rate <= 0 or cv <= 0 or n < 0:
        raise ValueError(f"need rate > 0, cv > 0, n >= 0 (got rate={rate}, cv={cv}, n={n})")
    rng = random.Random(seed)
    shape, scale = 1.0 / (cv * cv), cv * cv / rate
    t, out = 0.0, []
    for _ in range(n):
        t += rng.gammavariate(shape, scale)
        out.append(t)
    return out


def n_for_duration(rate: float, duration_s: float) -> int:
    """How many requests a schedule of `duration_s` seconds at `rate` req/s contains."""
    return max(1, math.ceil(rate * duration_s))
