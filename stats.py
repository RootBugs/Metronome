"""Sample statistics and Welch's unequal-variance t-test.

Welch's t-test compares two sample means without assuming equal variances:

    t  = (mean_a - mean_b) / sqrt(var_a/n_a + var_b/n_b)
    df = (s_a + s_b)^2 / (s_a^2/(n_a-1) + s_b^2/(n_b-1))     [Welch–Satterthwaite]

where ``s_x = var_x / n_x`` and variances are *sample* variances (n-1
denominator). Degenerate cases are defined explicitly: zero variance with
equal means gives t = 0; zero variance with different means gives an
infinite t with the sign of the difference.
"""

from __future__ import annotations

import math
from typing import NamedTuple

__all__ = ["mean", "stddev", "WelchResult", "welch_t_test"]


class WelchResult(NamedTuple):
    """Outcome of :func:`welch_t_test`: the t statistic and its degrees of
    freedom."""

    t: float
    df: float


def mean(xs) -> float:
    """Arithmetic mean of *xs* (raises ``ValueError`` when empty)."""
    if not xs:
        raise ValueError("mean of an empty sample")
    return sum(xs) / len(xs)


def stddev(xs) -> float:
    """Sample standard deviation (n-1 denominator).

    Returns ``0.0`` for fewer than two observations.
    """
    n = len(xs)
    if n < 2:
        return 0.0
    m = mean(xs)
    variance = sum((x - m) ** 2 for x in xs) / (n - 1)
    return math.sqrt(variance)


def welch_t_test(a, b) -> WelchResult:
    """Welch's t-test between independent samples *a* and *b*.

    Each group needs at least two observations. See module docstring for the
    exact formulas and degenerate-case definitions.
    """
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        raise ValueError("welch_t_test needs >= 2 samples in each group")
    ma, mb = mean(a), mean(b)
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    sa, sb = va / na, vb / nb  # squared standard errors
    se2 = sa + sb
    if se2 == 0.0:
        # No variance anywhere: t is 0 for equal means, infinite otherwise.
        if ma == mb:
            return WelchResult(0.0, float("inf"))
        return WelchResult(math.copysign(float("inf"), ma - mb), float("inf"))
    t = (ma - mb) / math.sqrt(se2)
    denom = (sa * sa) / (na - 1) + (sb * sb) / (nb - 1)
    df = (se2 * se2) / denom if denom > 0 else float("inf")
    return WelchResult(t, df)
