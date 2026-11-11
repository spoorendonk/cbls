"""Small statistics the benchmark reports share, written out rather than pulled
from scipy: the reports' dependencies are the repository's."""

from __future__ import annotations

import math

#: Two-sided 95% Student multipliers (the 97.5% t quantile) by degrees of
#: freedom.
_T975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    15: 2.131,
    20: 2.086,
    30: 2.042,
    40: 2.021,
    60: 2.000,
}


def t975(df: int) -> float:
    """The 97.5% t quantile at `df`, NaN below 1.

    Off the tabulated points, and past the last one, the next LOWER df's value is
    used: it is the wider band, so a marginal difference is not called a result.
    The normal 1.96 is never returned; it is narrower than t at every finite df.
    """
    if df < 1:
        return math.nan
    return _T975[max(k for k in _T975 if k <= df)]


def sign_test_p(a: int, b: int) -> float:
    """Exact two-sided sign test p-value for `a` against `b` wins under p = 1/2;
    ties are excluded by the caller. Symmetric in its arguments."""
    n = a + b
    if n == 0:
        return 1.0
    # Integer true division is correctly rounded at any n; a float 2**n
    # overflows past n = 1023.
    tail = sum(math.comb(n, k) for k in range(min(a, b) + 1)) / (1 << n)
    return min(1.0, 2.0 * tail)
