"""Probabilistic Sharpe Ratio (Bailey & Lopez de Prado, 2012)."""

from __future__ import annotations

import numpy as np
from scipy import stats

from oos_lab.metrics.returns import to_array


def probabilistic_sharpe_ratio(returns, benchmark_sr: float = 0.0) -> float:
    """Probability that the true Sharpe exceeds benchmark_sr.

    Uses the observed sample's Sharpe, length, skewness and kurtosis to
    compute the probability that the underlying true Sharpe is greater
    than benchmark_sr, accounting for non-normality of returns. Sharpe
    values are per-period (not annualised); pass a per-period benchmark.
    """
    r = to_array(returns)
    n = r.size
    if n < 4:
        raise ValueError("need at least four observations")
    mu = r.mean()
    sigma = r.std(ddof=1)
    if sigma <= 0:
        raise ValueError("returns variance must be positive")
    sr = mu / sigma
    skew = float(stats.skew(r, bias=False))
    kurt = float(stats.kurtosis(r, fisher=False, bias=False))
    denom = np.sqrt(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr)
    if denom <= 0:
        raise ValueError("non-finite PSR denominator; check return distribution")
    z = (sr - benchmark_sr) * np.sqrt(n - 1) / denom
    return float(stats.norm.cdf(z))
