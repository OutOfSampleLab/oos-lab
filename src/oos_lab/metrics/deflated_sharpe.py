"""Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

The DSR corrects a probabilistic Sharpe statement for the number of
strategy variants that were searched, via the False Strategy Theorem.
A backtest Sharpe that looks great in isolation may simply be the
luckiest of many trials; DSR raises the benchmark accordingly.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from oos_lab.metrics.returns import to_array

_EULER_MASCHERONI = 0.5772156649015329


def expected_max_sharpe(n_trials: int, var_sharpe_across_trials: float) -> float:
    """Expected maximum Sharpe across n_trials trials under the null.

    Closed-form approximation from Bailey & Lopez de Prado (2014),
    valid for n_trials sufficiently large. Returns 0 for n_trials=1
    or for zero variance. Sharpe values are per-period (not annualised).
    """
    if n_trials < 1:
        raise ValueError("n_trials must be positive")
    if var_sharpe_across_trials < 0:
        raise ValueError("var_sharpe_across_trials must be non-negative")
    if n_trials == 1 or var_sharpe_across_trials == 0:
        return 0.0
    sd = math.sqrt(var_sharpe_across_trials)
    inv_n = 1.0 / n_trials
    inv_ne = 1.0 / (n_trials * math.e)
    term = ((1.0 - _EULER_MASCHERONI) * stats.norm.ppf(1.0 - inv_n)
            + _EULER_MASCHERONI * stats.norm.ppf(1.0 - inv_ne))
    return float(sd * term)


def deflated_sharpe_ratio(returns,
                          n_trials: int,
                          var_sharpe_across_trials: float) -> float:
    """Probability the observed Sharpe is real after correcting for trials.

    The benchmark Sharpe is set to expected_max_sharpe(n_trials,
    var_sharpe_across_trials) and then evaluated as a PSR. A value
    close to 1 means the strategy survives multiple-testing correction;
    a value below 0.95 typically does not.
    """
    benchmark = expected_max_sharpe(n_trials, var_sharpe_across_trials)
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
        raise ValueError("non-finite DSR denominator; check return distribution")
    z = (sr - benchmark) * np.sqrt(n - 1) / denom
    return float(stats.norm.cdf(z))
