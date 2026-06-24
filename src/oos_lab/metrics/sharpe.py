"""Sharpe ratio."""

from __future__ import annotations

import numpy as np

from oos_lab.metrics.returns import to_array


def sharpe_ratio(returns, periods_per_year: int = 1, risk_free: float = 0.0) -> float:
    """Annualised sample Sharpe ratio.

    Sample standard deviation uses ddof=1. Pass periods_per_year=1 to get
    the per-period Sharpe (no annualisation). risk_free is a constant
    subtracted from every return before computing.
    """
    r = to_array(returns) - risk_free
    if r.size < 2:
        raise ValueError("need at least two return observations")
    sigma = r.std(ddof=1)
    if sigma <= 0:
        raise ValueError("returns variance must be positive")
    return float(r.mean() / sigma * np.sqrt(periods_per_year))
