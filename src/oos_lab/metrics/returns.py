"""Return-series utilities."""

from __future__ import annotations

import numpy as np


def to_array(returns) -> np.ndarray:
    """Coerce returns to a 1-D float ndarray, dropping NaNs."""
    arr = np.asarray(returns, dtype=float).reshape(-1)
    return arr[~np.isnan(arr)]


def log_returns(prices) -> np.ndarray:
    """Compute log returns from a price series."""
    p = np.asarray(prices, dtype=float).reshape(-1)
    if p.size < 2:
        raise ValueError("need at least two prices")
    return np.diff(np.log(p))
