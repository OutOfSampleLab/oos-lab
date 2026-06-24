"""Multiple-testing haircut on Sharpe ratios (Harvey & Liu 2015).

Reference
---------
Harvey, C. R., & Liu, Y. (2015). Backtesting. Journal of Portfolio
Management, 42(1), 13-28.

Implements Holm-Bonferroni (FWER) and Benjamini-Hochberg-Yekutieli
(FDR under arbitrary dependence) adjustments, then converts adjusted
p-values back to a haircut Sharpe ratio for each tested strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats


@dataclass
class HaircutResult:
    """Holm or BHY adjusted Sharpes and survival mask.

    Attributes
    ----------
    raw_sharpes : original per-strategy Sharpe estimates.
    haircut_sharpes : Sharpe back-implied from the adjusted p-value at
        the same n_obs, representing the corrected significance.
    raw_pvalues : one-sided p-values for the null SR = 0.
    adjusted_pvalues : multiple-testing-adjusted p-values.
    survives_at_alpha : boolean mask of which strategies survive at the
        given alpha after correction.
    method : "holm" or "bhy".
    """

    raw_sharpes: np.ndarray
    haircut_sharpes: np.ndarray
    raw_pvalues: np.ndarray
    adjusted_pvalues: np.ndarray
    survives_at_alpha: np.ndarray
    method: str


def _sharpe_to_pvalue(sharpe: float, n_obs: int) -> float:
    """One-sided p-value for the null SR = 0 under normality."""
    t = sharpe * np.sqrt(n_obs)
    return float(1.0 - stats.norm.cdf(t))


def _pvalue_to_sharpe(pvalue: float, n_obs: int) -> float:
    """Invert: given a p-value at n_obs, return the implied Sharpe."""
    if pvalue >= 1.0:
        return 0.0
    if pvalue <= 0.0:
        return float("inf")
    t = float(stats.norm.ppf(1.0 - pvalue))
    return t / float(np.sqrt(n_obs))


def haircut_sharpe(sharpes,
                   n_obs: int,
                   method: Literal["holm", "bhy"] = "holm",
                   alpha: float = 0.05) -> HaircutResult:
    """Apply Harvey-Liu multiple-testing haircut to a list of Sharpes.

    Parameters
    ----------
    sharpes : sequence of Sharpe ratio estimates. Each is treated as
        coming from an independent strategy backtest of length n_obs.
        Sharpe units must match n_obs (per-period if n_obs is the
        number of periods).
    n_obs : number of observations used to estimate each Sharpe.
    method : "holm" for Holm-Bonferroni control of family-wise error
        rate, "bhy" for Benjamini-Hochberg-Yekutieli control of false
        discovery rate under arbitrary dependence.
    alpha : significance level for the survives_at_alpha mask.

    Returns
    -------
    HaircutResult with raw and haircut Sharpes, raw and adjusted
    p-values, and a survival mask under the chosen alpha.
    """
    if method not in ("holm", "bhy"):
        raise ValueError("method must be 'holm' or 'bhy'")
    if n_obs < 2:
        raise ValueError("n_obs must be at least 2")

    arr = np.asarray(sharpes, dtype=float).reshape(-1)
    k = arr.size
    if k == 0:
        raise ValueError("sharpes must not be empty")

    pvalues = np.array([_sharpe_to_pvalue(float(s), n_obs) for s in arr])
    order = np.argsort(pvalues)
    sorted_p = pvalues[order]

    if method == "holm":
        weights = np.arange(k, 0, -1, dtype=float)
        sorted_adj = sorted_p * weights
        sorted_adj = np.maximum.accumulate(sorted_adj)
    else:
        c_k = float((1.0 / np.arange(1, k + 1)).sum())
        raw_adj = sorted_p * k * c_k / np.arange(1, k + 1, dtype=float)
        sorted_adj = np.minimum.accumulate(raw_adj[::-1])[::-1]

    sorted_adj = np.clip(sorted_adj, 0.0, 1.0)
    adjusted = np.empty_like(sorted_adj)
    adjusted[order] = sorted_adj
    adjusted = np.maximum(adjusted, pvalues)

    haircut = np.array([_pvalue_to_sharpe(float(p), n_obs) for p in adjusted])
    haircut = np.minimum(haircut, arr)
    survives = adjusted < alpha

    return HaircutResult(
        raw_sharpes=arr,
        haircut_sharpes=haircut,
        raw_pvalues=pvalues,
        adjusted_pvalues=adjusted,
        survives_at_alpha=survives,
        method=method,
    )
