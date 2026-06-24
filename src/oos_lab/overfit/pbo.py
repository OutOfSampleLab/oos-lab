"""Probability of Backtest Overfitting via CSCV (Bailey et al. 2014).

Reference
---------
Bailey, D. H., Borwein, J., Lopez de Prado, M., & Zhu, Q. J. (2014).
The Probability of Backtest Overfitting. Journal of Computational
Finance, 20(4), 39-70.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class PBOResult:
    """Output of a Combinatorially Symmetric Cross-Validation analysis.

    Attributes
    ----------
    pbo : probability of backtest overfitting, fraction of CSCV
        partitions where the in-sample best strategy underperforms the
        out-of-sample median (logit < 0).
    logits : per-combination relative-rank logits.
    in_sample_sharpes : per-combination in-sample Sharpe of the selected
        best variant.
    out_of_sample_sharpes : per-combination out-of-sample Sharpe of the
        in-sample best variant.
    performance_degradation_slope : slope of the OLS line regressing
        out-of-sample on in-sample Sharpes across combinations. Values
        substantially below 1 indicate degradation under selection.
    performance_degradation_intercept : intercept of the same regression.
    """

    pbo: float
    logits: np.ndarray
    in_sample_sharpes: np.ndarray
    out_of_sample_sharpes: np.ndarray
    performance_degradation_slope: float
    performance_degradation_intercept: float


def _per_variant_sharpe(returns: np.ndarray) -> np.ndarray:
    """Per-column sample Sharpe of a returns matrix (ddof=1)."""
    mu = returns.mean(axis=0)
    sigma = returns.std(axis=0, ddof=1)
    sigma_safe = np.where(sigma > 0, sigma, np.inf)
    return mu / sigma_safe


def probability_of_backtest_overfit(returns_matrix: np.ndarray,
                                    n_partitions: int = 16) -> PBOResult:
    """Compute the Probability of Backtest Overfitting via CSCV.

    Partitions the time axis into n_partitions equal blocks, enumerates
    every C(n_partitions, n_partitions / 2) way to assign half the
    blocks to in-sample. For each split, selects the variant with the
    highest in-sample Sharpe and measures its out-of-sample rank among
    all variants. PBO is the fraction of splits where the selected
    variant ranks below the out-of-sample median.

    Parameters
    ----------
    returns_matrix : ndarray of shape (n_obs, n_variants). Each column
        is the return time series of one backtest variant.
    n_partitions : even integer at least 4. Number of equal-length
        blocks the time axis is partitioned into. C(n_partitions,
        n_partitions / 2) splits are evaluated.

    Returns
    -------
    PBOResult including the PBO scalar, per-split logits, selected
    variant in and out of sample Sharpes, and OLS degradation
    coefficients.
    """
    if returns_matrix.ndim != 2:
        raise ValueError("returns_matrix must be 2-D (n_obs, n_variants)")
    if n_partitions < 4 or n_partitions % 2 != 0:
        raise ValueError("n_partitions must be even and at least 4")

    n_obs, n_variants = returns_matrix.shape
    if n_obs < n_partitions:
        raise ValueError(
            f"n_obs ({n_obs}) must be at least n_partitions ({n_partitions})"
        )
    if n_variants < 2:
        raise ValueError("need at least 2 variants for PBO analysis")

    boundaries = np.linspace(0, n_obs, n_partitions + 1, dtype=int)
    submatrices = [
        returns_matrix[boundaries[i]:boundaries[i + 1]]
        for i in range(n_partitions)
    ]

    half = n_partitions // 2
    combos = list(itertools.combinations(range(n_partitions), half))
    n_combos = len(combos)

    logits = np.zeros(n_combos)
    is_picks = np.zeros(n_combos)
    oos_picks = np.zeros(n_combos)

    for k, in_groups in enumerate(combos):
        out_groups = tuple(i for i in range(n_partitions) if i not in in_groups)
        in_sample = np.vstack([submatrices[i] for i in in_groups])
        out_sample = np.vstack([submatrices[i] for i in out_groups])

        in_sharpes = _per_variant_sharpe(in_sample)
        out_sharpes = _per_variant_sharpe(out_sample)

        best_in = int(np.argmax(in_sharpes))
        out_for_best = float(out_sharpes[best_in])
        rank = float((out_sharpes <= out_for_best).sum()) / n_variants
        rank = min(max(rank, 1.0 / (n_variants + 1)),
                   n_variants / (n_variants + 1))

        logits[k] = math.log(rank / (1.0 - rank))
        is_picks[k] = float(in_sharpes[best_in])
        oos_picks[k] = out_for_best

    pbo = float((logits < 0).mean())

    if n_combos > 1:
        x = is_picks - is_picks.mean()
        y = oos_picks - oos_picks.mean()
        var_x = float((x * x).sum())
        if var_x > 0:
            slope = float((x * y).sum() / var_x)
            intercept = float(oos_picks.mean() - slope * is_picks.mean())
        else:
            slope = 0.0
            intercept = float(oos_picks.mean())
    else:
        slope = 0.0
        intercept = 0.0

    return PBOResult(
        pbo=pbo,
        logits=logits,
        in_sample_sharpes=is_picks,
        out_of_sample_sharpes=oos_picks,
        performance_degradation_slope=slope,
        performance_degradation_intercept=intercept,
    )
