"""Combinatorial Purged Cross-Validation (Lopez de Prado AFML Ch.12.4)."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np


@dataclass
class CombinatorialPurgedKFold:
    """CPCV splitter for time-series strategies with overlapping labels.

    Combines combinatorial cross-validation with purge and embargo. Each
    test set is the union of n_test_splits contiguous groups chosen from
    n_splits total groups, yielding C(n_splits, n_test_splits) splits.
    Training observations whose label end time t1[i] falls inside any
    test group are purged. An embargo of embargo_pct * n observations
    after each test block is also removed from the training set.

    The number of reconstructible backtest paths equals
    C(n_splits - 1, n_test_splits - 1) per Lopez de Prado AFML Ch.12.4.

    Parameters
    ----------
    n_splits : total number of contiguous partitions of the series.
    n_test_splits : number of groups used as test in each combination,
        must be strictly less than n_splits.
    embargo_pct : fraction of total observations to embargo after each
        test block (default 0.0, meaning no embargo applied).

    Notes
    -----
    The purge implemented here follows the simplification of dropping
    training observations whose label end time falls inside a test
    group; full AFML purging also considers observations whose label
    start falls inside a test group, which is left to a future version.
    """

    n_splits: int
    n_test_splits: int
    embargo_pct: float = 0.0

    def __post_init__(self) -> None:
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if self.n_test_splits < 1:
            raise ValueError("n_test_splits must be at least 1")
        if self.n_test_splits >= self.n_splits:
            raise ValueError("n_test_splits must be less than n_splits")
        if not 0.0 <= self.embargo_pct < 1.0:
            raise ValueError("embargo_pct must be in [0, 1)")

    @property
    def n_splits_total(self) -> int:
        """Total number of train/test combinations produced by split()."""
        return math.comb(self.n_splits, self.n_test_splits)

    @property
    def paths(self) -> int:
        """Number of reconstructible backtest paths."""
        return math.comb(self.n_splits - 1, self.n_test_splits - 1)

    def split(self, n: int,
              t1: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield (train_idx, test_idx) ndarrays for each combination.

        Parameters
        ----------
        n : total number of observations.
        t1 : ndarray of shape (n,). For each observation i, t1[i] is the
            integer index at which observation i's label window ends.
            Use t1 = np.arange(1, n + 1) for label windows of length 1.
        """
        if n < self.n_splits:
            raise ValueError(
                f"n ({n}) must be at least n_splits ({self.n_splits})"
            )
        t1_arr = np.asarray(t1, dtype=int).reshape(-1)
        if t1_arr.size != n:
            raise ValueError(
                f"t1 must have length {n}, got {t1_arr.size}"
            )

        boundaries = np.linspace(0, n, self.n_splits + 1, dtype=int)
        groups = [
            (int(boundaries[i]), int(boundaries[i + 1]))
            for i in range(self.n_splits)
        ]

        embargo_size = int(math.floor(self.embargo_pct * n))
        all_indices = np.arange(n)

        for test_combo in itertools.combinations(range(self.n_splits),
                                                 self.n_test_splits):
            test_mask = np.zeros(n, dtype=bool)
            for g in test_combo:
                start, end = groups[g]
                test_mask[start:end] = True

            embargo_mask = np.zeros(n, dtype=bool)
            if embargo_size > 0:
                for g in test_combo:
                    _, end = groups[g]
                    stop = min(end + embargo_size, n)
                    embargo_mask[end:stop] = True

            purge_mask = np.zeros(n, dtype=bool)
            for g in test_combo:
                start, end = groups[g]
                purge_mask |= (t1_arr >= start) & (t1_arr < end)

            train_mask = ~test_mask & ~embargo_mask & ~purge_mask
            yield all_indices[train_mask], all_indices[test_mask]
