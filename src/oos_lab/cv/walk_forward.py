"""Walk-forward cross-validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np


@dataclass
class WalkForward:
    """Walk-forward index splitter for time-series data.

    Parameters
    ----------
    train_size : initial training window length in samples.
    test_size  : test window length in samples.
    step       : how far to advance after each split (default test_size,
                 which yields non-overlapping test windows).
    anchored   : if True the training window is always anchored at index
                 0 and grows over time; if False the window is fixed-size
                 and slides forward.
    """

    train_size: int
    test_size: int
    step: int = 0
    anchored: bool = False

    def __post_init__(self) -> None:
        if self.train_size < 1:
            raise ValueError("train_size must be positive")
        if self.test_size < 1:
            raise ValueError("test_size must be positive")
        if self.step <= 0:
            self.step = self.test_size

    def split(self, n: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield (train_idx, test_idx) tuples for a series of length n."""
        if n < self.train_size + self.test_size:
            raise ValueError("n must accommodate at least train_size + test_size")
        t = self.train_size
        while t + self.test_size <= n:
            train_start = 0 if self.anchored else t - self.train_size
            train_idx = np.arange(train_start, t)
            test_idx = np.arange(t, t + self.test_size)
            yield train_idx, test_idx
            t += self.step

    def n_splits(self, n: int) -> int:
        """Number of splits this configuration produces for length n."""
        if n < self.train_size + self.test_size:
            return 0
        return 1 + (n - self.train_size - self.test_size) // self.step
