"""Tests for probabilistic_sharpe_ratio."""

import unittest
import numpy as np

from oos_lab.metrics.psr import probabilistic_sharpe_ratio


class PSRTests(unittest.TestCase):
    def test_strong_strategy_high_psr(self):
        rng = np.random.default_rng(1)
        r = rng.normal(0.005, 0.01, size=500)
        self.assertGreater(probabilistic_sharpe_ratio(r, benchmark_sr=0.0), 0.99)

    def test_zero_mean_psr_in_middle_range(self):
        rng = np.random.default_rng(0)
        r = rng.normal(0.0, 0.01, size=500)
        psr = probabilistic_sharpe_ratio(r, benchmark_sr=0.0)
        self.assertGreater(psr, 0.05)
        self.assertLess(psr, 0.95)

    def test_high_benchmark_lowers_psr(self):
        rng = np.random.default_rng(2)
        r = rng.normal(0.001, 0.01, size=500)
        low = probabilistic_sharpe_ratio(r, benchmark_sr=0.0)
        high = probabilistic_sharpe_ratio(r, benchmark_sr=0.5)
        self.assertGreater(low, high)

    def test_too_short_raises(self):
        with self.assertRaises(ValueError):
            probabilistic_sharpe_ratio([0.01, 0.02])

    def test_zero_variance_raises(self):
        with self.assertRaises(ValueError):
            probabilistic_sharpe_ratio([0.01, 0.01, 0.01, 0.01, 0.01])


if __name__ == "__main__":
    unittest.main()
