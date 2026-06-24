"""Tests for sharpe_ratio."""

import unittest
import numpy as np

from oos_lab.metrics.sharpe import sharpe_ratio


class SharpeTests(unittest.TestCase):
    def test_positive_mean_gives_positive_sharpe(self):
        rng = np.random.default_rng(42)
        r = rng.normal(0.001, 0.01, size=1000)
        self.assertGreater(sharpe_ratio(r, periods_per_year=252), 0.5)

    def test_zero_variance_raises(self):
        with self.assertRaises(ValueError):
            sharpe_ratio([0.01, 0.01, 0.01])

    def test_single_observation_raises(self):
        with self.assertRaises(ValueError):
            sharpe_ratio([0.01])

    def test_annualisation_factor(self):
        r = [0.001, -0.002, 0.003, -0.001, 0.002]
        sr_per_period = sharpe_ratio(r, periods_per_year=1)
        sr_annual = sharpe_ratio(r, periods_per_year=252)
        self.assertAlmostEqual(sr_annual, sr_per_period * np.sqrt(252), places=10)

    def test_risk_free_subtracts(self):
        r = [0.005, 0.006, 0.004, 0.005]
        with_rf = sharpe_ratio(r, periods_per_year=1, risk_free=0.005)
        self.assertAlmostEqual(with_rf, 0.0, places=10)


if __name__ == "__main__":
    unittest.main()
