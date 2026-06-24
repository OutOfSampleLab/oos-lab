"""Tests for deflated_sharpe_ratio and expected_max_sharpe."""

import unittest
import numpy as np

from oos_lab.metrics.deflated_sharpe import (
    expected_max_sharpe,
    deflated_sharpe_ratio,
)
from oos_lab.metrics.psr import probabilistic_sharpe_ratio


class ExpectedMaxSharpeTests(unittest.TestCase):
    def test_single_trial_is_zero(self):
        self.assertEqual(expected_max_sharpe(1, 0.04), 0.0)

    def test_zero_variance_is_zero(self):
        self.assertEqual(expected_max_sharpe(100, 0.0), 0.0)

    def test_grows_with_trials(self):
        a = expected_max_sharpe(10, 0.04)
        b = expected_max_sharpe(1000, 0.04)
        self.assertGreater(b, a)

    def test_grows_with_variance(self):
        a = expected_max_sharpe(100, 0.01)
        b = expected_max_sharpe(100, 0.04)
        self.assertGreater(b, a)

    def test_negative_variance_raises(self):
        with self.assertRaises(ValueError):
            expected_max_sharpe(10, -0.1)

    def test_zero_n_trials_raises(self):
        with self.assertRaises(ValueError):
            expected_max_sharpe(0, 0.04)


class DeflatedSharpeTests(unittest.TestCase):
    def test_single_trial_zero_variance_equals_psr(self):
        rng = np.random.default_rng(7)
        r = rng.normal(0.001, 0.01, size=500)
        psr = probabilistic_sharpe_ratio(r, benchmark_sr=0.0)
        dsr = deflated_sharpe_ratio(r, n_trials=1, var_sharpe_across_trials=0.0)
        self.assertAlmostEqual(psr, dsr, places=10)

    def test_more_trials_lower_dsr(self):
        rng = np.random.default_rng(8)
        r = rng.normal(0.001, 0.01, size=500)
        few = deflated_sharpe_ratio(r, n_trials=1, var_sharpe_across_trials=0.04)
        many = deflated_sharpe_ratio(r, n_trials=1000, var_sharpe_across_trials=0.04)
        self.assertGreater(few, many)

    def test_too_short_raises(self):
        with self.assertRaises(ValueError):
            deflated_sharpe_ratio([0.01, 0.02], n_trials=10,
                                  var_sharpe_across_trials=0.04)


if __name__ == "__main__":
    unittest.main()
