"""Tests for probability_of_backtest_overfit."""

import math
import unittest
import numpy as np

from oos_lab.overfit.pbo import probability_of_backtest_overfit, PBOResult


class PBOTests(unittest.TestCase):
    def test_random_matrix_pbo_in_middle(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.01, size=(1000, 30))
        result = probability_of_backtest_overfit(returns, n_partitions=8)
        self.assertGreater(result.pbo, 0.2)
        self.assertLess(result.pbo, 0.8)

    def test_real_alpha_low_pbo(self):
        rng = np.random.default_rng(0)
        n = 1000
        m = 30
        returns = rng.normal(0.0, 0.01, size=(n, m))
        returns[:, 0] += 0.005
        result = probability_of_backtest_overfit(returns, n_partitions=8)
        self.assertLess(result.pbo, 0.3)

    def test_result_is_pboresult(self):
        rng = np.random.default_rng(1)
        returns = rng.normal(0.0, 0.01, size=(200, 10))
        result = probability_of_backtest_overfit(returns, n_partitions=8)
        self.assertIsInstance(result, PBOResult)
        self.assertTrue(0.0 <= result.pbo <= 1.0)
        self.assertEqual(result.logits.shape, (math.comb(8, 4),))

    def test_invalid_dim_raises(self):
        with self.assertRaises(ValueError):
            probability_of_backtest_overfit(np.zeros((10,)), n_partitions=4)

    def test_invalid_n_partitions_raises(self):
        with self.assertRaises(ValueError):
            probability_of_backtest_overfit(np.zeros((100, 5)), n_partitions=5)
        with self.assertRaises(ValueError):
            probability_of_backtest_overfit(np.zeros((100, 5)), n_partitions=2)

    def test_too_few_obs_raises(self):
        with self.assertRaises(ValueError):
            probability_of_backtest_overfit(np.zeros((10, 5)), n_partitions=16)

    def test_too_few_variants_raises(self):
        with self.assertRaises(ValueError):
            probability_of_backtest_overfit(np.zeros((100, 1)), n_partitions=8)

    def test_degradation_attributes(self):
        rng = np.random.default_rng(7)
        returns = rng.normal(0.0, 0.01, size=(1000, 20))
        result = probability_of_backtest_overfit(returns, n_partitions=8)
        self.assertIsInstance(result.performance_degradation_slope, float)
        self.assertIsInstance(result.performance_degradation_intercept, float)


if __name__ == "__main__":
    unittest.main()
