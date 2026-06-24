"""Tests for haircut_sharpe."""

import unittest
import numpy as np

from oos_lab.overfit.haircut import haircut_sharpe, HaircutResult


class HaircutBasicTests(unittest.TestCase):
    def test_single_strategy_no_haircut(self):
        result = haircut_sharpe([0.5], n_obs=252, method="holm")
        self.assertEqual(result.haircut_sharpes.size, 1)
        self.assertAlmostEqual(result.haircut_sharpes[0], 0.5, places=8)

    def test_holm_haircut_does_not_increase_sharpe(self):
        sharpes = [0.6, 0.5, 0.4, 0.3, 0.2]
        result = haircut_sharpe(sharpes, n_obs=252, method="holm")
        for raw, hc in zip(result.raw_sharpes, result.haircut_sharpes):
            self.assertLessEqual(hc, raw + 1e-9)

    def test_bhy_haircut_does_not_increase_sharpe(self):
        sharpes = [0.6, 0.5, 0.4, 0.3, 0.2]
        result = haircut_sharpe(sharpes, n_obs=252, method="bhy")
        for raw, hc in zip(result.raw_sharpes, result.haircut_sharpes):
            self.assertLessEqual(hc, raw + 1e-9)

    def test_both_methods_produce_valid_monotone_adjusted_pvalues(self):
        sharpes = np.linspace(0.2, 0.6, 20)
        for method in ("holm", "bhy"):
            result = haircut_sharpe(sharpes, n_obs=252, method=method)
            order = np.argsort(result.raw_pvalues)
            sorted_adj = result.adjusted_pvalues[order]
            for v in sorted_adj:
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 1.0)
            for i in range(len(sorted_adj) - 1):
                self.assertLessEqual(sorted_adj[i], sorted_adj[i + 1] + 1e-12)

    def test_more_strategies_harsher_haircut(self):
        small = haircut_sharpe([0.5], n_obs=252, method="holm")
        large = haircut_sharpe([0.5] + [0.0] * 99, n_obs=252, method="holm")
        self.assertLess(large.haircut_sharpes[0], small.haircut_sharpes[0])

    def test_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            haircut_sharpe([0.5], n_obs=252, method="invalid")

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            haircut_sharpe([], n_obs=252)

    def test_invalid_n_obs_raises(self):
        with self.assertRaises(ValueError):
            haircut_sharpe([0.5], n_obs=1)

    def test_result_is_haircutresult(self):
        result = haircut_sharpe([0.5, 0.3], n_obs=252)
        self.assertIsInstance(result, HaircutResult)
        self.assertEqual(result.method, "holm")


if __name__ == "__main__":
    unittest.main()
