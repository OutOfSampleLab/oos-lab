"""Tests for CombinatorialPurgedKFold."""

import unittest
import numpy as np

from oos_lab.cv.cpcv import CombinatorialPurgedKFold


class CPCVInitTests(unittest.TestCase):
    def test_invalid_n_splits_raises(self):
        with self.assertRaises(ValueError):
            CombinatorialPurgedKFold(n_splits=1, n_test_splits=1)

    def test_invalid_n_test_splits_raises(self):
        with self.assertRaises(ValueError):
            CombinatorialPurgedKFold(n_splits=6, n_test_splits=0)

    def test_n_test_splits_ge_n_splits_raises(self):
        with self.assertRaises(ValueError):
            CombinatorialPurgedKFold(n_splits=6, n_test_splits=6)
        with self.assertRaises(ValueError):
            CombinatorialPurgedKFold(n_splits=6, n_test_splits=7)

    def test_invalid_embargo_pct_raises(self):
        with self.assertRaises(ValueError):
            CombinatorialPurgedKFold(n_splits=6, n_test_splits=2,
                                     embargo_pct=-0.1)
        with self.assertRaises(ValueError):
            CombinatorialPurgedKFold(n_splits=6, n_test_splits=2,
                                     embargo_pct=1.0)


class CPCVCombinatoricsTests(unittest.TestCase):
    def test_n_splits_total_matches_combinatorics(self):
        cv = CombinatorialPurgedKFold(n_splits=6, n_test_splits=2)
        self.assertEqual(cv.n_splits_total, 15)
        cv2 = CombinatorialPurgedKFold(n_splits=10, n_test_splits=3)
        self.assertEqual(cv2.n_splits_total, 120)

    def test_paths_count(self):
        cv = CombinatorialPurgedKFold(n_splits=6, n_test_splits=2)
        self.assertEqual(cv.paths, 5)
        cv2 = CombinatorialPurgedKFold(n_splits=10, n_test_splits=3)
        self.assertEqual(cv2.paths, 36)


class CPCVSplitTests(unittest.TestCase):
    def setUp(self):
        self.n = 1000
        self.t1 = np.arange(1, 1001)
        self.cv = CombinatorialPurgedKFold(n_splits=6, n_test_splits=2,
                                           embargo_pct=0.01)

    def test_yields_correct_number_of_splits(self):
        splits = list(self.cv.split(self.n, self.t1))
        self.assertEqual(len(splits), 15)

    def test_no_overlap_between_train_and_test(self):
        for tr, te in self.cv.split(self.n, self.t1):
            self.assertEqual(len(np.intersect1d(tr, te)), 0)

    def test_test_size_approximately_correct(self):
        expected = (2 * self.n) // 6
        for _, te in self.cv.split(self.n, self.t1):
            self.assertAlmostEqual(te.size, expected, delta=2)

    def test_n_too_small_raises(self):
        with self.assertRaises(ValueError):
            list(self.cv.split(n=3, t1=np.arange(1, 4)))

    def test_t1_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            list(self.cv.split(n=10, t1=np.arange(1, 8)))


class CPCVPurgeEmbargoTests(unittest.TestCase):
    def test_embargo_drops_observations_after_test_group(self):
        n = 600
        t1 = np.arange(1, n + 1)
        cv = CombinatorialPurgedKFold(n_splits=6, n_test_splits=1,
                                      embargo_pct=0.05)
        embargo_size = int(0.05 * n)
        for tr, te in cv.split(n, t1):
            test_end = int(te.max()) + 1
            embargo_zone = np.arange(test_end,
                                     min(test_end + embargo_size, n))
            overlap = np.intersect1d(tr, embargo_zone)
            self.assertEqual(overlap.size, 0)

    def test_purge_removes_label_overlap(self):
        n = 600
        t1 = np.arange(5, n + 5)
        cv = CombinatorialPurgedKFold(n_splits=6, n_test_splits=1,
                                      embargo_pct=0.0)
        for tr, te in cv.split(n, t1):
            test_set = set(int(x) for x in te)
            for i in tr:
                self.assertNotIn(int(t1[i]), test_set)


if __name__ == "__main__":
    unittest.main()
