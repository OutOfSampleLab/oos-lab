"""Tests for WalkForward splitter."""

import unittest
import numpy as np

from oos_lab.cv.walk_forward import WalkForward


class WalkForwardTests(unittest.TestCase):
    def test_rolling_basic_count_and_sizes(self):
        wf = WalkForward(train_size=100, test_size=20)
        splits = list(wf.split(200))
        self.assertEqual(len(splits), 5)
        for tr, te in splits:
            self.assertEqual(tr.size, 100)
            self.assertEqual(te.size, 20)
            self.assertEqual(te[0], tr[-1] + 1)

    def test_n_splits_matches(self):
        wf = WalkForward(train_size=100, test_size=20)
        self.assertEqual(wf.n_splits(200), 5)
        self.assertEqual(wf.n_splits(150), 2)
        self.assertEqual(wf.n_splits(110), 0)
        actual = sum(1 for _ in wf.split(160))
        self.assertEqual(wf.n_splits(160), actual)

    def test_anchored_grows(self):
        wf = WalkForward(train_size=100, test_size=20, anchored=True)
        prev = 100
        for i, (tr, te) in enumerate(wf.split(200)):
            self.assertEqual(tr[0], 0)
            if i > 0:
                self.assertGreater(tr.size, prev)
            prev = tr.size

    def test_test_windows_non_overlapping(self):
        wf = WalkForward(train_size=100, test_size=20)
        tests = [te for _, te in wf.split(200)]
        for i in range(len(tests) - 1):
            self.assertEqual(tests[i][-1] + 1, tests[i + 1][0])

    def test_too_small_n_raises(self):
        wf = WalkForward(train_size=100, test_size=20)
        with self.assertRaises(ValueError):
            list(wf.split(50))

    def test_invalid_train_size_raises(self):
        with self.assertRaises(ValueError):
            WalkForward(train_size=0, test_size=20)

    def test_invalid_test_size_raises(self):
        with self.assertRaises(ValueError):
            WalkForward(train_size=100, test_size=0)


if __name__ == "__main__":
    unittest.main()
