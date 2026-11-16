"""Welch's t-test and sample statistics with exact known values."""

from __future__ import annotations

import math
import unittest

from stats import WelchResult, mean, stddev, welch_t_test


class MeanStddevTests(unittest.TestCase):

    def test_mean_basic(self):
        self.assertEqual(mean([1, 2, 3, 4]), 2.5)
        self.assertEqual(mean([7]), 7.0)

    def test_mean_empty_raises(self):
        with self.assertRaises(ValueError):
            mean([])

    def test_stddev_sample_variance(self):
        # mean = 5, sum of squared deviations = 32, n-1 = 7
        # sample variance = 32/7 -> sd = sqrt(32/7)
        self.assertAlmostEqual(
            stddev([2, 4, 4, 4, 5, 5, 7, 9]),
            (32 / 7) ** 0.5,
            places=12,
        )

    def test_stddev_below_two_is_zero(self):
        self.assertEqual(stddev([]), 0.0)
        self.assertEqual(stddev([5.0]), 0.0)

    def test_stddev_constant_sample_is_zero(self):
        self.assertEqual(stddev([3.0, 3.0, 3.0]), 0.0)


class WelchTests(unittest.TestCase):

    def test_known_textbook_value(self):
        # Fixed reference computed from the Welch formula in stats.py
        # docstring (asserted to full float precision).
        result = welch_t_test([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
        self.assertEqual(result.t, -1.8973665961010275)
        self.assertEqual(result.df, 5.882352941176471)

    def test_identical_samples_give_zero_t(self):
        result = welch_t_test([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        self.assertEqual(result.t, 0.0)

    def test_zero_variance_different_means_gives_signed_infinity(self):
        result = welch_t_test([1, 1, 1], [2, 2, 2])
        self.assertEqual(result.t, -math.inf)
        result = welch_t_test([2, 2, 2], [1, 1, 1])
        self.assertEqual(result.t, math.inf)

    def test_clearly_separated_means_give_large_positive_t(self):
        result = welch_t_test([10.0, 10.1, 9.9, 10.0], [1.0, 1.1, 0.9, 1.0])
        self.assertGreater(result.t, 20)
        self.assertGreater(result.df, 0)

    def test_result_is_named_tuple(self):
        result = welch_t_test([1.0, 2.0], [3.0, 4.0])
        self.assertIsInstance(result, WelchResult)
        t, df = result  # unpackable
        self.assertEqual((t, df), (result.t, result.df))

    def test_needs_two_samples_per_group(self):
        with self.assertRaises(ValueError):
            welch_t_test([1.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            welch_t_test([1.0, 2.0], [1.0])

    def test_sign_convention(self):
        # First group slower (larger) -> positive t.
        self.assertGreater(welch_t_test([5.0, 5.1], [1.0, 1.1]).t, 0)


if __name__ == "__main__":
    unittest.main()
