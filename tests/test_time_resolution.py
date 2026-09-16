import unittest

import numpy as np

from xplat.tpp.time_resolution import completed_gap_minutes, conservative_minute_bounds


class TimestampResolutionTest(unittest.TestCase):
    def test_completed_minutes_preserve_current_discretization(self):
        ts = np.array([0, 30, 90, 209, 329], dtype=np.int64)
        np.testing.assert_array_equal(
            completed_gap_minutes(ts), np.array([0, 1, 1, 2])
        )

    def test_conservative_bounds_match_archived_protocol(self):
        lower, upper = conservative_minute_bounds(np.array([0, 1, 2, 10]))
        np.testing.assert_allclose(lower, [1e-3, 1e-3, 1.0, 9.0])
        np.testing.assert_allclose(upper, [1.0, 2.0, 3.0, 11.0])

    def test_exact_positive_gap_is_inside_conservative_interval(self):
        exact_minutes = np.array([0.5, 1.0, 1.983, 2.0, 10.25])
        completed = np.floor(exact_minutes).astype(np.int64)
        lower, upper = conservative_minute_bounds(completed)
        self.assertTrue(np.all(exact_minutes > lower))
        self.assertTrue(np.all(exact_minutes <= upper))

    def test_unsorted_input_is_rejected(self):
        with self.assertRaises(ValueError):
            completed_gap_minutes(np.array([120, 60]))


if __name__ == "__main__":
    unittest.main()
