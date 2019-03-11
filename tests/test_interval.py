from unittest import TestCase
import spec
import numpy as np

class TestInterval(TestCase):

    def test_left_interval_bounds(self):
        with self.assertRaises(AssertionError):
            spec._left_interval("]")

        with self.assertRaises(AssertionError):
            spec._left_interval(")")

    def test_left_interval_return_callable(self):
        self.assertEquals(spec._left_interval("("), np.greater)
        self.assertEquals(spec._left_interval("["), np.greater_equal)


    def test_right_interval_bounds(self):
        with self.assertRaises(AssertionError):
            spec._right_interval("[")

        with self.assertRaises(AssertionError):
            spec._right_interval("(")

    def test_right_interval_return_callable(self):
        self.assertEquals(spec._right_interval(")"), np.less)
        self.assertEquals(spec._right_interval("]"), np.less_equal)

    def test_interval_filter(self):
        x = np.array([1, 2, 3, 4, 5])
        np.testing.assert_array_equal(spec.interval_filter(x, 3, "]", "right"),
                                      np.array([True, True, True, False, False]))



