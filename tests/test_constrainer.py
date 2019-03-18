from unittest import TestCase
from constraints import Constrainer, Missing, Value, Interval
import numpy as np


class TestConstrainer(TestCase):

    def test_intervals_property(self):
        constraints = [
            Missing(0),
            Value(-1, 0),
            Interval(1.0, 5.0, True, True, 0, 0)
        ]

        # test intervals returns list of interval constraints
        x = Constrainer(constraints)
        self.assertEqual(x.intervals, [constraints[2]])

        # test constrainer with no intervals returns empty list
        x = Constrainer(constraints[:2])
        self.assertEqual(x.intervals, [])

    def test_order(self):
        constraints = [
            Missing(0),
            Value(-1, 1),
            Interval(1.0, 5.0, True, True, 0, 2)
        ]

        # test intervals returns list of interval constraints
        x = Constrainer(constraints)

        self.assertEqual(x.order(), [0, 1, 2])
        self.assertEqual(x.order(desc=True), [2, 1, 0])

    def test_transform(self):
        constraints = [
            Missing(0),
            Value(-1, 1),
            Interval(1.0, 5.0, False, False, 0, 2)
        ]

        tf = Constrainer(constraints)
        x = np.concatenate([np.array([np.nan, -1]), np.arange(1, 5, 1)])
        xout = tf.fit_transform(x.reshape(-1, 1))

        expected = np.array([
            [-2, 8],
            [-1, 7],
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4]
        ])

        np.testing.assert_array_equal(xout, expected)
