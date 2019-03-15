from unittest import TestCase
from constraints import *


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

