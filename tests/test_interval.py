from unittest import TestCase
from constraints import *
import numpy as np

class IntervalTest(TestCase):
    x = np.array([1, 2, np.nan, 4, 5], dtype=np.float)

    def test_invalid_mono(self):
        self.assertRaises(ValueError, Interval, 0.0, 10.0, True, True, 2, 0)

    def test_factory_oo(self):
        self.assertIsInstance(Interval(0.0, 10.0, True, True, 0, 0), IntervalOO)

    def test_factory_oc(self):
        self.assertIsInstance(Interval(0.0, 10.0, True, False, 0, 0), IntervalOC)

    def test_factory_co(self):
        self.assertIsInstance(Interval(0.0, 10.0, False, True, 0, 0), IntervalCO)

    def test_factory_cc(self):
        self.assertIsInstance(Interval(0.0, 10.0, False, False, 0, 0), IntervalCC)

    def test_filter_oo(self):
        oo = Interval(1.0, 5.0, True, True, 0, 0)
        np.testing.assert_array_equal(
            oo.get_filter(self.x),
            np.array([False, True, False, True, False]))

    def test_filter_oc(self):
        oc = Interval(1.0, 5.0, True, False, 0, 0)
        np.testing.assert_array_equal(
            oc.get_filter(self.x),
            np.array([False, True, False, True, True]))

    def test_filter_co(self):
        co = Interval(1.0, 5.0, False, True, 0, 0)
        np.testing.assert_array_equal(
            co.get_filter(self.x),
            np.array([True, True, False, True, False]))

    def test_filter_cc(self):
        cc = Interval(1.0, 5.0, False, False, 0, 0)
        np.testing.assert_array_equal(
            cc.get_filter(self.x),
            np.array([True, True, False, True, True]))

    def test_filter_missing(self):
        mc = Missing(0)
        np.testing.assert_array_equal(
            mc.get_filter(self.x),
            np.array([False, False, True, False, False]))

    def test_filter_value(self):
        vc = Value(2.0, 0)
        np.testing.assert_array_equal(
            vc.get_filter(self.x),
            np.array([False, True, False, False, False]))