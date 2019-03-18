from unittest import TestCase
import constraints as cn
import numpy as np


class IntervalTest(TestCase):
    x = np.array([1, 2, np.nan, 4, 5], dtype=np.float)

    def test_invalid_mono(self):
        self.assertRaises(ValueError, cn.Interval, 0.0, 10.0, True, True, 2, 0)

    def test_factory_oo(self):
        self.assertIsInstance(cn.Interval(0.0, 10.0, True, True, 0, 0),
                              cn.IntervalOO)

    def test_factory_oc(self):
        self.assertIsInstance(cn.Interval(0.0, 10.0, True, False, 0, 0),
                              cn.IntervalOC)

    def test_factory_co(self):
        self.assertIsInstance(cn.Interval(0.0, 10.0, False, True, 0, 0),
                              cn.IntervalCO)

    def test_factory_cc(self):
        self.assertIsInstance(cn.Interval(0.0, 10.0, False, False, 0, 0),
                              cn.IntervalCC)

    def test_filter_oo(self):
        oo = cn.Interval(1.0, 5.0, True, True, 0, 0)
        np.testing.assert_array_equal(
            oo.get_filter(self.x),
            np.array([False, True, False, True, False]))

    def test_filter_oc(self):
        oc = cn.Interval(1.0, 5.0, True, False, 0, 0)
        np.testing.assert_array_equal(
            oc.get_filter(self.x),
            np.array([False, True, False, True, True]))

    def test_filter_co(self):
        co = cn.Interval(1.0, 5.0, False, True, 0, 0)
        np.testing.assert_array_equal(
            co.get_filter(self.x),
            np.array([True, True, False, True, False]))

    def test_filter_cc(self):
        cc = cn.Interval(1.0, 5.0, False, False, 0, 0)
        np.testing.assert_array_equal(
            cc.get_filter(self.x),
            np.array([True, True, False, True, True]))

    def test_filter_missing(self):
        mc = cn.Missing(0)
        np.testing.assert_array_equal(
            mc.get_filter(self.x),
            np.array([False, False, True, False, False]))

    def test_filter_value(self):
        vc = cn.Value(2.0, 0)
        np.testing.assert_array_equal(
            vc.get_filter(self.x),
            np.array([False, True, False, False, False]))
