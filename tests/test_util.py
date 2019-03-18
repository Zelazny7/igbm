from unittest import TestCase
from util import indices
import numpy as np


class UtilTest(TestCase):
    def test_indices(self):
        self.assertEqual(indices([0, 1, 2, 3, 4]), [0, 1, 2, 3, 4])

        self.assertEqual(indices([5, 4, 3, 2, 1]), [4, 3, 2, 1, 0])

        self.assertEqual(indices([0, 0, 1, 1, 2]), [0, 0, 1, 1, 2])

        self.assertEqual(indices([1, 1, 1, 1, 2]), [0, 0, 0, 0, 1])

        self.assertEqual(indices([1, 1, 1, 1, 2]), [0, 0, 0, 0, 1])

        self.assertEqual(indices([2, 2, 2, -1, -1]), [1, 1, 1, 0, 0])

        self.assertEqual(indices(np.array([2, 2, 2, -1, -1])), [1, 1, 1, 0, 0])
