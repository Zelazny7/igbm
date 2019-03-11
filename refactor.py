import numpy as np
from typing import *
from sklearn.base import BaseEstimator, TransformerMixin
from functools import reduce
from abc import ABC, abstractmethod

class Map(ABC):
    """Super class for other Map types"""

    def __init__(self, order: int):
        """:param order: the priority order for handling masks"""
        self.order = order

    @abstractmethod
    def get_filter(self, x: np.ndarray):
        pass


class Interval(Map):
    """Holds a range of values"""

    def __init__(self, ll: float, ul: float, left_open: bool, right_open: bool, mono: int, order: int):
        """
        :param order:
        :param ll: lower limit
        :param ul: upper limit
        :param left_open: boundary for left
        :param right_open: boundary for right
        :param mono: monotonicity
        """
        super().__init__(order)
        self.ll = ll
        self.ul = ul
        self.left_open = left_open
        self.right_open = right_open
        self.mono = mono

    def __str__(self):
        left = "(" if self.left_open else "["
        right = ")" if self.right_open else "]"
        return f"{left} {self.ll:.2f}, {self.ul:.2f} {right} => {self.order}"

    def get_filter(self, x: np.ndarray):
        nan = np.isnan(x)
        lfun = np.greater if self.left_open else np.greater_equal
        rfun = np.less if self.right_open else np.less_equal

        result = lfun(x, self.ll, where=~nan) & rfun(x, self.ul, where=~nan)
        result[nan] = False
        return result


class Missing(Map):
    """Holds missing"""
    def __init__(self, order: int):
        super().__init__(order)

    def get_filter(self, x: np.ndarray):
        return np.isnan(x)

    def __str__(self):
        return f"Missing => {self.order}"


class Value(Map):
    """Holds a single value"""
    def __init__(self, value: Any, order: int):
        super().__init__(order)
        self.value = value

    def get_filter(self, x: np.ndarray):
        return x == self.value

    def __str__(self):
        return f"Value: {self.value} => {self.order}"


class GBMTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, *args: Map):
        self.maps = args

    def value_filter(self, x):
        """get filter for all values so they can be excluded from intervals"""
        vals = [m.get_filter(x) for m in self.maps if not isinstance(m, Interval)]
        return reduce(lambda a, b: a | b, vals)

    def transform(self, x: np.ndarray):
        vf = self.value_filter(x)

        out = np.zeros_like(x)
        for map in self.maps:
            f = map.get_filter(x)
            if isinstance(map, Interval):
                out[f & ~vf] = map.order
            else:
                out[f] = map.order

        return out

    def __str__(self):
        strs = [str(m) for m in self.maps]
        pos = [s.index("=>") for s in strs]
        max_pos = max(pos)

        # Align based on =>
        return "\n".join([" " * (max_pos - i) + s for s, i in zip(strs, pos)])








if __name__ == '__main__':

    tf = GBMTransformer(
        Missing(0),
        Interval(0, 10, True, False, 0, 1),
        Interval(10, 20, True, False, 0, 2),
        Interval(20, 30, True, False, 0, 3),
        Value(31, order=4)
    )

    x = np.array([np.nan] + list(range(0, 32)))

    tf.transform(x)