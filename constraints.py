import numpy as np
from typing import *
from sklearn.base import BaseEstimator, TransformerMixin
from functools import reduce
from abc import ABC, abstractmethod


class Constraint(ABC):
    def __init__(self, order: int):
        self.order = order

    @abstractmethod
    def get_filter(self, x: Iterable) -> np.ndarray:
        pass


class Interval(Constraint, ABC):
    """Base class and factory for interval constraints"""
    def __new__(cls, ll: float, ul: float, left_open: bool, right_open: bool, mono: int, order: int):
        if left_open:
            if right_open:
                obj = super(Interval, cls).__new__(IntervalOO)
            else:
                obj = super(Interval, cls).__new__(IntervalOC)
        else:
            if right_open:
                obj = super(Interval, cls).__new__(IntervalCO)
            else:
                obj = super(Interval, cls).__new__(IntervalCC)

        obj.ll = ll
        obj.ul = ul
        obj.mono = mono
        return obj

    def __init__(self, ll: float, ul: float, left_open: bool, right_open: bool, mono: int, order: int):
        super().__init__(order)

    def get_filter(self, x: Iterable) -> np.ndarray:
        return ~np.isnan(x)


class IntervalOO(Interval):
    def __str__(self):
        return f"({self.ll}, {self.ul}) => {self.order}"

    def get_filter(self, x: Iterable) -> np.ndarray:
        mask = super().get_filter(x)
        return np.greater(x, self.ll, where=mask) & np.less(x, self.ul, where=mask)


class IntervalOC(Interval):
    def __str__(self):
        return f"({self.ll}, {self.ul}] => {self.order}"

    def get_filter(self, x: Iterable) -> np.ndarray:
        mask = super().get_filter(x)
        return np.greater(x, self.ll, where=mask) & np.less_equal(x, self.ul, where=mask)


class IntervalCO(Interval):
    def __str__(self):
        return f"[{self.ll}, {self.ul}) => {self.order}"

    def get_filter(self, x: Iterable) -> np.ndarray:
        mask = super().get_filter(x)
        return np.greater_equal(x, self.ll, where=mask) & np.less(x, self.ul, where=mask)


class IntervalCC(Interval):
    def __str__(self):
        return f"[{self.ll}, {self.ul}] => {self.order}"

    def get_filter(self, x: Iterable) -> np.ndarray:
        mask = super().get_filter(x)
        return np.greater_equal(x, self.ll, where=mask) & np.less_equal(x, self.ul, where=mask)


class Value(Constraint):
    """Holds a single value"""

    def __init__(self, value: Any, order: int):
        super().__init__(order)
        self.value = value

    def get_filter(self, x: np.ndarray):
        return x == self.value

    def __str__(self):
        return f"Value: {self.value} => {self.order}"


class Missing(Constraint):
    """Holds missing"""

    def __init__(self, order: int):
        super().__init__(order)

    def get_filter(self, x: np.ndarray):
        return np.isnan(x)

    def __str__(self):
        return f"Missing => {self.order}"


class Constrainer(BaseEstimator, TransformerMixin):

    def __init__(self, constraints: List[Constraint]):
        self.constraints = constraints

    def __str__(self):
        strings = [str(m) for m in self.constraints]
        pos = [s.index("=>") for s in strings ]
        max_pos = max(pos)

        # Align based on =>
        return "\n".join([" " * (max_pos - i) + s for s, i in zip(strings, pos)])



if __name__ == '__main__':
    u = Missing(0)
    v = Value(-1, 0)

    w = Interval(1.0, 10.0, True, True, 0, 0)
    x = Interval(1.0, 10.0, True, False, 0, 0)
    y = Interval(1.0, 10.0, False, True, 0, 0)
    z = Interval(1.0, 10.0, False, False, 0, 0)

    tf = Constrainer([u, v, w, x, y, z])
    print(tf)
