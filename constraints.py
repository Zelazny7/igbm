import numpy as np
from typing import Iterable, Any, Optional, List, OrderedDict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from abc import ABC, abstractmethod
from util import indices
from itertools import chain


class Constraint(ABC):
    def __init__(self, order: int):
        self.order = order

    @abstractmethod
    def get_filter(self, x: Iterable) -> np.ndarray:
        """Return boolean numpy array where constraint is satisfied for input
           iterable x

        :param Iterable x: an iterable that can be coerced to a numpy array
        :returns np.ndarray: with dtype np.bool
        """
        pass


class BaseInterval(Constraint, ABC):
    """Base class for interval constraints

    :param float ll: Lower limit of interval
    :param float ul: Upper limit of interval
    :param bool left_open: If True the left side of the interval is open
    :param bool right_open: If True the right side of the interval is open
    :param int mono: Monotonicity constraint of the interval in {-1,0,1}
    :param int order: Absolute ordering w.r.t. target variable treatment
    """
    def __init__(self, ll: float, ul: float, left_open: bool,
                 right_open: bool, mono: int = 0, order: int = 0):
        super().__init__(order=order)
        self.ll = ll
        self.ul = ul
        self.left_open = left_open
        self.right_open = right_open
        self.mono = mono

    def get_filter(self, x: Iterable):
        raise NotImplementedError()

    @property
    def limits(self):
        return self.ll, self.ul


def Interval(ll: float, ul: float, left_open: bool, right_open: bool,
             mono: int = 0, order: int = 0):
    """Factory for interval constraints

    :param float ll: Lower limit of interval
    :param float ul: Upper limit of interval
    :param bool left_open: If True the left side of the interval is open
    :param bool right_open: If True the right side of the interval is open
    :param int mono: Monotonicity constraint of the interval in {-1,0,1}
    :param int order: Absolute ordering w.r.t. target variable treatment
    """
    if mono not in {-1, 0, 1}:
        raise ValueError(f"Invalid argument mono: {mono}")

    if left_open:
        if right_open:
            obj = IntervalOO(ll, ul, left_open, right_open, mono, order)
        else:
            obj = IntervalOC(ll, ul, left_open, right_open, mono, order)
    else:
        if right_open:
            obj = IntervalCO(ll, ul, left_open, right_open, mono, order)
        else:
            obj = IntervalCC(ll, ul, left_open, right_open, mono, order)

    return obj


class IntervalOO(BaseInterval):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __str__(self):
        rng = f"({self.ll}, {self.ul})"
        return f"{self.order:5} | {rng:<20}"

    def get_filter(self, x: Iterable) -> np.ndarray:
        z = np.ma.masked_invalid(x)
        return np.ma.filled(np.ma.greater(z, self.ll) &
                            np.ma.less(z, self.ul), False)


class IntervalOC(BaseInterval):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __str__(self):
        rng = f"({self.ll}, {self.ul}]"
        return f"{self.order:5} | {rng:<20}"

    def get_filter(self, x: Iterable) -> np.ndarray:
        z = np.ma.masked_invalid(x)
        return np.ma.filled(np.ma.greater(z, self.ll) &
                            np.ma.less_equal(z, self.ul), False)


class IntervalCO(BaseInterval):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __str__(self):
        rng = f"[{self.ll}, {self.ul})"
        return f"{self.order:5} | {rng:<20}"

    def get_filter(self, x: Iterable) -> np.ndarray:
        z = np.ma.masked_invalid(x)
        return np.ma.filled(np.ma.greater_equal(z, self.ll) &
                            np.ma.less(z, self.ul), False)


class IntervalCC(BaseInterval):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __str__(self):
        rng = f"[{self.ll}, {self.ul}]"
        return f"{self.order:5} | {rng:<20}"

    def get_filter(self, x: Iterable) -> np.ndarray:
        z = np.ma.masked_invalid(x)
        return np.ma.filled(np.ma.greater_equal(z, self.ll) &
                            np.ma.less_equal(z, self.ul), False)


class Value(Constraint):
    """Holds a single value"""

    def __init__(self, value: Any, order: int):
        super().__init__(order)
        self.value = value

    def get_filter(self, x: np.ndarray):
        return x == self.value

    def __str__(self):
        return f"{self.order:5} | {f'Value: {self.value}':<20}"


class Missing(Constraint):
    """Holds missing"""

    def __init__(self, order: int):
        super().__init__(order)

    def get_filter(self, x: np.ndarray):
        return np.isnan(x)

    def __str__(self):
        return f"{self.order:5} | {f'Missing: ':<20}"


class FittedConstraint:
    """store constraint and the value it should map to after fitting"""

    def __init__(self, constraint, value: Optional[float]):
        self.constraint = constraint
        self.value = value

    def __str__(self):
        return f"{str(self.constraint)} => {self.value}"

    def transform(self, X: np.ndarray, result: np.ndarray) -> np.ndarray:
        replace = X if self.value is None else self.value
        # make sure to only update output vector where filter is true
        # AND result == np.nan
        f = self.constraint.get_filter(X) & np.isnan(result)
        return np.where(f, replace, result)


class Blueprint:
    def __init__(self, constraints: List[FittedConstraint],
                 mono: Optional[int]):
        self.constraints = constraints
        self.mono = mono

    def __iter__(self):
        return iter(self.constraints)

    def __str__(self):
        # TODO: Need to add print method here to include monotonicity
        pass


class Constrainer(BaseEstimator, TransformerMixin):
    """Constrainer class that transforms vector into features
       for constrained learning"""

    def __init__(self, constraints: List[Constraint]):
        self.constraints = constraints
        self.blueprints: List[List[FittedConstraint]] = list()
        self.fitted = False

    def __str__(self):
        strings = [str(m) for m in self.constraints]
        pos = [s.index("=>") for s in strings]
        max_pos = max(pos)

        # Align based on =>
        out = [" " * (max_pos - i) + s for s, i in zip(strings, pos)]
        return "\n".join(out)

    @property
    def intervals(self) -> List[BaseInterval]:
        return [i for i in self.constraints if isinstance(i, BaseInterval)]

    @property
    def mono(self) -> List[int]:
        m = {-1: (-1, -1), 0: (-1, 1), 1: (1, 1)}
        if ~self.fitted:
            out = [m[i.mono] for i in self.intervals]
        else:
            out = [bp.mono for bp in self.blueprints]
        return list(chain.from_iterable(out))

    def order(self, desc=False):
        mul = -1 if desc else 1
        return indices([x.order * mul for x in self.constraints])

    def fit_interval(self, interval: Interval) -> List[Blueprint]:
        # each interval generates two vectors with different monotonicities
        if interval.mono == 0:
            monos = (1, -1)
        elif interval.mono == 1:
            monos = (1, 1)
        else:
            monos = (-1, -1)

        out = []
        for mi, mono in enumerate(monos):
            order = self.order(desc=False if mono == 1 else True)
            ll, ul = interval.limits

            # need the index order of the current interval not the
            # original order
            pos = self.constraints.index(interval)
            i = order[pos]

            # this sets what the value of the mapping will be based on
            # where the current interval index is and the relative positions
            # of the other constraints
            vals = list()
            for j in order:
                if j < i:
                    vals.append(ll - 1 - (i - j))
                elif j == i:
                    if mi == 0:
                        vals.append(ll - 1)
                    else:
                        vals.append(ul + 1)
                else:
                    vals.append(ul + 1 - (i - j))

            # current interval gets None value to signal pass-through
            # predictions
            vals[pos] = None

            bp = list()
            for (con, val) in zip(self.constraints, vals):
                bp.append(FittedConstraint(con, val))

            out.append(Blueprint(bp, mono))
        return out

    def fit(self, X):
        self.blueprints.clear()
        intervals = self.intervals

        # check if there are interval constraints
        if len(intervals) > 0:
            for interval in intervals:
                self.blueprints += self.fit_interval(interval)
        else:
            bp = []
            for con, val in zip(self.constraints, self.order()):
                bp.append(FittedConstraint(con, val))
            self.blueprints += [Blueprint(bp, None)]

        self.fitted = True
        return self

    def transform(self, X):
        check_array(X, accept_sparse=False, force_all_finite=False)

        if hasattr(X, "iloc"): # DataFrame
            res = [self._transform(X.iloc[:, i]) for i in range(X.shape[1])]
        else:
            res = [self._transform(X[:,i]) for i in range(X.shape[1])]
        return np.hstack(res)

    def _transform(self, X):
        out = []
        for bp in self.blueprints:
            # start with a vector of np.nan to fill with the
            # transformed results
            res = np.full_like(X, np.nan)
            for cons in bp:
                res = cons.transform(X, res)
            out.append(res.reshape(-1, 1))

        out = np.hstack(out)
        check_array(out, accept_sparse=False, force_all_finite=False)
        return out


if __name__ == '__main__':
    u = Missing(3)
    v = Value(-1, 4)
    w = Interval(-1, 10, False, False, 0, 2)
    x = Interval(10, 20, True, True, 0, 2)

    print(type(w.get_filter([1, 10])))

    tf = Constrainer([u, v, w, x])
    tf = Constrainer([u, v])

    print(tf.mono)

    z = np.arange(-1, 20, 1, dtype=np.float)
    z = np.concatenate([z, [np.nan]]).reshape(-1, 1)

    tf.fit(z)
    print(np.hstack([z, tf.transform(z)]))

    print(tf.mono)
    # tf._generate_blueprint()
    # print(tf.fit_transform(z))
