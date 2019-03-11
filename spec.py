from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Tuple, Callable
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class Relationship(ABC):
    __priority = 0

    def __init__(self, order, rc):
        self.order = order
        self.rc = rc

    @abstractmethod
    def get_filter(self, x):
        pass

    @property
    def priority(self):
        return self.__priority


def _left_interval(bound) -> Callable:
    assert bound in "(["
    return np.greater if bound == "(" else np.greater_equal


def _right_interval(bound) -> Callable:
    assert bound in "])"
    return np.less if bound == ")" else np.less_equal


def interval_filter(x: np.ndarray, lim: float, bound: str, side: str) -> np.ndarray:
    assert side in ["left", "right"]
    fun = _left_interval(bound) if side == "left" else _right_interval(bound)
    nan = np.isnan(x)
    return fun(x, lim, where=~nan)


class Interval(Relationship):
    def __init__(self, mono: int = 0, ll: float = -np.inf, ul: float = np.inf, bounds: str = "(]",
                 order: int = 0, rc: str = ""):
        super().__init__(order, rc)
        self.has_mono = True
        self.mono = mono
        self.ll = ll
        self.ul = ul
        self.bounds = bounds

    def get_filter(self, x):
        nan = np.isnan(x)
        lf = interval_filter(x, self.ll, self.bounds[0], "left")
        rf = interval_filter(x, self.ul, self.bounds[1], "right")
        return lf & rf & ~nan

    def __str__(self):
        return self.bounds[0] + str(self.ll) + "," + str(self.ul) + self.bounds[1] + " => %d" % self.order


class SingleValue(Relationship):
    __priority = -1

    def __init__(self, value, order: int = 0, rc: str = ""):
        super().__init__(order, rc)
        self.value = value

    def get_filter(self, x):
        return x == self.value

    def __str__(self):
        return "Single Value"


class Missing(Relationship):

    def __init__(self, order: int = 0, rc: str = ""):
        super().__init__(order, rc)

    def get_filter(self, x):
        return np.isnan(x)

    def __str__(self):
        return "Missing"


# TODO: these functions should be in charge of the numpy return shape
# return a tuple of array, mono
def _reorder_mono_none(x: np.ndarray, interval: Interval, rels: List[Relationship]) -> \
        Tuple[np.ndarray, List[int]]:

    x1, m1 = _reorder_mono(x, interval, rels)
    x2, m2 = _reorder_mono(x, interval, rels)

    return np.hstack([x1, x2]), m1 + m2


def _reorder_mono(x: np.ndarray, interval: Interval, rels: List[Relationship]) -> \
        Tuple[np.ndarray, List[int]]:

    res = np.zeros_like(x)
    prev = np.array([False])

    rev = True if interval.mono == -1 else False
    sorted_rels = sorted(rels, key=lambda r: (r.priority, r.order), reverse=rev)

    # get index of the interval in the relationships
    idx = sorted_rels.index(interval)
    ll, ul = int(interval.ll), int(interval.ul)
    vals = list(range(ll - idx, ll)) + list(range(ul + 1, ul + len(sorted_rels) - idx))
    print(vals)

    pos = 0
    for rel in sorted_rels:
        f = rel.get_filter(x) & ~prev

        # apply the mask to the vector
        if rel == interval:
            res[f] = x[f]
        else:
            res[f] = vals[pos]
            pos += 1

        # update the mask with the newest filter
        prev = prev | f

    return res.reshape((-1, 2)), [interval.mono]


class RelationshipMapper(BaseEstimator, TransformerMixin):
    """Map data elements to training vectors to enforce required relationships"""

    def __init__(self, rels: List[Relationship]):
        """

        :param rels: A list of Relationship objects:
        :returns A RelationshipMapper object
        """
        self.rels = rels

    def __str__(self):
        return "\n".join([str(x) for x in self.rels])

    def fit(self, X):
        # do checks here
        pass

    def transform_refactor(self, x: np.ndarray) -> Tuple[np.ndarray, List[int]]:

        intervals = [x for x in self.rels if isinstance(x, Interval)]

        res, mono = ([], [])
        for i in intervals:
            if i.mono == 0:
                tmp, m = _reorder_mono_none(x, i, self.rels)
            else:
                tmp, m = _reorder_mono(x, i, self.rels)
            res += [tmp]
            mono += m

        return np.vstack(res), mono


    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        :param X: A numpy array-like object
        :return: A transformed numpy array with the same number of rows as X
        the dimension of the array is (x.shape[0], # Interval rules)
        """

        intervals = [x for x in self.rels if isinstance(x, Interval)]
        res = np.zeros((X.shape[0], len(intervals)))

        for (i, interval) in enumerate(intervals):
            # different behavior depending on the monotonicity
            # zero mono must create two vectors for each interval
            # - a vector with +1 mono and another with -1
            # the other relationships should be shuffled around depending on the current mono

            if interval.mono == 0:
                rels = sorted(self.rels, key=lambda x: (x.priority, x.order))

                # keep building the mask as relationships are processed
                prev = np.array([False])

                for rel in rels:
                    f = rel.get_filter(X) & ~prev

                    # apply the mask to the vector
                    res[f, i] = rel.order if not rel == interval else X[f]

                    # update the mask with the newest filter
                    prev = prev | f

            else:
                pass

        return res

    def fit_transform(self, X, y=None, **fit_params):
        pass


if __name__ == '__main__':

    rm = RelationshipMapper([
        Interval(order=99, ll=10, ul=20, bounds="[]"),
        Interval(order=999, ll=21, ul=400, bounds="(]", mono=0),
        SingleValue(value=-1, order=100),
        Missing(order=-5)
    ])

    z = np.array([-1, 2, 3, 10, 20, 30, 40, np.nan, 400, 2])
    # print(Interval(order=3).get_filter(x))
    y = rm.transform_refactor(z)

    print(z)
    print(pd.DataFrame(y[0]))


    # print(np.concatenate([np.reshape(z, (-1, 1)), y], axis=1))

    # rm = RelationshipMapper(rels=None)
