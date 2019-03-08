from sklearn.base import BaseEstimator, TransformerMixin
from typing import NewType, Dict, List, Set, Union, Tuple, Callable
from abc import ABC, abstractmethod
import numpy as np


class Relationship(ABC):
    __priority = 0
    has_mono = False

    def __init__(self, order, rc):
        self.order = order
        self.rc = rc

    @abstractmethod
    def get_filter(self, X):
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

    def get_filter(self, X):
        nan = np.isnan(X)
        lf = interval_filter(X, self.ll, self.bounds[0], "left")
        rf = interval_filter(X, self.ul, self.bounds[1], "right")
        return lf & rf & ~nan

    def __str__(self):
        return self.bounds[0] + str(self.ll) + "," + str(self.ul) + self.bounds[1] + " => %d" % self.order


class SingleValue(Relationship):
    __priority = -1

    def __init__(self, value, order: int = 0, rc: str = ""):
        super().__init__(order, rc)
        self.value = value

    def get_filter(self, X):
        return X == self.value

    def __str__(self):
        return "Single Value"


class Missing(Relationship):

    def __init__(self, order: int = 0, rc: str = ""):
        super().__init__(order, rc)

    def get_filter(self, X):
        return np.isnan(X)

    def __str__(self):
        return "Missing"


# TODO: these functions should be in charge of the numpy return shape
# return a tuple of array, mono
def _reorder_mono_none(x: np.ndarray, interval: Interval, rels: List[Relationship]) -> \
        Tuple[np.ndarray, List[int]]:
    x1, _ = _reorder_mono_dec(x, interval, rels)
    x2, _ = _reorder_mono_inc(x, interval, rels)

    return np.concatenate([x1, x2]), [1, 2]


def _reorder_mono_inc(x: np.ndarray, interval: Interval, rels: List[Relationship]) -> \
        Tuple[np.ndarray, List[int]]:

    pass


def _reorder_mono_dec(x: np.ndarray,  interval: Interval, rels: List[Relationship]) -> \
        Tuple[np.ndarray, List[int]]:
    pass


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

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        :param X: A numpy array-like object
        :return: A transformed numpy array with the same number of rows as X
        the dimension of the array is (x.shape[0], # Interval rules)
        """
        # if only one mono, simply rearrange the elements
        # TODO: dispatch based on monotonicity, pass in intervals and rels

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
                    res[f,i] = rel.order if not rel == interval else X[f]

                    # update the mask with the newest filter
                    prev = prev | f

            else:
                pass

        return res

    def fit_transform(self, X, y=None, **fit_params):
        pass


if __name__ == '__main__':

    rels = [
        Interval(order=99, ll=2, ul=3, bounds="[]"),
        Interval(order=999, ll=3, ul=400, bounds="(]"),
        SingleValue(value=400, order=100),
        Missing(order=-5)
    ]


    rm = RelationshipMapper(rels)

    x = np.array([-1, 2, 3, np.nan, 400, 2])

    #print(Interval(order=3).get_filter(x))

    y = rm.transform(x)
    print(np.concatenate([np.reshape(x, (-1,1)),y], axis=1))


    #rm = RelationshipMapper(rels=None)
