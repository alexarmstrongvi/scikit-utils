
# Standard library
import itertools
from collections.abc import Iterable

# Local
from ._types import TrainTestIterable, NDArrayInt

################################################################################
class TrainOnAllWrapper:
    def __init__(self, cv: TrainTestIterable):
        self.cv = cv

    def get_n_splits(self, *args, **kwargs) -> int:
        return self.cv.get_n_splits(*args, **kwargs) + 1

    def split(self, X, y=None, groups=None) -> Iterable[tuple[NDArrayInt, NDArrayInt]]:
        iterable: Iterable[tuple[NDArrayInt, NDArrayInt]]
        if hasattr(self.cv, 'split'):
            iterable = self.cv.split(X, y, groups)
        elif hasattr(self.cv, '__iter__'):
            iterable = self.cv
        yield from itertools.chain(iterable, [(X.index, X.index)])
