
# Standard library
from collections.abc import Iterable

# 3rd party
from sklearn.model_selection import train_test_split

# Local
from ._types import NDArrayInt

################################################################################
class TrainTestSplit: # TODO: (BaseCrossValidator):
    """Wrapper around train_test_split to give it the BaseCrossValidator API"""
    # train_test_split docs say it is a wrapper around ShuffleSplit but
    # ShuffleSplit does not accept all the same arguments as train_test_split so
    # this wrapper seems necessary to allow users to specify this as their train
    # test splitter for fit_supervised_model.py
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @staticmethod
    def get_n_splits(X=None, y=None, groups=None) -> int:
        return 1

    def split(self, X, y=None, groups=None) -> Iterable[tuple[NDArrayInt, NDArrayInt]]:
        # Allow user to specify stratify in config even though train_test_split
        # requires array
        stratify = self.kwargs.pop('stratify', None)
        if stratify is True:
            stratify = y
        elif isinstance(stratify, str):
            stratify = X[stratify]

        _, _, y_train, y_test = train_test_split(X, y, stratify=stratify, **self.kwargs)

        return (
            (train.index, test.index)
            for train, test in ((y_train, y_test),)
        )
