# 3rd party
import numpy as np
from sklearn.model_selection import KFold, ShuffleSplit

# 1st party
from skutils.train_test_iterators import (
    AblationSplit,
    TrainOnAllWrapper,
    TrainTestIterable,
    TrainTestSplit
)


################################################################################
def test_AblationSplit():
    cv = AblationSplit()
    assert AblationSplit().get_n_splits() == 4
    assert AblationSplit(KFold(n_splits=10)).get_n_splits() == 9
    assert AblationSplit(
        cv=KFold(n_splits=5),
        cv_nested=KFold(n_splits=2)
    ).get_n_splits() == 4 * 2


    # Test default split
    X = np.arange(5)
    cv = AblationSplit()
    splits = list(cv.split(X))
    assert len(splits) == cv.get_n_splits() == 4
    np.testing.assert_equal(splits, [
        ([0], [1, 2, 3, 4]),
        ([0, 1], [2, 3, 4]),
        ([0, 1, 2], [3, 4]),
        ([0, 1, 2, 3], [4])
    ])

    # Test configured CV
    X = np.arange(5)
    cv = AblationSplit(cv=KFold(shuffle=True, random_state=1))
    splits = list(cv.split(X))
    assert len(splits) == cv.get_n_splits() == 4
    np.testing.assert_equal(splits, [
        ([2], [0, 1, 3, 4]),
        ([1, 2], [0, 3, 4]),
        ([1, 2, 4], [0, 3]),
        ([0, 1, 2, 4], [3]),
    ])

    # Test non fold-based CV
    X = np.arange(5)
    cv = AblationSplit(cv=ShuffleSplit(n_splits=5, test_size=2, random_state=1))
    splits = list(cv.split(X))
    np.testing.assert_equal(splits, [
        # Shuffle test splits = [[1, 2], [0, 2], [2, 3], [0, 3], [0, 3]]
        ([1, 2], [0, 0, 0, 2, 2, 3, 3, 3]),
        ([0, 1, 2, 2], [0, 0, 2, 3, 3, 3]),
        ([0, 1, 2, 2, 2, 3], [0, 0, 3, 3]),
        ([0, 0, 1, 2, 2, 2, 3, 3], [0, 3]),
    ])

    # Test with test idxs on next fold instead of complement folds
    X = np.arange(5)
    cv = AblationSplit(test_on_complement=False)
    splits = list(cv.split(X))
    assert len(splits) == cv.get_n_splits() == 4
    np.testing.assert_equal(splits, [
        ([0], [1]),
        ([0, 1], [2]),
        ([0, 1, 2], [3]),
        ([0, 1, 2, 3], [4]),
    ])

    # Test fixed train and test sets
    X = np.arange(9)
    cv = AblationSplit(
        fixed_train = np.array([0,2]),
        fixed_test  = np.array([6,8]),
    )
    splits = list(cv.split(X))
    assert len(splits) == cv.get_n_splits() == 6
    np.testing.assert_equal(splits, [
        ([0, 2], [1, 3, 4, 5, 6, 7, 8]),
        ([0, 1, 2], [3, 4, 5, 6, 7, 8]),
        ([0, 1, 2, 3], [4, 5, 6, 7, 8]),
        ([0, 1, 2, 3, 4], [5, 6, 7, 8]),
        ([0, 1, 2, 3, 4, 5], [6, 7, 8]),
        ([0, 1, 2, 3, 4, 5, 7], [6, 8]),
    ])

    # Test nested cv splitting
    X = np.arange(10)
    cv = AblationSplit(
        cv = KFold(n_splits=5),
        cv_nested = KFold(n_splits=2),
    )
    splits = list(cv.split(X))
    assert len(splits) == cv.get_n_splits() == 8
    np.testing.assert_equal(splits, [
        # 5 folds: ([0, 1], [2, 3], [4, 5], [6, 7], [8, 9])
        # CV Split 1: ([0, 1], [2, 3, 4, 5, 6, 7, 8, 9])
        # Nested CV Splits 1 & 2:
        ([1], [0, 2, 3, 4, 5, 6, 7, 8, 9]),
        ([0], [1, 2, 3, 4, 5, 6, 7, 8, 9]),
        # CV Split 2: ([0, 1, 2, 3], [4, 5, 6, 7, 8, 9])
        # Nested CV Splits 3 & 4
        ([2, 3], [0, 1, 4, 5, 6, 7, 8, 9]),
        ([0, 1], [2, 3, 4, 5, 6, 7, 8, 9]),
        # CV Split 3: ([0, 1, 2, 3, 4, 5], [6, 7, 8, 9])
        # Nested CV Splits 5 & 6:
        ([3, 4, 5], [0, 1, 2, 6, 7, 8, 9]),
        ([0, 1, 2], [3, 4, 5, 6, 7, 8, 9]),
        # CV Split 4 ([0, 1, 2, 3, 4, 5, 6, 7], [8, 9])
        # Nested CV Splits 7 & 8:
        ([4, 5, 6, 7], [0, 1, 2, 3, 8, 9]),
        ([0, 1, 2, 3], [4, 5, 6, 7, 8, 9]),
    ])
