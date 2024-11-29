# Standard library
from collections.abc import Iterable, Sequence
import itertools
import logging
from typing import Callable

# 3rd party
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.model_selection import (
    BaseCrossValidator,
    BaseShuffleSplit,
    GroupKFold,
    KFold,
    PredefinedSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split
)

# Type aliases
NDArrayInt = NDArray[np.int64]
# TODO: Support Indexes = NDArrayInt | pd.Index
Indexes = NDArrayInt # Indices for a subset of the input dataset
Fold = Indexes # Indexes object that is one subset among a partition of the dataset
Split  = tuple[Indexes, Indexes] # Single train-test division of dataset
TrainTestIndexIterable = Iterable[Split]
TrainTestIterable = (
    BaseCrossValidator
    | TrainTestIndexIterable
    | BaseShuffleSplit
)

# Globals
log = logging.getLogger(__name__)

################################################################################
class TrainTestSplit:
    """Wrapper around train_test_split to give it the BaseCrossValidator API"""
    # TODO: Find a way to do this with ShuffleSplit.
    # train_test_split docs say it is a wrapper around ShuffleSplit but
    # ShuffleSplit does not accept all the same arguments as train_test_split so
    # this wrapper seems necessary to allow users to specify this as their train
    # test splitter for fit_supervised_model.py
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @staticmethod
    def get_n_splits(X=None, y=None, groups=None) -> int:
        return 1

    def split(self, X, y=None, groups=None) -> TrainTestIndexIterable:
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

class TrainOnAllWrapper:
    '''Meta CV Iterator that adds to the splits a final split that trains and
    then evaluates on all data'''
    def __init__(self, cv: TrainTestIterable):
        self.cv = cv

    def get_n_splits(self, *args, **kwargs) -> int:
        return self.cv.get_n_splits(*args, **kwargs) + 1

    def split(self, X, y=None, groups=None) -> TrainTestIndexIterable:
        iterable: TrainTestIndexIterable
        if hasattr(self.cv, 'split'):
            iterable = self.cv.split(X, y, groups)
        elif hasattr(self.cv, '__iter__'):
            iterable = self.cv
        yield from itertools.chain(iterable, [(X.index, X.index)])

IndexSpecification = (
    None  # No indexes
    | str # DataFrame column mask or eval statement (e.g. 'feature > 20')
    | np.ndarray  # Predefined indexes
    | Callable[[pd.DataFrame], np.ndarray] # Compute indexes from dataframe
)

def get_indexes(
    fixed_split: IndexSpecification,
    X: pd.DataFrame,
) -> np.ndarray:
    # TODO: Always return pd.Index
    if fixed_split is None:
        return np.array([], dtype=int)
    if isinstance(fixed_split, str):
        return np.flatnonzero(X.eval(fixed_split))
    if callable(fixed_split):
        return fixed_split(X)
    if isinstance(fixed_split, np.ndarray):
        return fixed_split
    raise TypeError(type(fixed_split))

class AblationSplit:
    """Cross-validator for ablation studies

    Ablation studies in machine learning, named after similar studies in
    biology, involve measuring the contribution of components on an ML pipeline
    by removing those components one at a time. For example, measuring feature
    importance by retraining a model with each feature removed form the training
    data is an ablation study. This Meta-CV Iterator wraps base iterators to
    enable iteration over training datasets that grow with each split, forming a
    superset of the previous training set.
    """
    def __init__(
        self,
        cv                : BaseCrossValidator | None = None,
        cv_nested         : BaseCrossValidator | None = None,
        fixed_train       : IndexSpecification = None,
        fixed_test        : IndexSpecification = None,
        test_on_complement: bool = True,
    ):
        # Defaults
        if cv is None:
            cv = KFold()
        self.cv          = cv
        self.cv_nested   = cv_nested
        self.fixed_train = fixed_train
        self.fixed_test  = fixed_test
        self.test_on_complement = test_on_complement

    def get_n_splits(self, X=None, y=None, groups=None):
        # 1 of the n splits is used only for testing
        n_splits = self.cv.get_n_splits(X,y,groups)-1

        if self.cv_nested is not None:
            # TODO: Is this guaranteed to work? The second cv splitting is
            # applied separately to each main cv split so are those all
            # guaranteed to have the same number of splits? It depends if there
            # splitters where n_splits depends on X?
            n_splits *= self.cv_nested.get_n_splits(X,y,groups)

        if self.fixed_train is not None:
            n_splits += 1
        if self.fixed_test is not None:
            n_splits += 1

        return n_splits

    def split(self, X, y=None, groups=None):
        fixed_train_idxs = get_indexes(self.fixed_train, X)
        fixed_test_idxs  = get_indexes(self.fixed_test, X)
        if len(np.intersect1d(fixed_train_idxs, fixed_test_idxs)) > 0:
            raise ValueError('Fixed train and test sets overlap')
        fixed_idxs = np.hstack([fixed_train_idxs, fixed_test_idxs])
        iter_idxs  = get_index_complement(fixed_idxs, len(X))

        # Use test indexes to generate accumulating ablation training data
        # indexes given that, for KFold-style iterators, the test indexes
        # partition the datasets. Works well enough for ShuffleSplit-style
        # iterators though those are not the main use case for AblationSplit.
        idxs_iter = (iter_idxs[idxs] for _, idxs in self.cv.split(iter_idxs))

        # Accumulate indexes to get growing training sets
        cv_iter = _create_accumulating_cv(
            idxs_iter             = idxs_iter,
            n_splits              = self.cv.get_n_splits(X,y,groups),
            test_on_complement    = self.test_on_complement,
            include_train_on_none = self.fixed_train is not None,
            include_test_on_none  = self.fixed_test is not None,
        )

        # Apply nested cv splitting so that more/all data is included in test
        # set at some point
        if self.cv_nested is not None:
            cv_iter = nested_cv_split(self.cv_nested, cv_iter)

        # Combine all index arrays into a single train and test array
        for train_idx, test_idx in cv_iter:
            train_idx = np.hstack((fixed_train_idxs, train_idx))
            test_idx = np.hstack((fixed_test_idxs, test_idx))
            # NOTE: Sorting not necessary but makes results more predictable and
            # simplifies testing
            train_idx.sort()
            test_idx.sort()
            # log.debug('Train indexes: %s', train_idx)
            # log.debug('Test indexes : %s', test_idx)
            yield train_idx, test_idx

def _create_accumulating_cv(
    idxs_iter            : Iterable[Indexes],
    n_splits             : int,
    test_on_complement   : bool,
    include_train_on_none: bool = False,
    include_test_on_none : bool = False,
) -> TrainTestIndexIterable:
    idxs = list(idxs_iter)
    # log.debug('Accumulating indexes: %s', idxs)
    start = 0 if include_train_on_none else 1
    stop = n_splits+1 if include_test_on_none else n_splits
    for i in range(start, stop):
        train_idxs = idxs[:i]
        test_idxs  = idxs[slice(i, n_splits+1 if test_on_complement else i+1)]
        yield (
            np.hstack(train_idxs) if len(train_idxs) > 0 else np.array([], int),
            np.hstack(test_idxs)  if len(test_idxs)  > 0 else np.array([], int),
        )

def nested_cv_split(cv: BaseCrossValidator, cv_iter: TrainTestIndexIterable) -> TrainTestIndexIterable:
    """Update CV iterator to, at each split, perform a nested CV split of
    the training data in the split, combining the nested test indexes with the
    initial test indexes"""
    for train_idxs, test_idxs in cv_iter:
        # log.debug('Nested split of indexes')
        # log.debug('\tTrain: %s', train_idxs)
        # log.debug('\tTest : %s', test_idxs)
        for train_idxs_nested, test_idxs_nested in cv.split(train_idxs):
            # log.debug('\t\tNested Train: %s', train_idxs_nested)
            # log.debug('\t\tNested Test : %s', test_idxs_nested)
            yield (
                train_idxs[train_idxs_nested],
                np.hstack((train_idxs[test_idxs_nested], test_idxs))
            )

def get_index_complement(indexes: Indexes, n_indexes: int) -> Indexes:
    if indexes is None:
        return np.arange(n_indexes)
    mask = np.full(n_indexes, fill_value=True, dtype=bool)
    mask[indexes] = False
    return np.flatnonzero(mask)

################################################################################
# NOTE: Work in progress
class AblatedRepeatedKFold:
    """Repeated KFold where the number of folds decreases with repetition"""
    def __init__(
        self,
        n_splits: Sequence[int] = range(5,1),
        logarithmic: bool = False
    ):
        pass

class UnshuffledGroupKFold:
    """GroupKFold but the groups are not shuffled into splits"""
    pass

class UnshuffledStratifiedGroupKFold:
    """StratifiedGroupKFold but the groups are not shuffled into splits

    Only works if all entries in a group are part of a single stratum.
    Otherwise, the order of groups can differ between strata and determining
    folds is impossible.
    """
    pass

class CVIterator:
    def __init__(
        self,
        n_splits = 5,
        n_repeats = 1,
        shuffle : bool = True,
        incremental : bool = False,
        test_fold = None,
        preserve_order : bool = False,
        shuffle_folds  : bool = False,
        random_state = None,
    ):
        """
        All-in-one configurable iterator for cross-validation.

        Parameters
        ==========
        n_splits:
            Number of folds. Must be at least 2.
        n_repeats:
            Number of times cross-validator needs to be repeated.
        shuffle:
            Whether to shuffle data before splitting into folds and/or deciding
            the order in which to process the folds
        incremental:
            Incrementally add folds to dataset used for training,
            validating only on the next fold before it gets included in
            training. Useful to time series data. Similar behavior to sklearn's
            TimeSeriesSplit.
        test_fold:
            The entry test_fold[i] represents the index of the test set that
            sample i belongs to. It is possible to exclude sample i from any
            test set (i.e. include sample i in every training set) by setting
            test_fold[i] equal to -1.
        preserve_order:
            TODO:
        shuffle_folds:
            TODO:
        random_state:
            Controls the randomness of each repeated cross-validation instance.
            Pass an int for reproducible output across multiple function calls.
        """
        self.n_splits       = n_splits
        self.n_repeats      = n_repeats
        self.shuffle        = shuffle
        self.incremental    = incremental
        self.test_fold      = test_fold
        self.preserve_order = preserve_order
        self.shuffle_folds  = shuffle_folds
        self.rng            = np.random.default_rng(random_state)

        # Set after each call to split()
        self._strata_id = None
        self._fold_id = None

    def split(
        self,
        data,
        *,
        X       = None,
        y       = None,
        stratum = None,
        group   = None,

    ): # -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        data, col_names = _require_dataframe(data, features=X, y_true=y, stratum=stratum, group=group)
        data = data.reset_index(drop=True)

        # Check for problematic configurations
        # TODO: test this logic
        folds_are_deterministic = (
            not (self.shuffle_folds and self.incremental) # Otherwise, always random
            and (
                # folds are fixed
                self.test_fold is not None
                # samples/groups will get assigned to folds deterministically
                or not self.shuffle
                # groups get assigned deterministically and there are no other
                # assignments for shuffling to effect
                or (not self.preserve_order and group is not None and stratum is None)
            )
        )
        if self.n_repeats > 1 and folds_are_deterministic:
            print(
                'WARNING | Repeating CV but results are expected to be redundant'
                'during each repitition for current configuration. Consider '
                'enabling `shuffle` and `preserve_order`.'
            )
        if self.test_fold is not None:
            if 'group' in col_names:
                print('WARNING | Predefined folds overriding requested groups')
            if 'stratum' in col_names:
                print('WARNING | Predefined folds overriding requested stratum')

        ########################################
        # Main loop
        for i_rep in range(self.n_repeats):
            if self.n_repeats > 1:
                print('DEBUG | Repetition %d of cross-validation' % i_rep)
            if self.shuffle:
                data = data.sample(frac=1)
            yield from self._single_split(data, col_names)
        ########################################

    def _single_split(self, data, col_names):
        if self.test_fold is None:
            print('DEBUG | Determining test fold')
            test_fold = self._determine_test_fold(data, **col_names)['fold_id']
        else:
            print('DEBUG | Using predefined test folds')
            test_fold = self.test_fold if isinstance(self.test_fold, pd.Series) else pd.Series(self.test_fold)

        if self.shuffle_folds:
            print('DEBUG | Shuffling folds')
            # Remap fold ids to new ones except for -1 which does not change as
            # it indicates samples to be included in all training folds
            remap = {
                old_fold_id : new_fold_id for new_fold_id, old_fold_id
                in enumerate(
                    self.rng.permutation(
                        [x for x in test_fold.unique() if x != -1]
                    )
                )
            }
            remap[-1] = -1
            print('DEBUG | remapped folds: %s' % remap)
            test_fold = np.vectorize(remap.__getitem__)(test_fold)

        cv = PredefinedSplit(test_fold)
        cv_iter = cv.split()

        if self.incremental:
            print('DEBUG | Transforming to incremental iteration')
            # TODO: Implement other TimeSeriesSplit features: max_train_size, gap
            test_iter = (test_idxs for _, test_idxs in cv_iter)
            test_iter1, test_iter2 = itertools.tee(test_iter)
            n_folds = cv.get_n_splits()
            cv_iter = (
                (train, test) for train, test in zip(
                    itertools.accumulate(itertools.islice(test_iter1, 0, n_folds-1), np.hstack),
                    itertools.islice(test_iter2, 1, n_folds)
                )
            )
        return cv_iter

    def _determine_test_fold(
        self,
        data    : pd.DataFrame,
        stratum : str = None,
        group   : str = None,
    ) -> pd.DataFrame:
        using_stratified_folds = stratum is not None
        using_group_folds      = group is not None
        dummy_X = range(len(data))

        if not (using_group_folds or using_stratified_folds):
            print('INFO | Using KFold')
            cv = KFold(n_splits=self.n_splits)
            cv_iter = cv.split(X=dummy_X)
            return pd.DataFrame(
                data  = cv_iter_to_test_fold_id(cv_iter),
                # data  = self.rng.permutation(np.arange(len(data)) % self.n_splits),
                columns = ['fold_id'],
                index = data.index,
            )

        col_names = []
        if using_group_folds:
            print('INFO | Using group folding over %s' % group)
            col_names.append(group)
        if using_stratified_folds:
            print('INFO | Using stratified folding over %s' % stratum)
            col_names.append(stratum)

        test_folds = data[col_names].copy()
        if using_group_folds:
            if not self.preserve_order:
                if using_stratified_folds:
                    print('DEBUG | Using StratifiedGroupKFold')
                    cv = StratifiedGroupKFold(n_splits=self.n_splits)
                    cv_iter = cv.split(X=dummy_X, y=test_folds[stratum], groups=test_folds[group])
                    fold_id = cv_iter_to_test_fold_id(cv_iter)
                    # grp = test_folds.groupby('stratum_id', sort=False)
                    # fold_id = grp.cumcount() % grp.ngroups
                else:
                    print('DEBUG | Using GroupKFold')
                    cv = GroupKFold(n_splits=self.n_splits)
                    cv_iter = cv.split(X=dummy_X, groups=test_folds[group])
                    fold_id = cv_iter_to_test_fold_id(cv_iter)
            # Any reason to have this option?
            # elif self.random_group_balance:
            #     if using_stratified_folds:
            #         raise NotImplementedError()
            #     else:
            #         fold_id = test_folds.groupby(['stratum_id', group], sort=False).ngroup() % self.n_splits
            else: # self.preserve_order
                print('DEBUG | Assigning groups while preserving order')
                if using_stratified_folds:
                    if test_folds.groupby(group, sort=False)[stratum].nunique().max() > 1:
                        print(
                            'WARNING | Order preserving stratified group fold '
                            'assumes entries in a group do not need to be '
                            'balanced across strata. It is likely groups will '
                            'be split across folds.'
                        )
                    fold_id = (test_folds
                        .groupby(stratum, sort=False)
                        [group]
                        .pipe(assign_groups_to_folds, n_splits=self.n_splits)
                    )
                else:
                    fold_id = assign_groups_to_folds(test_folds['groups'], self.n_splits)
        elif using_stratified_folds:
            print('DEBUG | Using StratifiedKFold')
            cv = StratifiedKFold(n_splits=self.n_splits)
            cv_iter = cv.split(X=dummy_X, y=test_folds[stratum])
            fold_id = cv_iter_to_test_fold_id(cv_iter)

        test_folds['fold_id'] = fold_id

        return test_folds

    #TODO: def get_n_splits(self):

def cv_iter_to_test_fold_id(cv_iter):
    return pd.concat([
        pd.Series(fold_id, index=test_idx)
        for fold_id, (_, test_idx) in enumerate(cv_iter)
    ]).sort_index().to_numpy()

def assign_groups_to_folds(group: pd.Series, n_splits: int) -> pd.Series:
    ideal_quantiles = np.linspace(0,1,n_splits+1)[1:-1]

    # Find slice points that achieve close to an even balance across splits
    # while keeping groups together and preserving data order
    cumsum         = group.groupby(group, sort=False).size().cumsum()
    cdf            = cumsum/cumsum.iloc[-1]
    right_idx      = np.searchsorted(cdf, ideal_quantiles)
    left_idx       = right_idx-1
    right_diff     = cdf.iloc[right_idx] - ideal_quantiles
    left_diff      = ideal_quantiles - cdf.iloc[left_idx]
    # TODO: Technically this will not always generate the best possible splits as it
    # considers what to do at each boundary point without considering the others
    # when that may be relevent. Is there a better way?
    nearest_idx    = np.where(left_diff.values < right_diff.values, left_idx, right_idx)
    best_quantiles = [0] + cdf.iloc[nearest_idx].tolist() + [1]

    # Assign fold IDs to bins
    group_to_fold_id = pd.cut(cdf, bins = best_quantiles, labels=range(len(best_quantiles)-1))

    # Return entries with
    return group.replace(group_to_fold_id).rename('fold_id')

def _require_dataframe(df = None, **columns) -> tuple[pd.DataFrame, list]:
    '''
    Take several possible inputs and always return a dataframe with it's column
    names.
    '''
    if isinstance(df, pd.DataFrame):
        col_names = {k:v for k,v in columns.items() if v is not None}
        if len(col_names) == 0:
            # Case 1) DataFrame and no column names implying index requested
            df = df.index.to_frame()
        else:
            # Case 2) DataFrame and it's column names passed
            df = df[list(col_names.values())]
    elif df is None:
        # Case 3) Separate arrays or Series passed for each column
        df, col_names = pd.DataFrame(columns), list(columns.keys())
    else:
        raise TypeError(f'Expected DataFrame but got {type(df)}')
    return df, col_names
