#!/usr/bin/env python
"""
Configurable script for fitting supervised models
"""
# Standard library
import argparse
from collections.abc import Collection, Iterable
import logging
from pathlib import Path
import time
from typing import (
    # TODO: Update to 3.9 type hints
    Any,
    Literal,
    NotRequired,
    TypedDict,
)

# 3rd party
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_validate

# 1st party
from skutils.data import get_default_cfg_supervised
from skutils.import_utils import import_object
from skutils.train_test_iterators import TrainOnAllWrapper, TrainTestIterable

# Globals
log = logging.getLogger(Path(__file__).stem)

# TODO: Update to work on multi-class problems
# TODO: Update to work on multi-label problems
# TODO: Update to work on multi-class & multi-label problems
# TODO: Update to work on regression problems
# TODO: Update to enable hyperparameter searching
################################################################################
def _get_test_inputs():
    cfg = get_default_cfg_supervised()

    # 3rd party
    import sklearn.datasets
    data_bunch = sklearn.datasets.load_breast_cancer(as_frame=True)
    data = data_bunch['frame']
    cfg['inputs'].update({
        'features' : data_bunch['feature_names'].tolist(),
        'target'   : 'target',
    })
    # Test 1: Binary target
    # Test 2: Target labels
    data['target'] = (data['target']
        .map({0:'NoCancer',1:'HasCancer'})
        .astype(pd.CategoricalDtype(['NoCancer','HasCancer']))
    )

    return cfg, data

def main():
    # args = parse_argv()
    # cfg  = get_config(args.configs)

    # icfg = cfg['inputs']
    # ocfg = cfg['outputs']
    # odir = Path(ocfg['path'])

    # scripting.require_empty_dir(ocfg['path'])
    # configure_logging(**ocfg['logging'])

    logging.basicConfig(
        level = 'DEBUG',
        format = '%(levelname)8s | %(module)s :: %(message)s',
        force = True,
    )
    logging.getLogger('asyncio').setLevel('WARNING')
    cfg, data = _get_test_inputs()
    results = fit_supervised_model(data, cfg)

    # log.info('Saving results')
    # save_results(results)
    # save_visualizations(results)

class FitSupervisedModelResults(TypedDict):
    fits                : NotRequired[DataFrame]
    is_test_data        : NotRequired[DataFrame]
    y_pred              : NotRequired[DataFrame]
    feature_importances : NotRequired[Series]

def fit_supervised_model(data: DataFrame, cfg: dict) -> FitSupervisedModelResults:
    ocfg = cfg['outputs']

    # reformat_cfg()
    ########################################
    # Checks
    if not any(ocfg.values()):
        log.info('All outputs disabled')

    ########################################
    # Setup
    log.info('Setting up for fit')
    random_gen   = np.random.default_rng(cfg['random_seed'])
    random_state = np.random.RandomState(random_gen.bit_generator)

    train_test_iterator = build_train_test_iterator(
        **cfg['train_test_iterator'],
        train_on_all = cfg['train_on_all'],
        random_state=random_state,
    )
    estimator = build_estimator(**cfg['estimator'])

    ########################################
    log.info('Reading in data')
    # data = read_data(icfg['input'])
    X, y = preprocess_data(data, **cfg['preprocess'])
    feature_names = X.columns.tolist()

    ########################################
    y_pred = None
    is_test_data = None
    feature_importances = None
    if cfg['model_selector'] is not None:
        # TODO: Implement evaluation of multiple models with different
        # parameters. Not possible to save out as much information so the
        # outputs will be quite different for this use case
        model_selector = build_model_selector(estimator, **cfg['model_selector'])
        model_selector.fit(X, y, cv=train_test_iterator, **cfg['fit'])
        # fit_results.update(...)
    else:
        log.info('Running training and testing')
        save_predictions  = ocfg['save_train_predictions'] or ocfg['save_test_predictions']
        return_estimators = ocfg['save_estimators'] or ocfg['save_feature_importances'] or save_predictions
        return_indices    = ocfg['save_indices'] or save_predictions
        # TODO: Implement general train_test function that has the
        # parallelization of cross_validate but allows the user full
        # configuration to, say, save out train predictions and scores
        # without evaluating on test set.
        fit_results = cross_validate(
            estimator,
            X,
            y,
            cv                 = train_test_iterator,
            return_train_score = ocfg['save_train_scores'],
            return_estimator   = return_estimators,
            return_indices     = return_indices,
            **cfg['fit'],
        )

        if not ocfg['save_test_scores']:
            # cross_validate always saves out test scores so for now just delete
            # these if not requested.
            fit_results = {k:v for k,v in fit_results.items() if not k.startswith('test_')}
            if not ocfg['save_train_scores']:
                del fit_results['score_time']

        n_splits = len(fit_results['fit_time'])
        # TODO: Allow user to provide format string (e.g. "Split {i}")
        split_names = cfg['split_names'] or list(range(n_splits))
        if cfg['train_on_all']:
            split_names[-1] = 'all'
        fit_results, is_test_data = to_cross_validate_dataframes(
            fit_results,
            data_index = X.index,
            split_names = split_names,
        )

        if save_predictions:
            assert is_test_data is not None
            save_only_train_pred = not ocfg['save_test_predictions']
            save_only_test_pred  = not ocfg['save_train_predictions']

            if save_only_test_pred:
                masks = (mask for _, mask in is_test_data.items())
            elif save_only_train_pred:
                masks = (~mask for _, mask in is_test_data.items())
            else:
                masks = (mask.notna() for _, mask in is_test_data.items())

            y_pred = pd.concat(
                [
                    predict(est, X[mask])
                    for est, mask in zip(fit_results['estimator'], masks)
                ],
                keys = split_names,
                names = ['split'],
                axis=1,
            )

            if cfg['stack_split_predictions']:
                # Ensure 1 column of results per stack
                is_notna = y_pred.stack('prob/predict', future_stack=True).notna()
                if check_is_partition(is_notna):
                    y_pred = combine_predictions(y_pred)

    if ocfg['save_feature_importances']:
        feature_importances = pd.DataFrame(
            np.column_stack([est.feature_importances_ for est in fit_results['estimator']]),
            index = feature_names,
            columns = fit_results.index,
        )
        # Sort features
        if 'all' in feature_importances:
            feature_importances = feature_importances.sort_values('all', ascending=False)
        else:
            sorted_idxs = feature_importances.median(axis=1).sort_values(ascending=False).index
            feature_importances = feature_importances.loc[sorted_idxs]

    if not ocfg['save_estimators'] and 'estimator' in fit_results:
        # Estimators only returned to get predictions and/or feature importances
        fit_results = fit_results.drop(columns='estimator')

    if not ocfg['save_indices']:
        # Indices only returned to get predictions
        is_test_data = None

    if ocfg['save_indices'] and sum(ocfg.values()) == 1:
        # Only save_indices. Uncommon use case so not putting in effort to make
        # this efficient.
        fit_results = y_pred = feature_importances = None

    results = {k:v for k,v in {
        'fits'                : fit_results,
        'is_test_data'        : is_test_data,
        'y_pred'              : y_pred,
        'feature_importances' : feature_importances,
    }.items() if v is not None}

    return FitSupervisedModelResults(**results)

def check_is_partition(df: DataFrame) -> bool:
    return df.sum(axis=1).unique().tolist() == [1]

def _predict(
    estimator: BaseEstimator,
    X        : DataFrame,
    proba    : bool = True,
    simplify : bool = False,
) -> DataFrame:
    """Wrapper around estimator predict/predict_proba that preserves pandas data types"""
    # TODO: Generalize to work with regression
    target_dtype = estimator.classes_.dtype
    if pd.api.types.is_string_dtype(target_dtype):
        target_dtype = pd.CategoricalDtype(estimator.classes_)
    if proba and hasattr(estimator, 'predict_proba'):
        pred = (
            pd.DataFrame(
                estimator.predict_proba(X),
                index   = X.index,
                columns = estimator.classes_,
                # Would this be better?
                # columns = [f'prob_{c}' for c in estimator.classes_],
            )
            .rename_axis(columns='prob/predict')
        )
        if not simplify:
            pred['predict'] = pred.idxmax(axis=1).astype(target_dtype)
        elif estimator.n_classes_ == 2:
            # TODO: Allow user to configure which class corresponds to prob=1
            pred = pred[[estimator.classes_[0]]]
    else:
        pred = pd.DataFrame(
            estimator.predict(X),
            index   = X.index,
            columns = ['predict'],
            dtype   = target_dtype,
        )
    return pred

# 3rd party
from sklearn.metrics._scorer import _Scorer

ScoringType = str | Iterable[str]
def apply_scorers(
    y_true: Series,
    y_pred: Series,
    scoring: ScoringType,
):
    scores = {}
    if isinstance(scoring, str):
        scoring = [scoring]
    scorers = {s : get_scorer(s) if isinstance(s,str) else s for s in scoring}

    for name, scorer in scorers.items():
        if scorer._response_method == 'predict':
            score = apply_scorer(y_true, y_pred['predict'], scorer)
        elif scorer._response_method == 'predict_proba':
            score = apply_scorer(y_true, y_pred.drop(columns='predict'), scorer)#[estimator.classes_])
        scores[name] = score
    return scores

def apply_scorer(y_true, y_pred, scorer: _Scorer, **kwargs):
    # Copy of part of _scorer.py:_Scorer._score() method
    scoring_kwargs = {**scorer._kwargs, **kwargs}
    return scorer._sign * scorer._score_func(y_true, y_pred, **scoring_kwargs)

def predict_and_score(
    estimator: BaseEstimator,
    X: DataFrame,
    y: Series,
    scoring: str | Iterable[str] | Iterable[_Scorer],
) -> tuple[DataFrame, Series]:
    start = time.perf_counter()
    y_pred = predict(estimator, X)
    scores = apply_scorers(y, y_pred, scoring)
    scores['score_time'] = time.perf_counter() - start
    scores = pd.Series(scores)
    return y_pred, scores

def predict(
    estimators,
    X : DataFrame,
    proba    : bool = True,
    simplify : bool = False,
    **kwargs,
) -> DataFrame:
    if isinstance(estimators, BaseEstimator):
        return _predict(estimators, X, proba, simplify)

    if isinstance(estimators, dict):
        if 'keys' not in kwargs:
            kwargs['keys'] = list(estimators.keys())
        estimators = list(estimators.values())
    return pd.concat(
        [_predict(est, X) for est in estimators],
        axis = 1,
        **kwargs,
    )

def combine_predictions(predictions):
    target_dtype = predictions[(predictions.columns.levels[0][0], 'predict')].dtype
    return (predictions
        .stack(['split', 'prob/predict'], future_stack=True)
        .dropna()
        .unstack('prob/predict')
        # Option 1) Keep split as index (helpful if test folds overlap)
        # <NoOp>
        # Option 2) Keep split as column
        #.reset_index(level='split')
        # Option 3) Drop split
        #.reset_index(level='split', drop=True)
        .astype({'predict' : target_dtype})
    )

def add_index_level(
    df   : DataFrame,
    label: str,
    name : str,
    axis : Literal[0,1] = 0,
    level: int = 0
) -> DataFrame:
    index  = df.axes[axis]
    level_labels = [index.to_list()]
    names  = list(index.names)
    level_labels.insert(level, [label])
    names.insert(level, name)
    return df.set_axis(
        labels = pd.MultiIndex.from_product(level_labels, names = names),
        axis = axis,
    )

def to_cross_validate_dataframes(
    results : dict,
    data_index : pd.Index,
    split_names : list[Any],
) -> tuple[DataFrame, DataFrame | None]:
    """Convert cross_validate results into pandas DataFrames"""
    is_test_data = None
    if 'indices' in results:
        n_splits = len(results['fit_time'])
        is_test_data = DataFrame(
            None,
            index = data_index,
            columns = split_names,
            dtype = 'bool',
        ).rename_axis(index = data_index.name, columns='split')
        for i in range(n_splits):
            is_test_data.iloc[results['indices']['train'][i], i] = False
            is_test_data.iloc[results['indices']['test'][i], i] = True

    results_df = (
        DataFrame(
            data = {k : v for k,v in results.items() if k != 'indices'},
        )
        .rename_axis(index='split')
        .rename(split_names.__getitem__)
    )
    return results_df, is_test_data

def read_data(
    path: Path,
    **kwargs,
) -> DataFrame:
    if path.suffix == '.csv':
        data = pd.read_csv(path, **kwargs)
    elif path.suffix in ('.hdf', '.hdf5'):
        data = pd.read_hdf(path, **kwargs)
    else:
        raise NotImplementedError(
            f'No reader implemented for files of type {path.suffix}'
        )
    return data

def preprocess_data(
    data: pd.DataFrame,
    *,
    target: str,
    features: Collection[str] | None = None,
    index: Any = None,
) -> tuple[DataFrame, Series]:
    '''Preprocess the raw data for model fitting

    This SHOULD NOT INVOLVE data-dependent transformations (e.g. interpolate
    missing data) in order to avoid data leakage. Such transformations should be
    part of the estimator pipeline created in build_estimator
    '''

    if index is not None:
        data = data.set_index(index)
    else:
        data = data.rename_axis(index='index')

    if features is None:
        features = data.columns.drop(target)
        log.info('Features not specified. Using all but the target feature: %s', features.to_list())

    # TODO: Save any columns not in features or target to metadata and return?
    X = data[features]
    y = data[target]
    return X, y

def build_estimator(name: str, **kwargs) -> BaseEstimator:
    return import_object(name)(**kwargs)

# 3rd party
def build_train_test_iterator(
    name: str,
    train_on_all: bool = False,
    random_state: np.random.RandomState | None = None,
    **kwargs
) -> TrainTestIterable:
    tt_iter = None
    if name == 'train_test_split':
        tt_iter = import_object('skutils.train_test_iterators.TrainTestSplit')(**kwargs)
    else:
        try:
            if kwargs.get('shuffle') is True:
                # sklearn iterators raise a ValueError if random_state specified
                # but shuffle != True
                kwargs = {**kwargs, 'random_state' : random_state}
            tt_iter = import_object(name)(**kwargs)
        except ImportError:
            pass
    if tt_iter is None:
        raise NotImplementedError(f'name = {name}')

    if train_on_all:
        tt_iter = TrainOnAllWrapper(tt_iter)

    return tt_iter



# 3rd party
from sklearn.model_selection import GridSearchCV


def build_model_selector(estimator: BaseEstimator, param_grid: dict = None, **kwargs) -> BaseEstimator:
    return GridSearchCV(estimator, **kwargs)

# def get_config(args: argparse.Namespace):
#     default_cfg = load_default_config()
#     override_cfgs = (scripting.read_config(Path(p)) for p in args.configs)
#     cfg = scripting.merge_config_files(override_cfgs, default_cfg)
#     cfg = override_config(cfg, args)
#     validate_config(cfg)
#     return cfg

if __name__ == '__main__':
    main()