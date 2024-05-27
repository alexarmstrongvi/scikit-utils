#!/usr/bin/env python
"""
Configurable script for fitting supervised models
"""
# Standard library
import argparse
import logging
from pathlib import Path
import time
from typing import (
    Any,
    Collection,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Union
)

# 3rd party
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator
from sklearn.metrics import get_scorer
import sklearn.model_selection
from sklearn.model_selection import cross_validate

# 1st party
from skutils.data import get_default_cfg_supervised

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

def fit_supervised_model(data: DataFrame, cfg: dict) -> dict:
    icfg = cfg['inputs']
    ocfg = cfg['outputs']

    # reformat_cfg()
    ########################################
    # Setup
    log.info('Setting up for fit')
    random_gen   = np.random.default_rng(cfg['random_seed'])
    random_state = np.random.RandomState(random_gen.bit_generator)
    estimator    = build_estimator(**cfg['estimator'])
    cv           = build_cv_iterator(**cfg['cv_iterator'])

    n_splits      = cv.get_n_splits()
    split_names   = list(range(n_splits))

    ########################################
    log.info('Reading in data')
    # data = read_data(**icfg)
    if icfg['index'] is not None:
        data = data.set_index(icfg['index'])
    else:
        data = data.rename_axis(index='index')
    X, y = preprocess_data(
        data,
        features = icfg['features'],
        target   = icfg['target'],
        **cfg['preprocess']
    )
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        random_state=random_state,
        **cfg['train_test_split'],
    )
    return_test_predictions = X_test is not None and y_test is not None

    ########################################
    if cfg['model_selector'] is None:
        log.info('Running cross-validation')
        fit_results = cross_validate(
            estimator,
            X_train,
            y_train,
            cv=cv,
            return_train_score = ocfg['save_cv_train_scores'],
            return_estimator   = ocfg['save_cv_estimators'],
            return_indices     = ocfg['save_cv_indices'],
            **cfg['fit']
        )
        fit_results, is_test_data = _reformat_cross_validate_results(
            fit_results,
            index = X_train.index,
            columns = split_names,
        )
    else:
        # TODO: Implement evaluation of multiple models with different
        # parameters. Not possible to save out as much information so the
        # outputs will be quite different for this use case
        model_selector = build_model_selector(estimator, **cfg['model_selector'])
        results_per_model = model_selector.fit(X_train, y_train, cv=cv, **cfg['fit'])

    log.info('Fitting estimator on all training data')
    results_all : Dict[str, Any] = {}
    start = time.perf_counter()
    estimator_all = estimator.fit(X_train, y_train)
    results_all['fit_time'] = time.perf_counter() - start
    # TODO: Does this need to be cloned to avoid it getting updated if
    # estimator_all.fit is called again?
    results_all['estimator'] = estimator_all

    y_train_pred = y_cv_test_pred = None
    if ocfg['save_train_predictions']:
        log.info('Getting training data predictions')
        y_train_pred = predict(
            fit_results['estimator'].to_list(),
            X_train,
            keys  = fit_results.index.to_list(),
            names = ['split'],
        )
        y_train_all_pred, train_all_score = predict_and_score(estimator_all, X_train, y_train, scoring=cfg['fit']['scoring'])
        y_train_pred = pd.concat(
            [
                y_train_pred,
                y_train_all_pred.pipe(
                    add_index_level,
                    label = 'all',
                    name  = y_train_pred.columns.names[0],
                    axis  = 1,
                ),
            ],
            axis=1,
        )
        results_all.update({
            # Add prefix to metrics columns (e.g. train_accuracy) but not others
            # (e.g. score_time)
            f'train_{k}' if k not in fit_results else k : v
            for k,v in train_all_score.items()
        })
        # is_partitioned_by_val_folds = is_test_data is not None and (is_test_data.sum(axis=1) == 1).all()
        # TODO: Handle user only wanting to save validation predictions and not
        # full training predictions
        if ocfg['save_cv_test_predictions']:
            y_cv_test_pred = combine_test_results(
                y_train_pred[split_names],
                is_test_data[split_names],
            )

    y_test_pred = None
    if return_test_predictions:
        log.info('Getting test data predictions')
        y_test_pred, test_score = predict_and_score(estimator_all, X_test, y_test, scoring=cfg['fit']['scoring'])
        results_all.update({
            # Add prefix to metrics columns (e.g. test_accuracy) but not others
            # (e.g. score_time)
            f'test_{k}' if k not in fit_results else k : v
            for k,v in test_score.items()
        })
    fit_results.loc['all'] = pd.Series(results_all)
    del results_all

    feature_importances = None
    if ocfg['save_feature_importances']:
        feature_importances = pd.DataFrame(
            np.column_stack([est.feature_importances_ for est in fit_results['estimator']]),
            index = feature_names,
            columns = fit_results.index,
        ).sort_values('all', ascending=False)

    return {k:v for k,v in {
        'fits'                : fit_results,
        'is_test_data'        : is_test_data,
        'y_train_pred'        : y_train_pred,
        'y_cv_test_pred'      : y_cv_test_pred,
        'y_test_pred'         : y_test_pred,
        'feature_importances' : feature_importances,
    }.items() if v is not None}


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

ScoringType = Union[str, Iterable[str]]
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
    scoring: Union[str, Iterable[str], Iterable[_Scorer]],
) -> Tuple[DataFrame, Series]:
    start = time.perf_counter()
    y_pred = predict(estimator, X)
    scores = apply_scorers(y, y_pred, scoring)
    scores['score_time'] = time.perf_counter() - start
    scores = pd.Series(scores)
    return y_pred, scores

def train_test_split(X, y, random_state, **kwargs):
    """Wrapper around sklearn's train_test_split that handles user
    requesting no train test splitting"""
    if 'test_size' not in kwargs and 'train_size' not in kwargs:
        log.debug('No test set being set aside for evaluation')
        X_train, y_train = X, y
        X_test = y_test = None
    # TODO: Allow user to manually choose test set via column or callable
    else:
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            X, y,
            random_state=random_state,
            # TODO: Enable use of stratify
            **kwargs,
        )
        log.debug('Split data into train (%.0f%%) and test (%.0f%%) sets',
            len(X_train)/len(X) * 100,
            len(X_test)/len(X) * 100,
        )
    return X_train, X_test, y_train, y_test

# TODO: Make a generic predict that accepts a single model or iterable of
# models. Allow user to provide 'keys' and 'names' or just **kwargs for concat.
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

def combine_test_results(results, is_test_data):
    target_dtype = results[(results.columns.levels[0][0], 'predict')].dtype
    return (results
        [is_test_data]
        .stack(['split', 'prob/predict'], future_stack=True)
        .dropna()
        .unstack('prob/predict')
        # Option 1) Keep split as index (helpful if test folds overlap)
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

def _reformat_cross_validate_results(
    results : dict,
    index : pd.Index,
    columns : List[Any],
) -> Tuple[DataFrame, Optional[DataFrame]]:
    is_test_data = None
    if 'indices' in results:
        n_splits = len(results['fit_time'])
        is_test_data = DataFrame(
            None,
            index = index,
            columns = columns,
            dtype = 'bool',
        ).rename_axis(index = index.name, columns='split')
        for i in range(n_splits):
            is_test_data.iloc[results['indices']['train'][i], i] = False
            is_test_data.iloc[results['indices']['test'][i], i] = True

    results_df = DataFrame(
        data = {k : v for k,v in results.items() if k != 'indices'},
    ).rename_axis(index='split')
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
    features: Optional[Collection[str]] = None,
) -> Tuple[DataFrame, Series]:
    '''Preprocess the raw data for model fitting

    This SHOULD NOT INVOLVE data-dependent transformations (e.g. interpolate
    missing data) in order to avoid data leakage. Such transformations should be
    part of the estimator pipeline created in build_estimator'''
    if features is None:
        features = X.columns.drop(target)
    X = data[features]
    y = data[target]
    return X, y

def build_estimator(name: str, **kwargs) -> BaseEstimator:
    return import_object(name)(**kwargs)

# Standard library
import importlib
from typing import Type

_MODULE_MAP = {
    # Classifiers
    'ExtraTreesClassifier' : 'sklearn.ensemble',
    # CV Iterators
    'KFold' : 'sklearn.model_selection',
}
def import_object(name: str) -> Type:
    if name in _MODULE_MAP:
        name = f'{_MODULE_MAP[name]}.{name}'
    else:
        log.debug('Importing unknown object: %s', name)
    module_name, class_name = name.rsplit('.', maxsplit=1)
    log.debug('Importing object: `from %s import %s`', module_name, class_name)
    return getattr(importlib.import_module(module_name), class_name)

# 3rd party
from numpy.typing import NDArray
from sklearn.model_selection import BaseCrossValidator, BaseShuffleSplit

NDArrayInt = NDArray[np.int64]
def build_cv_iterator(
    name : str,
    **kwargs,
) -> Union[BaseCrossValidator, Iterable[Tuple[NDArrayInt, NDArrayInt]], BaseShuffleSplit, None]:
    return import_object(name)(**kwargs)

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