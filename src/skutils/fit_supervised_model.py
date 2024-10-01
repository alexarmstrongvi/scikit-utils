#!/usr/bin/env python
"""
Utilities for fitting supervised models
"""
# Standard library
from collections.abc import Collection
import logging
from typing import Any, TypedDict

# 3rd party
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, cross_validate

# 1st party
from skutils.import_utils import import_object
from skutils.pipeline import make_pipeline
from skutils.prediction import combine_predictions, predict
from skutils.train_test_iterators import TrainOnAllWrapper, TrainTestIterable

# Globals
log = logging.getLogger(__name__)

################################################################################

class FitSupervisedModelResults(TypedDict, total=False):
    X                   : DataFrame
    y_true              : Series | DataFrame
    fits                : DataFrame
    is_test_data        : DataFrame
    y_pred              : DataFrame
    feature_importances : DataFrame

def fit_supervised_model(data: DataFrame, cfg: dict) -> FitSupervisedModelResults:
    '''Run full pipepline to fit supervised model to data'''
    # TODO: Split up cfg into kwargs once there is a stable format
    cfg_return = cfg['returns']
    n_outputs = sum(cfg_return.values())

    ########################################
    # Checks
    if not any(cfg_return.values()):
        log.info('All outputs disabled')

    ########################################
    # Setup
    log.info('Setting up for fit')
    random_gen   = np.random.default_rng(cfg['random_seed'])
    random_state = np.random.RandomState(random_gen.bit_generator)

    train_test_iterator = build_train_test_iterator(
        cfg = cfg['train_test_iterator'],
        random_state = random_state,
    )
    if cfg['train_on_all']:
        train_test_iterator = TrainOnAllWrapper(train_test_iterator)

    estimator = build_estimator(cfg['estimator'])

    ########################################
    log.info('Preprocessing data')
    X, y = preprocess_data(data, **cfg['preprocess'])
    feature_names = X.columns.tolist()

    if n_outputs == 1:
        if cfg_return['return_X']:
            return {'X' : X}
        elif cfg_return['return_y_true']:
            return {'y_true' : y}

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
        return_predictions  = cfg_return['return_train_predictions'] or cfg_return['return_test_predictions']
        return_estimators = cfg_return['return_estimators'] or cfg_return['return_feature_importances'] or return_predictions
        return_indices    = cfg_return['return_indices'] or return_predictions
        # TODO: Implement general train_test function that has the
        # parallelization of cross_validate but allows the user full
        # configuration to, say, save out train predictions and scores
        # without evaluating on test set.
        fit_results = cross_validate(
            estimator,
            X,
            y,
            cv                 = train_test_iterator,
            return_train_score = cfg_return['return_train_scores'],
            return_estimator   = return_estimators,
            return_indices     = return_indices,
            **(cfg['fit'] or {}),
        )

        if not cfg_return['return_test_scores']:
            # cross_validate always saves out test scores so for now just delete
            # these if not requested.
            fit_results = {k:v for k,v in fit_results.items() if not k.startswith('test_')}
            if not cfg_return['return_train_scores']:
                del fit_results['score_time']

        n_splits = len(fit_results['fit_time'])
        # TODO: Allow user to provide format string (e.g. "Split {i}")
        split_names = _get_split_names(cfg['split_names'], n_splits, cfg['train_on_all'])

        fit_results, is_test_data = to_cross_validate_dataframes(
            fit_results,
            data_index = X.index,
            split_names = split_names,
        )

        if return_predictions:
            assert is_test_data is not None
            return_only_train_pred = not cfg_return['return_test_predictions']
            return_only_test_pred  = not cfg_return['return_train_predictions']

            if return_only_test_pred:
                masks = (mask for _, mask in is_test_data.items())
            elif return_only_train_pred:
                masks = (~mask for _, mask in is_test_data.items())
            else:
                masks = (mask.notna() for _, mask in is_test_data.items())

            y_pred = pd.concat(
                [
                    predict(est, X[mask])
                    for est, mask in zip(fit_results['estimator'], masks)
                ],
                keys  = split_names,
                names = ['split', 'pred'],
                axis  = 1,
            )

            if cfg['stack_split_predictions']:
                # Ensure 1 column of results per stack
                is_notna = y_pred.stack('pred', future_stack=True).notna()
                if check_is_partition(is_notna):
                    y_pred = combine_predictions(y_pred)

    # TODO: Calculate this outside of fit_supervised_model() by returning estimators
    if cfg_return['return_feature_importances']:
        feature_importances = []
        for est in fit_results['estimator']:
            if hasattr(est, 'steps'): # It's a Pipeline:
                est = est.steps[-1][1]
            feature_importances.append(est.feature_importances_)
        feature_importances = pd.DataFrame(
            np.column_stack(feature_importances),
            index = feature_names,
            columns = fit_results.index,
        ).rename_axis(index='Feature')
        # Sort features
        if 'all' in feature_importances.columns:
            feature_importances = feature_importances.sort_values('all', ascending=False)
        else:
            sorted_idxs = feature_importances.median(axis=1).sort_values(ascending=False).index
            feature_importances = feature_importances.loc[sorted_idxs]

    if not cfg_return['return_estimators'] and 'estimator' in fit_results:
        # Estimators only returned to get predictions and/or feature importances
        fit_results = fit_results.drop(columns='estimator', errors='ignore')

    if not cfg_return['return_indices']:
        # Indices only returned to get predictions
        is_test_data = None

    if cfg_return['return_indices'] and n_outputs == 1:
        # Only return_indices. Uncommon use case so not putting in effort to make
        # this efficient.
        fit_results = y_pred = feature_importances = None

    results = {
        'X'            : X if cfg_return['return_X'] else None,
        'y_true'       : y if cfg_return['return_y_true'] else None,
        'fits'         : fit_results,
        'is_test_data' : is_test_data,
        'y_pred'       : y_pred,
        'feature_importances' : feature_importances,
    }
    results = {k:v for k,v in results.items() if v is not None}
    return FitSupervisedModelResults(**results)

################################################################################
def preprocess_data(
    data       : pd.DataFrame,
    *,
    target     : str,
    features   : Collection[str] | None = None,
    target_map : dict | None = None,
    astype     : dict | None = None,
    index      : Any = None,
) -> tuple[DataFrame, Series]:
    '''Preprocess the raw data for model fitting'''
    # This SHOULD NOT INVOLVE data-dependent transformations (e.g. interpolate
    # missing data) in order to avoid data leakage. Such transformations should be
    # part of the estimator pipeline created in build_estimator

    if astype is not None:
        data = data.astype(astype)

    if index is not None:
        data = data.set_index(index)
    else:
        data = data.rename_axis(index='index')

    if features is None:
        features = data.columns.drop(target)
        log.info('Features not specified. Using all but the target feature: %s', features.to_list())

    # NOTE: User can put any quickfixes here but in general scikit-learn expects
    # user to have done all general preprocessing prior to running pipeline. The
    # preprocessing that is part of the pipeline is reserved for steps that risk
    # data leakage and should be run separately in the case of cross validation.

    # TODO: Save any columns not in features or target to metadata and return?
    X = data[features]
    y = data[target]

    if target_map is not None:
        y = y.map(target_map)

    return X, y

def build_estimator(cfg: str | dict) -> BaseEstimator:
    '''Build sklearn estimators from a yaml compatible dictionary'''
    match cfg:
        case str():
            class_ = cfg
            estimator = import_object(class_)()
        case {'make_pipeline' : kwargs} | {'Pipeline' : kwargs}:
            estimator = make_pipeline(**kwargs)
        case dict() if len(cfg) == 1:
            class_, kwargs = tuple(cfg.items())[0]
            estimator = import_object(class_)(**kwargs)
    return estimator

def build_train_test_iterator(
    cfg: dict,
    random_state: np.random.RandomState | None = None,
) -> TrainTestIterable:
    '''Build sklearn train-test iterators from a yaml compatible dictionary'''
    name, kwargs = _parse_train_test_iterator_cfg(cfg)
    tt_iter = None
    if name == 'train_test_split':
        tt_iter = import_object('skutils.train_test_iterators.TrainTestSplit')(**kwargs)
    else:
        if kwargs.get('shuffle') is True:
            # sklearn iterators raise a ValueError if random_state specified
            # but shuffle != True
            kwargs = {**kwargs, 'random_state' : random_state}
        try:
            tt_iter = import_object(name)(**kwargs)
        except ImportError:
            pass
    if tt_iter is None:
        raise NotImplementedError(f'name = {name}')

    return tt_iter

def _parse_train_test_iterator_cfg(cfg: str | dict[str, dict]) -> tuple[str, dict[str, dict]]:
    if isinstance(cfg, str):
        name = cfg
        kwargs = {}
    else:
        name = next(iter(cfg.keys()))
        kwargs = cfg[name]
    return name, kwargs


def build_model_selector(estimator: BaseEstimator, param_grid: dict = None, **kwargs) -> BaseEstimator:
    return GridSearchCV(estimator, **kwargs)

################################################################################
def check_is_partition(df: DataFrame) -> bool:
    return df.sum(axis=1).unique().tolist() == [1]

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
            pd.NA,
            index = data_index,
            columns = split_names,
            dtype = 'boolean',
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

def _get_split_names(
    split_names  : list[Any] | str | None,
    n_splits     : int,
    train_on_all : bool,
) -> list[str | int]:
    if isinstance(split_names, list):
        for i in range(len(split_names)):
            if split_names[i] == 'all':
                log.warning(
                    '"all" is a reserved split name for train_on_all. '
                    'Updating user provided split name to "ALL"'
                )
                split_names[i] = "ALL"

    if isinstance(split_names, str):
        split_names = [split_names.format(i) for i in range(n_splits - train_on_all)]
    elif split_names is None:
        split_names = list(range(n_splits - train_on_all))
    elif len(split_names) + train_on_all < n_splits:
        log.warning(
            '%d split names provided for %d splits. Adding default names',
            len(split_names), n_splits
        )
        split_names += [f'Split {i}' for i in range(len(split_names), n_splits)]
    elif len(split_names) + train_on_all > n_splits:
        log.warning(
            '%d split names but only %d splits. Using first %d',
            len(split_names), n_splits, n_splits
        )
        split_names = split_names[:n_splits]

    if train_on_all:
        split_names.append('all')

    return split_names
