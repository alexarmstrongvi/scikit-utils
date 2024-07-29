"""
Utilities for predicting with estimators
"""
# Standard library
from collections.abc import Iterable
import logging
from pathlib import Path
import time

# 3rd party
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import sklearn
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.metrics import get_scorer
from sklearn.metrics._scorer import _Scorer

# Globals
log = logging.getLogger(Path(__file__).stem)

################################################################################
def predict(
    estimator: BaseEstimator,
    X        : DataFrame,
    proba    : bool = True,
    simplify : bool = False,
) -> DataFrame:
    """Wrapper around estimator predict/predict_proba that preserves pandas data types

    Parameters
    ==========
    estimator:
        Estimator for making predictions
    X:
        Data to get predictions for
    proba:
        Get proba() outputs if possible
    simplify:
        Retain predict() or, if possible, proba() outputs but not both.
        Only applies to estimators with proba() method when `proba` = True.
    """
    if proba and hasattr(estimator, 'predict_proba'):
        classes = tuple(estimator.classes_)
        pred = (
            pd.DataFrame(
                estimator.predict_proba(X),
                index   = X.index,
                columns = [f'prob({c})' for c in classes],
            )
        )
        if not simplify:
            target_dtype = get_estimator_predict_dtype(estimator)
            pred['predict'] = (pred
                .idxmax(axis=1)
                .map(dict(zip(pred.columns, classes)))
                .astype(target_dtype)
            )
        elif len(classes) == 2:
            # TODO: Allow user to configure which class corresponds to prob=1
            # col_idx = classes.index(positive_class)]
            col_idx = 0
            pred = pred.iloc[:,[col_idx]]
        else:
            raise NotImplementedError('Multiclass')
    else:
        pred = pd.DataFrame(
            estimator.predict(X),
            index   = X.index,
            columns = ['predict'],
            dtype = get_estimator_predict_dtype(estimator),
        )
    return pred

def predict_and_score(
    estimator: BaseEstimator,
    X: DataFrame,
    y: Series,
    scoring: str | Iterable[str] | Iterable[_Scorer],
) -> tuple[DataFrame, Series]:
    """Predict on data and score predictions, returning results as pandas types"""
    start = time.perf_counter()
    y_pred = predict(estimator, X)
    scores = apply_scorers(y, y_pred, scoring)
    scores['score_time'] = time.perf_counter() - start
    scores = pd.Series(scores)
    return y_pred, scores

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

def combine_predictions(predictions):
    target_dtype = predictions[(predictions.columns.levels[0][0], 'predict')].dtype
    return (predictions
        .stack(['split', 'pred'], future_stack=True)
        .dropna()
        .unstack('pred')
        # Option 1) Keep split as index (helpful if test folds overlap)
        # <NoOp>
        # Option 2) Keep split as column
        #.reset_index(level='split')
        # Option 3) Drop split
        #.reset_index(level='split', drop=True)
        .astype({'predict' : target_dtype})
    )

def get_estimator_predict_dtype(estimator):
    if is_classifier(estimator):
        target_dtype = estimator.classes_.dtype
        if pd.api.types.is_string_dtype(target_dtype):
            target_dtype = pd.CategoricalDtype(estimator.classes_)
    else:
        if not is_regressor(estimator):
            log.warning(
                'Unable to determine if estimator is classifier or regressor. '
                'Defaulting output type to float64. '
                'For custom estimators, define the _estimator_type attribute.'
            )
        target_dtype = np.float64
    return target_dtype
