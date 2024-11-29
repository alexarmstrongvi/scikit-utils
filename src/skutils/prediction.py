"""
Utilities for predicting with estimators
"""
# Standard library
import logging
from pathlib import Path
from typing import Any

# 3rd party
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.base import BaseEstimator, is_classifier, is_regressor

# Globals
log = logging.getLogger(Path(__file__).stem)

################################################################################
def predict(
    estimator: BaseEstimator,
    X        : DataFrame,
    proba    : bool = True,
    simplify : bool = False,
    pos_label: Any  = None,
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
        (only for estimators with proba() and when `proba` = True) Retain
        predict() or, if possible, proba() outputs but not both.
    pos_label:
        (only for classifiers with proba()) Positive label whose prob output is
        retained. If None, then clf.classes_[0] is retained.
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
            if pos_label is None:
                pos_label = classes[0]
            pred = pred.loc[:,[f'prob({pos_label})']]
        else:
            raise NotImplementedError('Multiclass')
    else:
        pred = pd.DataFrame(
            estimator.predict(X),
            index   = X.index,
            columns = ['predict'],
            dtype = get_estimator_predict_dtype(estimator),
        )
    # TODO: return y_pred, y_proba so that there no longer needs to be a
    # convention over the column names
    return pred

def combine_predictions(predictions):
    comb = (predictions
        .stack(['split', 'pred'], future_stack=True)
        .dropna()
        .unstack('pred')
        # Option 1) Keep split as index (helpful if test folds overlap)
        # <NoOp>
        # Option 2) Keep split as column
        #.reset_index(level='split')
        # Option 3) Drop split
        #.reset_index(level='split', drop=True)
    )
    if 'predict' in comb.columns:
        target_dtype = predictions[(predictions.columns.levels[0][0], 'predict')].dtype
        comb = comb.astype({'predict' : target_dtype})
    return comb

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
