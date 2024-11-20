# Standard library
from pathlib import Path

# 3rd party
import numpy as np
import pandas as pd
import pytest
from sklearn import linear_model, tree
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.datasets import make_classification, make_regression

# 1st party
from skutils.persistance import dump_model, load_model


################################################################################
@pytest.mark.parametrize(
    'fmt,model', [
    # TODO Loop over multiple combinations of models and hyperparameters
    ('.pkl'   , tree.DecisionTreeClassifier()),
    ('.joblib', tree.DecisionTreeClassifier(max_depth=3)),
    ('.pkl'   , linear_model.LinearRegression()),
    ('.onnx'  , tree.DecisionTreeClassifier()),
    ]
)
def test_dump_model_pickle(
    model    : BaseEstimator,
    fmt      : str,
    tmp_path : Path
):
    data_kwargs = {'n_samples' : 10, 'n_features' : 4, 'random_state' : 1}

    # Generate data
    if is_regressor(model):
        X, y = make_regression(n_targets=1, **data_kwargs)
    elif is_classifier(model):
        X, y = make_classification(n_classes=2, **data_kwargs)
    else:
        raise NotImplementedError(f'Model type: {type(model)}')
    X = pd.DataFrame(X)
    y = pd.Series(y)

    model.fit(X, y)
    pred_before = model.predict(X)

    opath = tmp_path / f'model{fmt}'
    dump_model(model, opath, X = X if fmt == '.onnx' else None)
    model_after = load_model(opath)

    pred_after = model_after.predict(X)
    np.testing.assert_array_equal(pred_before, pred_after)
