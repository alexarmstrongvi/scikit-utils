
# Standard library
import copy

# 3rd party
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

# 1st party
from skutils.bin.fit_supervised_model import fit_supervised_model
from skutils.data import get_default_cfg_supervised


def test_fit_supervised_model():
    data_cfg = dict(
        n_samples            = 100,
        n_features           = 2,
        n_informative        = 2,
        n_redundant          = 0,
        n_repeated           = 0,
        n_classes            = 2,
        n_clusters_per_class = 1,
        random_state         = 0,
    )
    X, y = make_classification(**data_cfg)
    data = pd.DataFrame(
        X,
        columns = [f'Feature {i}' for i in range(X.shape[1])],
    ).assign(target=y)
    cfg = get_default_cfg_supervised()
    cfg['inputs'].update({
        'features' : data.columns.drop('target'),
        'target'   : 'target',
    })
    # Test 1a: Binary classifier with categorical target
    test_cfg = copy.deepcopy(cfg)
    test_cfg['outputs'].update({
        # 'save_test_scores'         : True,
        'save_train_scores'        : True,
        'save_estimator'           : True,
        'save_test_predictions'    : True,
        'save_train_predictions'   : True,
        # 'save_cv_test_scores'      : True,
        'save_cv_train_scores'     : True,
        'save_cv_estimators'       : True,
        'save_cv_test_predictions' : True,
        'save_cv_train_predictions': True,
        'save_cv_indices'          : True,
        'save_feature_importances' : True,
    })
    # Test 1a: Binary classifier with categorical target
    results = fit_supervised_model(
        (data
            .replace({'target' : {0 : 'is_negative', 1 : 'is_positive'}})
            .astype({'target' : 'category'}) 
        ),
        test_cfg
    )
    fits                = results['fits']
    is_test_data        = results['is_test_data']
    y_train_pred        = results['y_train_pred']
    y_cv_test_pred      = results['y_cv_test_pred']
    y_test_pred         = results['y_test_pred']
    feature_importances = results['feature_importances']

    # Check dtype is preserved
    assert is_categorical_dtype(y_train_pred[('all','predict')].dtype)
    assert is_categorical_dtype(y_cv_test_pred['predict'].dtype)
    assert is_categorical_dtype(y_test_pred['predict'].dtype)

    # Check splits are concatenated correctly
    split_names = set(fits.index)
    assert 'all' in split_names
    assert set(is_test_data.columns) <= split_names
    assert set(y_train_pred.columns.levels[0]) == split_names
    assert set(y_cv_test_pred.index.levels[1]) <= split_names
    assert set(feature_importances.columns) == split_names

    # Check indices are concatenated correctly
    assert y_train_pred.index.equals(is_test_data.index)
    assert y_train_pred.index.equals(y_cv_test_pred.index.levels[0])
    assert y_train_pred.index.intersection(y_test_pred.index).empty
    assert y_train_pred.index.union(y_test_pred.index).equals(data.index)

    # Test 1b: Binary classifier with boolean target
    results = fit_supervised_model(data.astype({'target' : bool}), cfg)
    assert pd.api.types.is_bool_dtype(results['y_test_pred']['predict'])

    # Test 1c: Binary classifier with integer target
    results = fit_supervised_model(data.astype({'target' : np.uint8}), cfg)
    assert pd.api.types.is_integer_dtype(results['y_test_pred']['predict'])

    # Test X: Multiclass classifier

    # Test X: Multiclass & multilabel classifier

    # pytest.set_trace()

def is_categorical_dtype(x):
    return isinstance(x, pd.CategoricalDtype)