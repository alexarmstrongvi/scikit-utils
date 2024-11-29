
# Standard library
from collections import Counter
import copy
import logging

# 3rd party
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

# 1st party
from skutils.bin.fit_supervised_model import (
    _set_unset_returns,
    fit_supervised_model,
    predict
)
from skutils.data import get_default_cfg_supervised

logging.getLogger('asyncio').setLevel('WARNING')


################################################################################
def test_fit_supervised_model():
    X, y = make_classification(
        n_samples            = 100,
        n_features           = 2,
        n_informative        = 2,
        n_redundant          = 0,
        n_repeated           = 0,
        n_classes            = 2,
        n_clusters_per_class = 1,
        random_state         = 0,
    )
    data = pd.DataFrame(
        X,
        columns = [f'Feature {i}' for i in range(X.shape[1])],
    ).assign(target=y)
    cfg = get_default_cfg_supervised()
    del cfg['input']
    ocfg = cfg.pop('outputs')
    _set_unset_returns(cfg['returns'], ocfg['toggles'])

    cfg['estimator'] = 'ExtraTreesClassifier'
    cfg['fit']['scoring'] = ['accuracy']

    ########################################
    # Test default train-test evaluation
    results = fit_supervised_model(data, cfg)
    assert list(results.keys()) == ['fits']
    fits = results['fits'] # type: ignore[reportTypedDictNotRequiredAccess]
    pd.testing.assert_index_equal(fits.index, pd.Index(['split0'], name='split'))
    assert fits.columns.to_list() == ['fit_time', 'score_time', 'test_accuracy']

    # Test simple cross validation
    cfg_mod = copy.deepcopy(cfg)
    #TODO: cfg = get_example_cfg('KFold')
    cfg_mod['train_test_iterator'] = 'KFold'
    results = fit_supervised_model(data, cfg_mod)
    fits = results['fits'] # type: ignore[reportTypedDictNotRequiredAccess]
    pd.testing.assert_index_equal(
        fits.index,
        pd.Index(['split0','split1','split2','split3','split4'], name='split')
    )
    assert fits.columns.to_list() == ['fit_time', 'score_time', 'test_accuracy']

    # Test grouped cross validation
    cfg_mod = copy.deepcopy(cfg)
    cfg_mod['train_test_iterator']       = 'GroupKFold'
    cfg_mod['preprocess']['groups']      = 'groups'
    cfg_mod['returns']['return_indices'] = True
    data_grp = data.copy()
    data_grp['groups'] = np.arange(len(data_grp)) % (len(data)//4)
    results = fit_supervised_model(data_grp, cfg_mod)
    # No group value are in multiple splits
    cnt = Counter()
    for _, filt in results['is_test_data'].items():
        cnt.update(data_grp.loc[filt, 'groups'].unique())
    assert max(cnt.values()) == 1

    # Test stratified cross validation

    ########################################
    # Test no output enabled
    cfg_mod_no_out = copy.deepcopy(cfg)
    cfg_mod_no_out['train_test_iterator'] = 'KFold'
    cfg_mod_no_out['returns'] = {k : False for k in cfg_mod['returns']}
    results = fit_supervised_model(data, cfg_mod_no_out)
    fits = results['fits'] # type: ignore[reportTypedDictNotRequiredAccess]
    pd.testing.assert_index_equal(
        fits.index,
        pd.Index(['split0','split1','split2','split3','split4'], name='split')
    )
    assert fits.columns.to_list() == ['fit_time']
    assert fits.notna().all().all() # type: ignore[reportAttributeAccessIssue]

    # Test single output enabled
    for return_key in cfg['returns']:
        cfg_mod = copy.deepcopy(cfg_mod_no_out)
        cfg_mod['returns'][return_key] = True
        results = fit_supervised_model(data, cfg_mod)

        results_keys = list(results.keys())
        fits_columns = []
        split_names = pd.Index([])
        if 'fits' in results:
            fits_columns = results['fits'].columns.to_list()
            split_names = results['fits'].index
            pd.testing.assert_index_equal(
                split_names,
                pd.Index(['split0','split1','split2','split3','split4'], name='split')
            )
        elif 'is_test_data' in results:
            split_names = results['is_test_data'].columns
            pd.testing.assert_index_equal(
                split_names,
                pd.Index(['split0','split1','split2','split3','split4'], name='split')
            )

        if return_key == 'return_y_true':
            assert results_keys == ['y_true']
            pd.testing.assert_series_equal(results['y_true'], data['target'], check_names=False)
        elif return_key == 'return_test_scores':
            assert results_keys == ['fits']
            assert fits_columns == ['fit_time', 'score_time', 'test_accuracy']
            _test_fits(results['fits'])
        elif return_key == 'return_train_scores':
            assert results_keys == ['fits']
            assert fits_columns == ['fit_time', 'score_time', 'train_accuracy']
            _test_fits(results['fits'])
        elif return_key == 'return_estimators':
            assert results_keys == ['fits']
            assert fits_columns == ['fit_time', 'estimator']
            _test_fits(results['fits'])
        elif return_key == 'return_test_predictions':
            assert results_keys == ['fits', 'y_pred']
            assert fits_columns == ['fit_time']
            _test_fits(results['fits'])
            _test_y_pred(results['y_pred'], split_names)
        elif return_key == 'return_train_predictions':
            assert results_keys == ['fits', 'y_pred']
            assert fits_columns == ['fit_time']
            _test_fits(results['fits'])
            _test_y_pred(results['y_pred'], split_names)
        elif return_key == 'return_indices':
            assert results_keys == ['is_test_data']
            _test_is_test_data(results['is_test_data'], split_names)
        elif return_key == 'return_feature_importances':
            assert results_keys == ['fits', 'feature_importances']
            assert fits_columns == ['fit_time']
            _test_fits(results['fits'])
            _test_feature_importances(results['feature_importances'], split_names)
        else:
            assert False, f'No test for return cfg: {return_key}'

    # Test all outputs and options enabled
    cfg_mod = copy.deepcopy(cfg)
    cfg_mod['train_test_iterator'] = 'KFold'
    cfg_mod['returns'] = {k : True for k in cfg_mod['returns']}
    cfg_mod['train_on_all'] = True
    results = fit_supervised_model(data, cfg_mod)

    fits                = results['fits']
    is_test_data        = results['is_test_data']
    y_pred              = results['y_pred']
    feature_importances = results['feature_importances']

    split_names = fits.index

    assert 'all' in split_names

    _test_fits(fits)
    _test_is_test_data(is_test_data, split_names)
    _test_y_pred(y_pred, split_names)
    _test_feature_importances(feature_importances, split_names)

    # Check indices are the same across result outputs
    pd.testing.assert_index_equal(y_pred.index, is_test_data.index)
    pd.testing.assert_index_equal(y_pred.index, data.index, check_names=False)

    ########################################
    cfg_mod = copy.deepcopy(cfg)
    cfg_mod['returns']['return_test_predictions'] = True

    # Test 1a: Binary classifier with boolean target
    results = fit_supervised_model(data.astype({'target' : bool}), cfg_mod)
    assert pd.api.types.is_bool_dtype(results['y_pred']['predict'])

    # Test 1b: Binary classifier with integer target
    results = fit_supervised_model(data.astype({'target' : np.uint8}), cfg_mod)
    assert pd.api.types.is_unsigned_integer_dtype(results['y_pred']['predict'])

    # Test 1c: Binary classifier with categorical target
    cfg_mod_cat = copy.deepcopy(cfg_mod)
    cfg_mod_cat['pos_label'] = 'is_positive'
    results = fit_supervised_model(
        (data
            .astype({'target' : object})
            .replace({'target' : {0 : 'is_negative', 1 : 'is_positive'}})
            .astype({'target' : 'category'})
        ),
        cfg_mod_cat,
    )
    assert is_categorical_dtype(results['y_pred']['predict'].dtype)

    # Test X: Multiclass classifier

    # Test X: Multiclass & multilabel classifier

def test_fit_supervised_model_regression():
    cfg = get_default_cfg_supervised()
    del cfg['input']
    ocfg = cfg.pop('outputs')
    _set_unset_returns(cfg['returns'], ocfg['toggles'])
    cfg['estimator'] = 'ExtraTreesRegressor'

    # Test regression
    X, y = make_regression(
        n_samples            = 100,
        n_features           = 2,
        n_informative        = 2,
        random_state         = 0,
    )
    data = pd.DataFrame(
        X,
        columns = [f'Feature {i}' for i in range(X.shape[1])],
    ).assign(target=y)

    cfg_mod = copy.deepcopy(cfg)
    cfg_mod['train_test_iterator'] = 'KFold'
    cfg_mod['returns'] = {k : True for k in cfg_mod['returns']}
    cfg_mod['train_on_all'] = True
    results = fit_supervised_model(data, cfg_mod)

    fits                = results['fits']
    is_test_data        = results['is_test_data']
    y_pred              = results['y_pred']
    feature_importances = results['feature_importances']

    split_names = fits.index

    assert 'all' in split_names

    _test_fits(fits)
    _test_is_test_data(is_test_data, split_names)
    _test_y_pred(y_pred, split_names)
    _test_feature_importances(feature_importances, split_names)

    # Check indices are the same across result outputs
    pd.testing.assert_index_equal(y_pred.index, is_test_data.index)
    pd.testing.assert_index_equal(y_pred.index, data.index, check_names=False)

def test_predict():
    # Test with classifier
    X, y = make_classification(
        n_samples            = 100,
        n_features           = 2,
        n_informative        = 2,
        n_redundant          = 0,
        n_repeated           = 0,
        n_classes            = 2,
        n_clusters_per_class = 1,
        random_state         = 0,
    )
    X_clf = pd.DataFrame(
        X,
        columns = [f'Feature {i}' for i in range(X.shape[1])]
    )
    y_clf = pd.Series(y, name='target')

    clf = ExtraTreesClassifier().fit(X_clf, y_clf)
    pred_clf = predict(clf, X_clf)
    np.testing.assert_array_equal(pred_clf.columns, ['prob(0)', 'prob(1)', 'predict'])
    pd.testing.assert_index_equal(pred_clf.index, X_clf.index)
    assert pred_clf['predict'].dtype is y_clf.dtype

    pred_clf = predict(clf, X_clf, proba=False)
    np.testing.assert_array_equal(pred_clf.columns, ['predict'])
    pd.testing.assert_index_equal(pred_clf.index, X_clf.index)
    assert pred_clf['predict'].dtype is y_clf.dtype

    pd.testing.assert_frame_equal(pred_clf, predict(clf, X_clf, proba=False, simplify=True))

    pred_clf = predict(clf, X_clf, simplify=True)
    np.testing.assert_array_equal(pred_clf.columns, ['prob(0)'])
    pd.testing.assert_index_equal(pred_clf.index, X_clf.index)

    # TODO: Add multi-class classifier outputs

    # TODO: Add multi-label classifier outputs

    # Test with regressor
    X, y = make_regression(
        n_samples            = 100,
        n_features           = 2,
        n_informative        = 2,
        random_state         = 0,
    )
    X_reg = pd.DataFrame(
        X,
        columns = [f'Feature {i}' for i in range(X.shape[1])]
    )
    y_reg = pd.Series(y, name='target')

    reg = ExtraTreesRegressor().fit(X_reg, y_reg)
    pred_reg = predict(reg, X_reg)
    np.testing.assert_array_equal(pred_reg.columns, ['predict'])
    pd.testing.assert_index_equal(pred_reg.index, X_reg.index)
    assert pred_reg['predict'].dtype is y_reg.dtype

    # TODO: Add multi-label regressor outputs

def _test_fits(fits: pd.DataFrame) -> None:
    assert fits.notna().all().all()
    assert pd.api.types.is_float_dtype(fits['fit_time'])
    if 'test_accuracy' in fits:
        assert pd.api.types.is_float_dtype(fits['test_accuracy'])
    if 'train_accuracy' in fits:
        assert pd.api.types.is_float_dtype(fits['train_accuracy'])
    if 'estimators' in fits:
        assert pd.api.types.is_object_dtype(fits['estimators'])
        assert fits['estimators'].apply(id).nunique() == len(fits)

def _test_is_test_data(is_test_data: pd.DataFrame, split_names: pd.Index) -> None:
    assert all(pd.api.types.is_bool_dtype(is_test_data[c]) for c in is_test_data.columns)
    pd.testing.assert_index_equal(is_test_data.columns, split_names)

def _test_y_pred(y_pred: pd.DataFrame, split_names: pd.Index) -> None:
    index_names = list(y_pred.index.names)
    if index_names == ['index']:
        n_classes = len(y_pred[split_names[0]].columns) - 1
        pred_cols = pd.Index([f'prob({i})' for i in range(n_classes)] + ['predict'], name='pred')
        pd.testing.assert_index_equal(y_pred.columns.levels[0], split_names)
        pd.testing.assert_index_equal(y_pred.columns.levels[1], pred_cols)
        # There should be at least one result for each data row across all splits
        assert y_pred.stack('pred', future_stack=True).notna().any(axis=1).all()
    elif index_names == ['index', 'split']:
        n_classes = len(y_pred.columns) - 1
        pred_cols = pd.Index([f'prob({i})' for i in range(n_classes)] + ['predict'], name='pred')
        pd.testing.assert_index_equal(y_pred.columns, pred_cols)
        assert y_pred.notna().all().all()
    else:
        assert False, f'Unexpected index names: {y_pred.index.names}'

def _test_feature_importances(feature_importances: pd.Series, split_names: pd.Index) -> None:
    assert all(pd.api.types.is_float_dtype(feature_importances[c]) for c in feature_importances.columns)
    pd.testing.assert_index_equal(feature_importances.columns, split_names)


################################################################################
# Utilities
def is_categorical_dtype(x):
    return isinstance(x, pd.CategoricalDtype)
