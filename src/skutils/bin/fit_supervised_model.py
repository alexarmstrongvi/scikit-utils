#!/usr/bin/env python
"""
Configurable script for fitting supervised models
"""
# Standard library
import argparse
from collections.abc import Collection
from copy import deepcopy
import logging
from pathlib import Path
from typing import Any, Literal, TypedDict

# 3rd party
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator, is_classifier
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.utils.validation import _check_pos_label_consistency
from tensorboardX import SummaryWriter
import yaml

# 1st party
from skutils import logging_utils, plot, scripting
from skutils.data import get_default_cfg_supervised
from skutils.import_utils import import_object
from skutils.persistance import dump_model, dump_pandas, read_pandas
from skutils.pipeline import make_pipeline
from skutils.prediction import combine_predictions, predict
from skutils.train_test_iterators import TrainOnAllWrapper, TrainTestIterable

# Globals
log = logging.getLogger(Path(__file__).stem)

# TODO: Update to work on multi-class problems
# TODO: Update to work on multi-label problems
# TODO: Update to work on multi-class & multi-label problems
# TODO: Update to enable hyperparameter searching
# TODO: Update to enable feature selection
################################################################################
def main(cfg):
    # Remove IO configs as they should not be used in fit_supervised_model()
    icfg = cfg.pop('input')
    ipath = Path(icfg.pop('path'))
    ocfg = cfg.pop('outputs')
    _set_unset_returns(cfg['returns'], ocfg['toggles'])

    data = read_pandas(ipath, **icfg)
    results = fit_supervised_model(data, cfg)

    if (fits := results.get('fits')) is not None:
        log.info('Fit results:\n%s', fits.drop(columns='estimator', errors='ignore'))

    _save_results(results, ocfg)

    if _any_plots_enabled(ocfg):
        random_state = np.random.RandomState(cfg['random_seed'])
        _save_visualizations(
            results,
            ocfg         = ocfg,
            cfg          = cfg['visualization'],
            pos_label    = cfg['pos_label'],
            random_state = random_state,
        )

class FitSupervisedModelResults(TypedDict, total=False):
    X                   : DataFrame
    y_true              : Series | DataFrame
    fits                : DataFrame
    is_test_data        : DataFrame
    y_pred              : DataFrame
    feature_importances : DataFrame

def fit_supervised_model(data: DataFrame, cfg: dict) -> FitSupervisedModelResults:
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
    is_classification = is_classifier(estimator)

    ########################################
    log.info('Preprocessing data')
    X, y, groups = preprocess_data(data, **cfg['preprocess'])
    feature_names = X.columns.tolist()

    if cfg_return['return_X'] and n_outputs == 1:
        return {'X' : X}
    if cfg_return['return_y_true'] and n_outputs == 1:
        return {'y_true' : y}

    ########################################
    y_pred              = None
    is_test_data        = None
    feature_importances = None
    fit_results         = None
    if cfg['model_selector'] is not None:
        # TODO: Implement evaluation of multiple models with different
        # parameters. Not possible to save out as much information so the
        # outputs will be quite different for this use case
        model_selector = build_model_selector(estimator, **cfg['model_selector'])
        model_selector.fit(X, y, cv=train_test_iterator, **cfg['fit'])
        # fit_results.update(...)
    else:
        log.info('Running training and testing')
        return_predictions = cfg_return['return_train_predictions'] or cfg_return['return_test_predictions']
        return_estimators  = cfg_return['return_estimators'] or cfg_return['return_feature_importances'] or return_predictions
        return_indices     = cfg_return['return_indices'] or return_predictions
        # TODO: Implement general train_test function that has the
        # parallelization of cross_validate but allows the user full
        # configuration to, say, save out train predictions and scores
        # without evaluating on test set.
        fit_results = cross_validate(
            estimator,
            X,
            y,
            groups             = groups,
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
        split_names = _get_split_names(cfg['split_names'], n_splits, cfg['train_on_all'])

        fit_results, is_test_data = to_cross_validate_dataframes(
            fit_results,
            data_index = X.index,
            split_names = split_names,
        )

        if return_predictions:
            # Checks and Setup
            assert is_test_data is not None
            return_only_train_pred = not cfg_return['return_test_predictions']
            return_only_test_pred  = not cfg_return['return_train_predictions']

            if return_only_test_pred:
                masks = (mask for _, mask in is_test_data.items())
            elif return_only_train_pred:
                masks = (~mask for _, mask in is_test_data.items())
            else:
                masks = (mask.notna() for _, mask in is_test_data.items())

            pos_label = cfg['pos_label']
            if is_classification:
                pos_label = _check_pos_label_consistency(pos_label, y)
            simplify = cfg['simplify_prediction_cols']

            # Get predictions from trained estimator on all requested splits of
            # the data. Predictions can involve multiple columns so the combined
            # dataframe will use a multi-index column
            y_pred : pd.DataFrame = pd.concat(
                [
                    predict(est, X[mask], simplify=simplify, pos_label=pos_label)
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
                    # Don't both keeping split index if there is only one split
                    if y_pred.index.get_level_values('split').nunique() == 1:
                        y_pred = y_pred.droplevel('split')

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
        if 'all' in feature_importances:
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
    astype     : dict | None = None,
    index      : Any = None,
    features   : Collection[str] | None = None,
    target     : str,
    target_map : dict | None = None,
    groups     : str | None = None,
) -> tuple[DataFrame, Series, Series | None]:
    '''Preprocess the raw data for model fitting

    This SHOULD NOT INVOLVE data-dependent transformations (e.g. interpolate
    missing data) in order to avoid data leakage. Such transformations should be
    part of the estimator pipeline created in build_estimator
    '''

    if astype is not None:
        data = data.astype(astype)

    if index is not None:
        data = data.set_index(index)
    else:
        data = data.rename_axis('index')

    if features is None:
        features = data.columns.drop(target)
        log.info('Features not specified. Using all but the target feature: %s', features.to_list())

    # NOTE: User can put any quickfixes here but in general scikit-learn expects
    # user to have done all general preprocessing prior to running pipeline. The
    # preprocessing that is part of the pipeline is reserved for steps that risk
    # data leakage and should be run separately in the case of cross validation.

    # TODO: Save any columns not in features or target to metadata and return?
    X   = data[features]
    y   = data[target]
    grp = data[groups] if groups is not None else None

    if target_map is not None:
        y = y.map(target_map)

    return X, y, grp

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
        case _:
            raise ValueError(f'Unexpected estimator config: {cfg}')
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

def _any_plots_enabled(ocfg: dict) -> bool:
    return any(v for k,v in ocfg['toggles'].items() if k.endswith('plot'))

def _save_results(
    results: FitSupervisedModelResults,
    ocfg   : dict,
) -> None:
    odir = Path(ocfg['path'])
    ext_pandas = ocfg['pandas_format']
    ext_model  = ocfg['model_format']
    cfg_save = ocfg['toggles']

    # TODO: Enable saving results with Tensorboard
    tb_writer = None
    if ocfg['tensorboard']:
        tb_writer = SummaryWriter(logdir=ocfg['path'])

    if cfg_save['save_test_scores'] or cfg_save['save_train_scores']:
        assert 'fits' in results
        opath = odir / f'fits.{ext_pandas}'
        dump_pandas(results['fits'].drop(columns='estimator', errors='ignore'), opath)
        log.info('Saved fit results: %s', opath)

    if cfg_save['save_estimators']:
        assert 'fits' in results
        assert 'estimator' in results['fits']
        odir_models = odir / 'models'
        odir_models.mkdir()
        for split, est in results['fits']['estimator'].items():
            assert isinstance(est, BaseEstimator)
            opath = odir_models / f'{est.__class__.__name__}_{split}.{ext_model}'
            dump_model(est, opath)
            log.info('Estimator saved: %s', opath)

    if cfg_save['save_test_predictions'] or cfg_save['save_train_predictions']:
        assert 'y_pred' in results
        opath = odir / f'predictions.{ext_pandas}'
        dump_pandas(results['y_pred'], opath)
        log.info('Saved predictions: %s', opath)

    if cfg_save['save_indices']:
        assert 'is_test_data' in results
        opath = odir / f'is_test_data.{ext_pandas}'
        dump_pandas(results['is_test_data'], opath)
        log.info('Saved test data per split flags: %s', opath)

    if cfg_save['save_feature_importances']:
        assert 'feature_importances' in results
        opath = odir / f'feature_importances.{ext_pandas}'
        dump_pandas(results['feature_importances'], opath)
        log.info('Saved feature importances results: %s', opath)

    if tb_writer is not None:
        tb_writer.close()

def _save_visualizations(
    results: FitSupervisedModelResults,
    ocfg: dict,
    cfg: dict,
    pos_label: Any | None,
    random_state: np.random.RandomState,
) -> None:
    ext          = ocfg['image_format']
    cfg_save     = ocfg['toggles']
    y_true       = results.get('y_true')
    y_pred       = results.get('y_pred')
    is_test_data = results.get('is_test_data')

    if ocfg['tensorboard']:
        tb_writer = SummaryWriter(logdir=ocfg['path'])
    else:
        tb_writer = None
        odir = Path(ocfg['path'])/'visualization'
        odir.mkdir()

    # No guarantee y_true and is_test_data have the same entries as y_pred so
    # get an index that can be used to select the corresponding entries when
    # comparing truth to prediction. For example, if only predictions on a
    # holdout set were done, y_pred would only exist for those values.
    pred_index = None
    if y_pred is not None:
        pred_index = y_pred.index
        if isinstance(pred_index, pd.MultiIndex):
            # TODO: Get correct index without assuming it is the first level
            pred_index = pred_index.levels[0]

    # Classifier predict
    if cfg_save['save_confusion_matrix_plot']:
        figures = plot.plot_confusion_matrix(
            y_true.loc[pred_index],
            y_pred,
            is_test_data.loc[pred_index],
            cfg['ConfusionMatrixDisplay']
        )
        for split, fig in figures:
            if ocfg['tensorboard']:
                tag = f"confusion_matrix/{split}"
                tb_writer.add_figure(tag, fig)
                log.info('Confusion matrix logged to tensorboard: %s', tag)
            else:
                opath = odir / f'confusion_matrix_{split}.{ext}'
                fig.savefig(opath, format=ext)
                log.info('Confusion matrix plot saved: %s', opath)

    # Classifier prob
    if cfg_save['save_roc_curve_plot']:
        figures = plot.plot_roc_curve(
            y_true.loc[pred_index],
            y_pred,
            is_test_data.loc[pred_index],
            pos_label,
            cfg['RocCurveDisplay']
        )
        for split, fig in figures.items():
            if ocfg['tensorboard']:
                tag = f"roc_curve/{split}"
                tb_writer.add_figure(tag, fig)
                log.info('ROC curve logged to tensorboard: %s', tag)
            else:
                opath = odir / f'roc_curve_{split}.{ext}'
                fig.savefig(opath, format=ext)
                log.info('ROC curve plot saved: %s', opath)
    # if cfg_save['save_det_curve']:
    #     fig, ax = plot_det_curve()
    #     metrics.DetCurveDisplay.from_predictions(y_true, y_pred)
    # if cfg_save['save_precision_recall_curve']:
    #     fig, ax = plot_precision_recall_curve()
    #     metrics.PrecisionRecallDisplay.from_predictions(y_true, y_pred)
    # if cfg_save['save_calibration_plot']:
    #     calibration.CalibrationDisplay(prob_true, prob_pred, y_prob, estimator_name=estimator_name, pos_label=pos_label)

    # # Regression predict
    # if cfg_save['save_prediction_error']:
    #     metrics.PredictionErrorDisplay.from_predictions(y_true, y_pred)
    if cfg_save['save_partial_dependence_plot']:
        figures = plot.plot_partial_dependence(
            estimators = results['fits']['estimator'],
            X = results['X'],
            cfg = cfg['PartialDependenceDisplay'] or {},
            random_state=random_state,
        )
        for split, fig in figures:
            if ocfg['tensorboard']:
                tag = f"partial_dependence/{split}"
                tb_writer.add_figure(tag, fig)
                log.info('Partial dependence plot logged to tensorboard: %s', tag)
            else:
                opath = odir / f'partial_dependence_{split}.{ext}'
                fig.savefig(opath, format=ext)
                log.info('Partial dependence plot saved: %s', opath)

    # inspection.DecisionBoundaryDisplay.from_estimator()
    # model_selection.LearningCurveDisplay.from_estimator()
    # model_selection.ValidationCurveDisplay.from_estimator()

    if ocfg['tensorboard']:
        tb_writer.close()

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

################################################################################
def parse_argv() -> argparse.Namespace:
    """Parse command line arguments (i.e. sys.argv)"""
    parser = argparse.ArgumentParser()
    add_arguments(parser, start_stage='run')
    return parser.parse_args()

def add_arguments(
    parser: argparse.ArgumentParser,
    start_stage : Literal["run", "score", "visualize"],
) -> argparse.ArgumentParser:
    parser.add_argument(
        "-c",
        "--configs",
        nargs="+",
        metavar='PATH',
        default=[],
        help="Configuration files",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        metavar='PATH',
        help="Path to input training data",
    )
    parser.add_argument(
        "-o",
        "--odir",
        type=Path,
        metavar='PATH',
        help="Directory for saving all outputs",
    )
    parser.add_argument(
        "--overwrite",
        action='store_true',
        help="Overwrite output directory",
    )

    # TODO: if start_stage <= MLStages.FIT:
    if start_stage == 'run':
        parser.add_argument(
            "--estimator",
            help="Estimator to fit",
        )
        parser.add_argument(
            "--cv",
            help="Train-test iterator to pass to cross_validate",
        )

    parser.add_argument(
        "-l",
        "--log-level",
        choices=logging_utils.LOG_LEVEL_CHOICES,
        help="Root logging level",
    )
    parser.add_argument(
        "--log-file",
        action = 'store_true',
        help="Save logs to file in output directory",
    )
    parser.add_argument(
        "--log-cli-level",  # log_cli_level
        choices=logging_utils.LOG_LEVEL_CHOICES,
        help="CLI logging level (if different from log-level)",
    )
    return parser

def get_config(args: argparse.Namespace):
    default_cfg = get_default_cfg_supervised()
    if args:
        override_cfgs = (yaml.safe_load(Path(p).open()) for p in args.configs)
        cfg = scripting.merge_config_files(override_cfgs, default_cfg)
    else:
        cfg = default_cfg
    cfg = override_config(cfg, args)
    validate_config(cfg)
    return cfg

def override_config(cfg: dict, args: argparse.Namespace, copy: bool = True) -> dict:
    """Override configuration settings with command line arguments."""
    if copy:
        cfg = deepcopy(cfg)
    if args.input:
        cfg["input"]["path"] = str(args.input)
    if args.odir:
        cfg["outputs"]["path"] = str(args.odir)
    if args.overwrite:
        cfg["outputs"]["overwrite"] = args.overwrite
    if args.estimator:
        cfg['estimator'] = args.estimator
    if args.cv:
        cfg['train_test_iterator']['name'] = args.cv
    # Logging
    log_cfg = cfg["logging"]
    if args.log_level and "level" in log_cfg:
            log_cfg["level"] = args.log_level
    if args.log_file and "filename" in log_cfg:
            log_cfg["filename"] = 'run.log'
    # args.log_cli_level handled in main()
    return cfg

def validate_config(cfg: dict) -> None:
    messages = []
    if cfg['input']['path'] is None:
        messages.append(
            'Input data not specified. Use --input CLI arg or input.path in config'
        )
    if cfg['outputs']['path'] is None:
        messages.append(
            'Output directory not specified. Use --odir CLI arg or outputs.path in config'
        )
    elif not (parent := Path(cfg['outputs']['path']).parent).is_dir():
        messages.append(
            f'Output directory parent path does not exist: {parent}'
        )
    if cfg['estimator'] is None:
        messages.append(
            'Estimator is not specified. Use --estimator CLI arg or estimator in config'
        )
    if len(messages) > 0:
        raise RuntimeError('Invalid configuration:\n\t- ' + '\n\t- '.join(messages))

def _set_unset_returns(cfg_return: dict[str, bool | None], cfg_save : dict[str, bool]) -> None:
    '''Automatically determine toggles for fit_supervised_model() return
    parameters based on the save toggles for fit_supervized_model.py if the
    return parameters are not set'''
    any_display_from_predictions = (
        # All Display visuals that use from_predictions()
        cfg_save['save_confusion_matrix_plot']
        or cfg_save['save_roc_curve_plot']
        or cfg_save['save_det_curve_plot']
        or cfg_save['save_predicion_recall_curve_plot']
        or cfg_save['save_calibration_plot']
        or cfg_save['save_prediction_error_plot']
    )
    any_display_from_estimators = (
        # All Display visuals that use from_estimator()
        cfg_save['save_partial_dependence_plot']
        or cfg_save['save_decision_bounary_plot']
        or cfg_save['save_learning_curve_plot']
        or cfg_save['save_validation_curve_plot']
)

    if cfg_return['return_X'] is None:
        cfg_return['return_X'] = any_display_from_estimators

    if cfg_return['return_y_true'] is None:
        cfg_return['return_y_true'] = any_display_from_predictions

    if cfg_return['return_test_scores'] is None:
        cfg_return['return_test_scores'] = cfg_save['save_test_scores']

    if cfg_return['return_train_scores'] is None:
        cfg_return['return_train_scores'] = cfg_save['save_train_scores']

    if cfg_return['return_test_predictions'] is None:
        cfg_return['return_test_predictions'] = (
            cfg_save['save_test_predictions'] or any_display_from_predictions
        )

    if cfg_return['return_estimators'] is None:
        cfg_return['return_estimators'] = (
            cfg_save['save_estimators']
            or any_display_from_estimators
        )

    if cfg_return['return_train_predictions'] is None:
        cfg_return['return_train_predictions'] = cfg_save['save_train_predictions']

    if cfg_return['return_indices'] is None:
        cfg_return['return_indices'] = (
            cfg_save['save_indices'] or any_display_from_predictions
        )

    if cfg_return['return_feature_importances'] is None:
        cfg_return['return_feature_importances'] = cfg_save['save_feature_importances']

# Accessed through run_skutils:main
# if __name__ == '__main__':
#     main()
