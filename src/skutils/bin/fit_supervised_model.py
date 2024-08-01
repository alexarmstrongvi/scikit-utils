#!/usr/bin/env python
"""
Configurable script for fitting supervised models
"""
# Standard library
import argparse
from collections.abc import Collection
from copy import deepcopy
from datetime import datetime
import logging
from pathlib import Path
import pickle
import pprint
import shutil
from typing import Any, TypedDict

# 3rd party
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import sklearn
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, cross_validate
import yaml

# 1st party
from skutils import git, logging_utils, scripting
from skutils.data import get_default_cfg_supervised
from skutils.import_utils import import_object
from skutils.pipeline import make_pipeline
from skutils.prediction import combine_predictions, predict
from skutils.train_test_iterators import TrainOnAllWrapper, TrainTestIterable

# Globals
log = logging.getLogger(Path(__file__).stem)

# Configuration
logging.getLogger('asyncio').setLevel('WARNING')
sklearn.set_config(transform_output='pandas')

# TODO: Update to work on multi-class problems
# TODO: Update to work on multi-label problems
# TODO: Update to work on multi-class & multi-label problems
# TODO: Update to enable hyperparameter searching
# TODO: Update to enable feature selection
################################################################################
def main():
    args = parse_argv()
    cfg  = get_config(args)

    ocfg = cfg['outputs']
    odir = Path(ocfg['path'])

    # Setup
    if ocfg["timestamp_subdir"]:
        odir.mkdir(exist_ok=True)
        odir = odir / datetime.now().strftime("%Y%m%d_%H%M%S")
    scripting.require_empty_dir(odir, overwrite=ocfg['overwrite'])
    logging_utils.update_log_filenames(odir, cfg['logging'])
    logging_utils.configure_logging(**cfg['logging'])
    logging_utils.require_root_console_handler(args.log_cli_level)
    log.debug("Logging Summary:\n%s", logging_utils.summarize_logging())

    # Reproducibility
    log.debug("Final configuration:\n%s", pprint.pformat(cfg, indent=4))
    if ocfg['save_input_configs']:
        buf = len(args.configs)
        for i, path in enumerate(args.configs):
            opath = odir / f'config_input{i:0{buf}d}_{Path(path).stem}.yml'
            shutil.copyfile(path, opath)
            log.info("Input configuration saved: %s", opath)
    if ocfg['save_final_config']:
        opath = odir / "config.yml"
        yaml.safe_dump(cfg, opath.open("w"))
        log.info("Final configuration saved: %s", opath)

    if (working_dir := git.find_working_dir(Path(__file__))) is not None:
        log.debug("Version Control Summary:\n%s", git.summarize_version_control(working_dir))
        if ocfg['save_git_diff']:
            opath = odir / "git_diff.patch"
            opath.write_text(git.get_diff(working_dir))
            log.info("Git diff patch saved: %s", opath)

    ########################################
    # Run
    ########################################
    icfg = cfg.pop('input')
    ipath = Path(icfg.pop('path'))
    data = read_data(ipath, **icfg)
    results = fit_supervised_model(data, cfg)

    if (fits := results.get('fits')) is not None:
        log.info('Fit results:\n%s', fits.drop(columns='estimators', errors='ignore'))

    # log.info('Saving results')
    save_results(odir, results)
    # save_visualizations(results, cfg['visualizations'])

class FitSupervisedModelResults(TypedDict, total=False):
    fits                : DataFrame
    is_test_data        : DataFrame
    y_pred              : DataFrame
    feature_importances : Series

def fit_supervised_model(data: DataFrame, cfg: dict) -> FitSupervisedModelResults:
    # TODO: Split up cfg into kwargs once there is a stable format
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
    estimator = build_estimator(cfg['estimator'])

    ########################################
    log.info('Preprocessing data')
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
                names = ['split', 'pred'],
                axis=1,
            )

            if cfg['stack_split_predictions']:
                # Ensure 1 column of results per stack
                is_notna = y_pred.stack('pred', future_stack=True).notna()
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

    if ocfg['save_indices'] and sum(v for k,v in ocfg.items() if k.startswith('save_')) == 1:
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

################################################################################
def read_data(path: Path, **kwargs) -> DataFrame:
    if path.suffix == '.csv':
        data = pd.read_csv(path, **kwargs)
    elif path.suffix in ('.hdf', '.hdf5'):
        data = pd.read_hdf(path, **kwargs)
    elif path.suffix in ('.parquet'):
        data = pd.read_parquet(path, **kwargs)
    else:
        raise NotImplementedError(
            f'No reader implemented for files of type {path.suffix}'
        )
    return data

def preprocess_data(
    data       : pd.DataFrame,
    *,
    target     : str,
    features   : Collection[str] | None = None,
    target_map : dict | None = None,
    astype     : dict | None = None,
    index      : Any = None,
) -> tuple[DataFrame, Series]:
    '''Preprocess the raw data for model fitting

    This SHOULD NOT INVOLVE data-dependent transformations (e.g. interpolate
    missing data) in order to avoid data leakage. Such transformations should be
    part of the estimator pipeline created in build_estimator
    '''


    if target_map is not None:
        data[target] = data[target].map(target_map)

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

    return X, y

def build_estimator(cfg: str | dict) -> BaseEstimator:
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

def build_model_selector(estimator: BaseEstimator, param_grid: dict = None, **kwargs) -> BaseEstimator:
    return GridSearchCV(estimator, **kwargs)

def save_results(odir: Path, results: FitSupervisedModelResults) -> None:
    # TODO: Enable saving to different formats (e.g. hdf, parquet, feather, etc.)
    if (fits := results.get('fits')) is not None:
        if 'estimator' in fits.columns:
            estimators = fits['estimator']
            fits = fits.drop(columns='estimator')
            (odir / 'models').mkdir()
            for split, est in estimators.items():
                opath = odir / f'models/estimator_{split}.pkl'
                # TODO: Enable saving in ONNX format
                with opath.open('wb') as ofile:
                    pickle.dump(est, ofile, protocol=pickle.HIGHEST_PROTOCOL)
                log.info('Estimator saved: %s', opath)

        opath = odir / 'fits.csv'
        fits.to_csv(opath)
        log.info('Saved fit results: %s', opath)

    if (is_test_data := results.get('is_test_data')) is not None:
        opath = odir / 'is_test_data.csv'
        is_test_data.to_csv(opath)
        log.info('Saved fit results: %s', opath)

    if (y_pred := results.get('y_pred')) is not None:
        opath = odir / 'y_pred.csv'
        y_pred.to_csv(opath)
        log.info('Saved fit results: %s', opath)

    if (feature_importances := results.get('feature_importances')) is not None:
        opath = odir / 'feature_importances.csv'
        feature_importances.to_csv(opath)
        log.info('Saved fit results: %s', opath)

def save_visualizations(results: FitSupervisedModelResults, cfg: dict) -> None:
    pass

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

################################################################################
def parse_argv() -> argparse.Namespace:
    """Parse command line arguments (i.e. sys.argv)"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--configs",
        nargs="+",
        default=[],
        help="Configuration files",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        help="Path to input training data",
    )
    parser.add_argument(
        "-o",
        "--odir",
        type=Path,
        help="Directory for saving all outputs",
    )
    parser.add_argument(
        "--overwrite",
        action='store_true',
        help="Overwrite output directory",
    )
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
    return parser.parse_args()

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

if __name__ == '__main__':
    main()
