#!/usr/bin/env python
"""
Configurable script for fitting supervised models
"""
# Standard library
import argparse
from copy import deepcopy
import logging
from pathlib import Path
from typing import Any, Literal

# 3rd party
import numpy as np
from tensorboardX import SummaryWriter
import yaml

# 1st party
from skutils import logging_utils, plot, scripting
from skutils.data import get_default_cfg_supervised
from skutils.fit_supervised_model import (
    FitSupervisedModelResults,
    fit_supervised_model
)
from skutils.persistance import dump_model, dump_pandas, read_pandas

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
    set_unset_returns(cfg['returns'], ocfg['toggles'])

    data = read_pandas(ipath, **icfg)
    results = fit_supervised_model(data, cfg)

    if (fits := results.get('fits')) is not None:
        log.info('Fit results:\n%s', fits.drop(columns='estimator', errors='ignore'))

    save_results(results, ocfg)

    if any_plots_enabled(ocfg):
        random_state = np.random.RandomState(cfg['random_seed'])
        save_visualizations(
            results,
            ocfg         = ocfg,
            cfg          = cfg['visualization'],
            pos_label    = cfg['pos_label'],
            random_state = random_state,
        )

################################################################################
def any_plots_enabled(ocfg: dict) -> bool:
    return any(v for k,v in ocfg['toggles'].items() if k.endswith('plot'))

def save_results(
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
        opath = odir / f'fits.{ext_pandas}'
        dump_pandas(results['fits'].drop(columns='estimator', errors='ignore'), opath)
        log.info('Saved fit results: %s', opath)

    if cfg_save['save_estimators']:
        odir_models = odir / 'models'
        odir_models.mkdir()
        for split, est in results['fits']['estimator'].items():
            opath = odir_models / f'{est.__class__.__name__}_{split}.{ext_model}'
            dump_model(est, opath)
            log.info('Estimator saved: %s', opath)

    if cfg_save['save_test_predictions'] or cfg_save['save_train_predictions']:
        opath = odir / f'predictions.{ext_pandas}'
        dump_pandas(results['y_pred'], opath)
        log.info('Saved predictions: %s', opath)

    if cfg_save['save_indices']:
        opath = odir / f'is_test_data.{ext_pandas}'
        dump_pandas(results['is_test_data'], opath)
        log.info('Saved test data per split flags: %s', opath)

    if cfg_save['save_feature_importances']:
        opath = odir / f'feature_importances.{ext_pandas}'
        dump_pandas(results['feature_importances'], opath)
        log.info('Saved feature importances results: %s', opath)

    if tb_writer is not None:
        tb_writer.close()

def save_visualizations(
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

def set_unset_returns(cfg_return: dict[str, bool | None], cfg_save : dict[str, bool]) -> None:
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
