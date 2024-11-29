
# Standard library
from typing import Iterable

# 3rd party
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from sklearn import calibration, inspection, metrics, model_selection
from sklearn.utils.validation import _check_pos_label_consistency


################################################################################
def plot_confusion_matrix(
    y_true       : pd.Series,
    y_pred       : pd.DataFrame,
    is_test_data : pd.DataFrame,
    display_kw   : dict | None = None,
) -> Iterable[tuple[str, Figure]]:
    assert len(y_true) == len(y_pred) == len(is_test_data)
    kwargs = display_kw or {}
    figures = {}
    iterable = []
    if 'split' in y_pred.index.names:
        iterable = [('combined_splits', (y_true, y_pred['predict']))]
    elif 'split' in y_pred.columns.names:
        splits = y_pred.columns.get_level_values('split').unique()
        iterable = []
        for split in splits:
            filt = is_test_data[split]
            y_true_split = y_true.loc[filt]
            y_pred_split = y_pred.loc[filt, (split, 'predict')]
            iterable.append((split, (y_true_split, y_pred_split)))

    for name, (y_true, y_pred) in iterable:
        disp = metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_pred, **kwargs)
        ax = disp.ax_
        ax.set_title(f'Confusion matrix [{name}]')
        yield (name, disp.figure_)

def plot_roc_curve(
    y_true       : pd.Series,
    y_pred       : pd.DataFrame,
    is_test_data : pd.DataFrame,
    pos_label    : int | str | None = None,
    display_kw   : dict | None = None,
    # TODO: Update to return generator like plot_confusion_matrix
) -> dict[str, Figure]:
    kwargs = display_kw or {}
    pos_label = _check_pos_label_consistency(pos_label, y_true)
    prob_col = f'prob({pos_label})'
    assert len(y_true) == len(y_pred) == len(is_test_data)

    figures = {}
    if 'split' in y_pred.columns.names:
        splits = y_pred.columns.get_level_values('split').unique()
        iterable = []
        for split in splits:
            filt = is_test_data[split]
            y_true_split = y_true.loc[filt]
            y_pred_split = y_pred.loc[filt, (split, prob_col)]
            iterable.append((split, (y_true_split, y_pred_split)))
    else:
        iterable = [('combined_splits', (y_true, y_pred[prob_col]))]

    # TODO: Decide how to plot multiple ROC curves for each split: Overlay in a
    # single plot or separate plots? How to set title and filename?
    for name, (y_true, y_pred) in iterable:
        disp = metrics.RocCurveDisplay.from_predictions(y_true, y_pred, pos_label=pos_label, **kwargs)
        ax = disp.ax_
        ax.set_title(f'ROC Curve [{name}]')
        figures[name] = disp.figure_

    return figures

def plot_partial_dependence(
    estimators   : pd.Series,
    X            : pd.DataFrame,
    cfg          : dict,
    random_state : np.random.RandomState,
):
    features = cfg.pop('features', X.columns)
    for name, est in estimators.items():
        disp = inspection.PartialDependenceDisplay.from_estimator(
            est,
            X,
            features,
            random_state=random_state,
            **cfg
        )
        yield name, disp.figure_
