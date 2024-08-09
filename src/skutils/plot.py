
# 3rd party
from matplotlib.figure import Figure
import pandas as pd
from sklearn import calibration, inspection, metrics, model_selection
from sklearn.utils.validation import _check_pos_label_consistency


################################################################################
def plot_confusion_matrix(
    y_true       : pd.DataFrame,
    y_pred       : pd.DataFrame,
    is_test_data : pd.DataFrame,
    display_kw   : dict | None = None,
) -> dict[str, Figure]:
    kwargs = display_kw or {}
    figures = {}
    if 'split' in y_pred.index.names:
        iterable = [('Combined Splits', (y_true, y_pred['predict']))]
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
        figures[name] = disp.figure_

    return figures

def plot_roc_curve(
    y_true       : pd.DataFrame,
    y_pred       : pd.DataFrame,
    is_test_data : pd.DataFrame,
    pos_label    : int | str | None = None,
    display_kw   : dict | None = None,
) -> dict[str, Figure]:
    kwargs = display_kw or {}
    pos_label = _check_pos_label_consistency(pos_label, y_true)
    prob_col = f'prob({pos_label})'

    figures = {}
    if 'split' in y_pred.index.names:
        iterable = [('combined_splits', (y_true, y_pred[prob_col]))]
    elif 'split' in y_pred.columns.names:
        splits = y_pred.columns.get_level_values('split').unique()
        iterable = []
        for split in splits:
            filt = is_test_data[split]
            y_true_split = y_true.loc[filt]
            y_pred_split = y_pred.loc[filt, (split, prob_col)]
            iterable.append((split, (y_true_split, y_pred_split)))

    # TODO: Decide how to plot multiple ROC curves for each split: Overlay in a
    # single plot or separate plots? How to set title and filename?
    for name, (y_true, y_pred) in iterable:
        disp = metrics.RocCurveDisplay.from_predictions(y_true, y_pred, pos_label=pos_label, **kwargs)
        ax = disp.ax_
        ax.set_title(f'ROC Curve [{name}]')
        figures[name] = disp.figure_

    return figures
