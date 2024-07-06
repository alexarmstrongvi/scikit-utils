"""Utilities for working with and plotting ROC Curves"""
# Standard library
from collections.abc import Collection, Sequence
from functools import cached_property, partial
import logging
from typing import Literal, NamedTuple

# 3rd party
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats
import seaborn as sns
from seaborn import FacetGrid
from shapely import LineString
import sklearn.metrics

# Globals
log = logging.getLogger(__name__)

# Alias
NDArrayFloat = npt.NDArray[np.floating]


################################################################################
def plot_roc_curve_comparison(
    data         : pd.DataFrame | None = None,
    *,
    y_true       : str | NDArrayFloat = 'y_true',
    y_score      : str | NDArrayFloat = 'y_score',
    # Optional
    roc_kwargs : dict | None = None,
    **kwargs
    # hue : str | None = None,
    # col : str | None = None,
    # row : str | None = None,
    # col_wrap : int | None = None,
    # row_order: list[str] | None = None,
    # col_order: list[str] | None = None,
) -> FacetGrid:
    # Argument check
    data, col_names = _require_dataframe(
        data,
        y_true  = y_true,
        y_score = y_score,
        hue     = kwargs.get('hue'),
        col     = kwargs.get('col'),
        row     = kwargs.get('row'),
    )
    y_true, y_score = col_names[:2]

    grid = sns.FacetGrid(data, **kwargs)
    roc_kwargs = roc_kwargs or {}
    grid.map(_facetgrid_plot_roc_curve, y_true, y_score, **roc_kwargs)
    grid.add_legend()
    return grid

def plot_roc_curve_grid(
    data         : pd.DataFrame | None = None,
    *,
    y_true       : str | NDArrayFloat,
    y_score      : str | NDArrayFloat,
    # Optional
    ci_band_rocs : list[pd.DataFrame] | None = None,
) -> tuple[Figure, Axes]:
    data, (y_true, y_score) = _require_dataframe(data, y_true=y_true, y_score=y_score)
    roc  = roc_curve(data, y_true=y_true, y_score=y_score, drop_intermediate=False)

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10,10), layout='tight')

    ax = axs[0][0]
    # TODO: Add thresholds as vlines
    sns.histplot(data, x=y_score, hue=y_true, bins=25, ax = ax)
    ax.legend(title=None, labels=['TP', 'FP'])
    ax.set_title('Score distribution')
    # ax.semilogy()

    plot_rate_curve(roc.index, roc['fpr'], label='FPR', ci_width=95, n_samples=len(data), ax=axs[0][1])
    plot_rate_curve(roc.index, roc['tpr'], label='TPR', ci_width=95, n_samples=len(data), ax=axs[1][0])

    plot_roc_curve(
        data            = roc.reset_index(),
        plot_thresholds = 'percentile',
        plot_diagonal   = True,
        ci_band_rocs    = ci_band_rocs,
        ax              = axs[1][1],
    )
    return fig, axs

def _facetgrid_plot_roc_curve(
    y_true  : str,
    y_score : str,
    data = None,
    *,
    n_resamples: int = 0,
    color = None,
    label = None,
    # plot_roc_curve kwargs
    **kwargs,
) -> Axes:
    data, (y_true, y_score) = _require_dataframe(data, y_true=y_true, y_score=y_score)
    roc  = roc_curve(data, y_true=y_true, y_score=y_score, drop_intermediate=False)
    bootstrap_rocs = None
    if n_resamples > 0:
        bootstrap_rocs = bootstrap(
            data,
            y_true=y_true,
            y_score=y_score,
            n_resamples=n_resamples,
            drop_intermediate=False,
        )
    ax = plot_roc_curve(
        data         = roc.reset_index(),
        ci_band_rocs = bootstrap_rocs,
        color        = color,
        label        = label,
        ax           = plt.gca(),
        **kwargs
    )
    # ax.set_title(None)
    return ax

def plot_roc_curve(
    data: pd.DataFrame | None = None,
    *,
    # Required
    fpr             : str | NDArrayFloat = 'fpr',
    tpr             : str | NDArrayFloat = 'tpr',
    thresholds      : str | NDArrayFloat = 'thresholds',
    # Optional
    plot_thresholds : str | Sequence | None = None,
    n_thresholds    : int = 5,
    plot_diagonal   : bool = False,
    ci_band_rocs    : list[pd.DataFrame] | None = None,
    ci_width        : int = 95,
    color           : str | tuple[float,float,float] | None = None,
    label           : str | None = None,
    ax              : Axes | None = None,
) -> Axes:
    '''
    Example usage:
    ax = plot_roc_curve(data=df, fpr='fpr', tpr='tpr', thresholds='thresholds')
    ax = plot_roc_curve(fpr=fpr, tpr=tpr, thresholds=thresholds)
    plot_roc_curve(fpr=fpr, tpr=tpr, thresholds=thresholds, ax=ax)
    '''
    if ax is None:
        ax = plt.subplot()
    data, (fpr, tpr, thresholds) = _require_dataframe(
        df  = data,
        fpr = fpr,
        tpr = tpr,
        thresholds = thresholds,
    )

    handles = []
    labels = []

    if ci_band_rocs is not None:
        roc_ci_bands = compute_roc_ci_bands(ci_band_rocs, 'tpr_averaging', ci_width)
        _, hdl_ci_band = plot_roc_ci_bands(
            roc_ci_bands,
            color = color,
            ax    = ax,
        )

    hdl_roc, = ax.plot(data['fpr'], data['tpr'], color=color, label=label)

    if ci_band_rocs is not None:
        handles.append((hdl_roc, hdl_ci_band))
        labels.append(rf'ROC $\pm$ {ci_width}% CI')

    if plot_diagonal:
        # TODO: Handle ROC curves that is not normalized between 0 and 1
        tpr_min = data[tpr].iloc[0]
        tpr_max = data[tpr].iloc[-1]
        fpr_min = data[fpr].iloc[0]
        fpr_max = data[fpr].iloc[-1]
        hdl_diag, = ax.plot([fpr_min, fpr_max], [tpr_min, tpr_max], color='k', linestyle=':')

    if plot_thresholds is not None:
        roc = ROCCurve(
            fpr = data[fpr].to_numpy(),
            tpr = data[tpr].to_numpy(),
            thresholds = data[thresholds].to_numpy()
        )
        if isinstance(plot_thresholds, str):
            fpr_interp, tpr_interp, thr_to_plot = roc.compute_thresholds(plot_thresholds, n_thresholds)
        elif isinstance(plot_thresholds, Sequence):
            thr_to_plot = plot_thresholds
            fpr_interp, tpr_interp, _ = roc.interpolate(thresholds=thr_to_plot)
        hdl_thr = ax.scatter(fpr_interp, tpr_interp, color='k', marker='x')

        handles.append(hdl_thr)
        labels.append('Ref thresholds')

    if ax.get_title() is None:
        title = 'ROC Curve'
        if plot_thresholds:
            thr_str = ', '.join(f'{x:.2f}' for x in reversed(thr_to_plot))
            title += f'\nThresholds: > {thr_str}'
        ax.set_title(title)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.grid()
    ax.legend(handles, labels)
    return ax

def plot_rate_curve(
    thresholds,
    rate,
    label,
    ci_width = 0,
    n_samples = 0,
    ax = None
) -> Axes:
    if ax is None:
        ax = plt.subplot()
    if ci_width > 0 and n_samples > 0:
        count_hi, count_lo = scipy.stats.binom.interval(ci_width/100, n=n_samples, p=rate)
        rate_hi, rate_lo = count_hi/n_samples, count_lo/n_samples
    rate_pts = ax.scatter(thresholds, rate, marker='.')
    ci_bands = ax.fill_between(thresholds, rate_lo, rate_hi, alpha=0.2, color='tab:blue')
    ax.set_xlabel('Threshold')
    ax.set_ylabel(label)
    ax.grid()
    ax.legend([(rate_pts, ci_bands)],[rf'{label} $\pm$ {ci_width}% CL'])
    return ax

def plot_roc_ci_bands(
    data,
    *,
    fpr_hi = 'fpr_hi',
    fpr_lo = 'fpr_lo',
    tpr_hi = 'tpr_hi',
    tpr_lo = 'tpr_lo',
    color  = None,
    ax : Axes | None = None,
) -> tuple[Axes, list[Polygon]]:
    if ax is None:
        ax = plt.subplot()
    data, (fpr_hi, fpr_lo, tpr_hi, tpr_lo) = _require_dataframe(
        data,
        fpr_hi = fpr_hi,
        fpr_lo = fpr_lo,
        tpr_hi = tpr_hi,
        tpr_lo = tpr_lo,
    )
    x = np.concatenate([
        data[fpr_lo],
        data[fpr_hi][::-1],
    ])
    y = np.concatenate([
        data[tpr_lo],
        data[tpr_hi][::-1],
    ])
    polygon, = ax.fill(x, y, color=color, alpha=0.2)
    return ax, polygon

def is_monotonic_increasing(arr):
    return np.all(arr[1:] >= arr[:-1])
def is_monotonic_decreasing(arr):
    return np.all(arr[1:] <= arr[:-1])

class ROCPoint(NamedTuple):
    fpr : float
    tpr : float
    threshold : float

# TODO: Define and then inherit from ParametricCurve2D(x,y,t):
class ROCCurve:
    _interp = partial(np.interp, left=np.nan, right=np.nan)

    def __init__(
        self,
        fpr        : NDArrayFloat,
        tpr        : NDArrayFloat,
        thresholds : NDArrayFloat,
        validate   : bool = True,
    ):
        if validate and not (
            is_monotonic_increasing(fpr)
            and is_monotonic_increasing(tpr)
            and is_monotonic_decreasing(thresholds)
        ):
            sort_idxs  = np.argsort(fpr)
            fpr        = fpr[sort_idxs]
            tpr        = tpr[sort_idxs]
            thresholds = thresholds[sort_idxs]
            if not (is_monotonic_increasing(tpr) and is_monotonic_decreasing(thresholds)):
                raise ValueError('ROC curve FPR and TPR not monotonically increasing with decreasing threshold')

        self.fpr = fpr
        self.tpr = tpr
        self.thresholds = thresholds


    @cached_property
    def curve(self):
        return LineString(zip(self.fpr, self.tpr))

    @cached_property
    def midpoint(self) -> ROCPoint:
        point = self.curve.interpolate(0.5, normalized=True)
        fpr, tpr = point.x, point.y
        threshold_fpr = self._interp(fpr, self.fpr, self.thresholds)
        threshold_tpr = self._interp(tpr, self.tpr, self.thresholds)
        threshold = (threshold_fpr + threshold_tpr) / 2
        return ROCPoint(fpr, tpr, threshold)

    def interpolate(
        self,
        *,
        fpr        : float | NDArrayFloat | None = None,
        tpr        : float | NDArrayFloat | None = None,
        thresholds : float | NDArrayFloat | None = None,
    ) -> tuple[float | NDArrayFloat, float | NDArrayFloat, float | NDArrayFloat]:
        # Check inputs
        n_args = sum(map(lambda x : x is not None,[fpr, tpr, thresholds]))
        if n_args == 0:
            raise ValueError('No args provided but one is required')

        # Interpolate
        # TODO: Handle edge case where
        if fpr is None:
            fpr_interp = self.interpolate_fpr(tpr=tpr, thresholds=thresholds)
        if tpr is None:
            tpr_interp = self.interpolate_tpr(fpr=fpr, thresholds=thresholds)
        if thresholds is None:
            # TODO: Handle interpolating outside bounds (e.g. np.inf) to get
            # fpr = 0 or tpr = 0
            thresholds_interp = self.interpolate_threshold(fpr=fpr, tpr=tpr)

        return (
            fpr_interp if fpr is None else fpr,
            tpr_interp if tpr is None else tpr,
            thresholds_interp if thresholds is None else thresholds,
        )

    def interpolate_fpr(
        self,
        *,
        tpr        : float | NDArrayFloat | None = None,
        thresholds : float | NDArrayFloat | None = None,
    ) -> float | NDArrayFloat:
        return self._interpolate_generic(
            x1          = tpr,
            xp1         = self.tpr,
            fp1         = self.fpr,
            x2          = thresholds,
            xp2         = self.thresholds[::-1],
            fp2         = self.fpr[::-1],
        )
    def interpolate_tpr(
        self,
        *,
        fpr        : float | NDArrayFloat | None = None,
        thresholds : float | NDArrayFloat | None = None,
    ) -> float | NDArrayFloat:
        return self._interpolate_generic(
            x1          = fpr,
            xp1         = self.fpr,
            fp1         = self.tpr,
            x2          = thresholds,
            xp2         = self.thresholds[::-1],
            fp2         = self.tpr[::-1],
        )
    def interpolate_threshold(
        self,
        *,
        fpr : float | NDArrayFloat | None = None,
        tpr : float | NDArrayFloat | None = None,
    ) -> float | NDArrayFloat:
        return self._interpolate_generic(
            x1          = fpr,
            xp1         = self.fpr,
            fp1         = self.thresholds,
            x2          = tpr,
            xp2         = self.tpr,
            fp2         = self.thresholds,
        )
    def _interpolate_generic(self, x1, xp1, fp1, x2, xp2, fp2):
        if x1 is not None:
            f = self._interp(x1, xp1, fp1)
        if x2 is not None:
            f_from_x2 = self._interp(x2, xp2, fp2)
            if x1 is None:
                f = f_from_x2
            else:
                f_from_x1 = f
                # Interpolation from either source is possible so use results
                # depending on when each is expected to give better results.
                # TODO: Handle np.inf leading to RuntimeWarning
                idx_after1 = np.searchsorted(xp1, x1)
                idx_before1 = np.clip(idx_after1 - 1, 0, None)
                slopes1 = (fp1[idx_after1]-fp1[idx_before1])/(xp1[idx_after1]-xp1[idx_before1])
                idx_after2 = np.searchsorted(xp2, x2)
                idx_before2 = np.clip(idx_after2 - 1, 0, None)
                slopes2 = (fp2[idx_after2]-fp2[idx_before2])/(xp2[idx_after2]-xp2[idx_before2])
                # print(f"""
                # FPR
                # \tIdx   : {idx_after1} {idx_before1}
                # \tx     : {xp1[idx_after1]} {xp1[idx_before1]}
                # \tf(x)  : {fp1[idx_after1]} {fp1[idx_before1]}
                # \tslope : {slopes1}
                # TPR
                # \tIdx   : {idx_after2} {idx_before2}
                # \tx     : {xp2[idx_after2]} {xp2[idx_before2]}
                # \tf(x)  : {fp2[idx_after2]} {fp2[idx_before2]}
                # \tslope : {slopes2}
                # """)
                f = np.where(np.abs(slopes1) < np.abs(slopes2), f_from_x1, f_from_x2)
        return f

    def interpolate_distance(
        self,
        distance: float | Collection,
        normalized: bool = True,
    ) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        print('DEBUG | distance =', distance)
        points = self.curve.interpolate(distance, normalized)
        arr = np.array([[p.x, p.y] for p in points])
        fpr, tpr = arr[:,0], arr[:,1]
        thresholds = self.interpolate_threshold(fpr=fpr, tpr=tpr)
        return fpr, tpr, thresholds

    def compute_thresholds(
        self,
        method : Literal['linear', 'log', 'percentile', 'evenly_spaced'],
        n_thresholds,
    ) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        thresholds_noinf = self.thresholds
        if np.isinf(self.thresholds[0]):
            thresholds_noinf = self.thresholds[1:]
        thr_max = thresholds_noinf[0]
        thr_min = self.thresholds[-1]

        fpr_interp = tpr_interp = None
        if method == 'linear':
            thr_to_plot = np.linspace(thr_min, thr_max, n_thresholds)
        elif method == 'log':
            thr_to_plot = np.logspace(np.log10(thr_min), np.log10(thr_max), n_thresholds)
        elif method == 'percentile':
            q = [50] if n_thresholds == 1 else np.linspace(0,100, n_thresholds)
            thr_to_plot = np.percentile(thresholds_noinf, q)
        elif method == 'evenly_spaced':
            distances = [0.5] if n_thresholds == 1 else np.linspace(0, 1, n_thresholds)
            fpr_interp, tpr_interp, thr_to_plot = self.interpolate_distance(distances)
        else:
            raise ValueError(f'Unexpected threshold method: {method}')
        print('DEBUG | Plotting thresholds:', thr_to_plot)

        if fpr_interp is None or tpr_interp is None:
            fpr_interp, tpr_interp, _ = self.interpolate(thresholds=thr_to_plot)
        return fpr_interp, tpr_interp, thr_to_plot

def roc_curve(
    data = None,
    *,
    y_true = 'y_true',
    y_score = 'y_score',
    **kwargs,
) -> pd.DataFrame:
    '''Wrapper for sklearn.metrics.roc_curve that can accept and will always
    return a DataFrame'''
    data, (y_true, y_score) = _require_dataframe(data, y_true=y_true, y_score=y_score)
    return pd.DataFrame(
        data = np.column_stack(
            sklearn.metrics.roc_curve(
                y_true  = data[y_true],
                y_score = data[y_score],
                **kwargs,
            )
        ),
        columns = ['fpr', 'tpr', 'thresholds'],
    ).set_index('thresholds')

def bootstrap(
    data = None,
    *,
    y_true,
    y_score,
    n_resamples,
    rng = None,
    **kwargs,
) -> list[pd.DataFrame]:
    data, (y_true, y_score) = _require_dataframe(data, y_true=y_true, y_score=y_score)
    log.info('Bootstrapping ROC curves')
    return [
        roc_curve(
            # TODO: Sample from smoothed distribution
            data    = data.sample(frac=1, replace=True, random_state=rng),
            y_true  = y_true,
            y_score = y_score,
            **kwargs,
        )
        for _ in range(n_resamples)
    ]

def _require_dataframe(df = None, **columns) -> tuple[pd.DataFrame, list]:
    """Force output to be a dataframe and column names whether user provides
    that initially or provides arrays for the columns instead"""
    columns = {k:v for k,v in columns.items() if v is not None}
    if isinstance(df, pd.DataFrame):
        columns = list(columns.values())
        df = df[columns]
    elif df is None:
        df = pd.DataFrame(columns)
        columns = df.columns.tolist()
    else:
        raise TypeError(f'Arg is of type {type(df)} but DataFrame is expected')
    return df, columns

def merge_roc_curves(
    roc_curves : Collection[pd.DataFrame],
    index      : str | None = None,
    level_name : str | None = None,
) -> pd.DataFrame:
    if index is not None:
        roc_curves = [
            df_.drop_duplicates(subset=index, keep='first').set_index(index)
            for df_ in roc_curves
        ]
    return (
        # TODO: Handle big dataframes as this will probably become to slow
        pd.concat(
            roc_curves,
            axis  = 1,
            keys  = range(len(roc_curves)),
            names = [level_name] if level_name is not None else None,
        )
        .sort_index()
        .interpolate('index', limit_direction='both', limit_area='inside')
    )

def _compute_roc_ci_band_rate_avg(
    roc_curves: Collection[pd.DataFrame],
    ci_width: float,
    sweep_axis: str,
) -> pd.DataFrame:
    roc_curves_wide = merge_roc_curves(roc_curves, index=sweep_axis)
    avg_axis = roc_curves_wide.columns.levels[1][0]

    q_lo = (100 - ci_width)/2/100
    q_hi = (q_lo + ci_width)/100

    roc_ci_bands = (
        roc_curves_wide
        .quantile([q_lo, q_hi], axis=1)
        .transpose()
        .rename(columns={
            q_lo : f'{avg_axis}_lo',
            q_hi : f'{avg_axis}_hi',
        })
        .assign(**{
            f'{sweep_axis}_lo' : lambda df_ : df_.index,
            f'{sweep_axis}_hi' : lambda df_ : df_.index,
        })
    ).rename_axis(columns=f'{ci_width}% CI Bands')
    return roc_ci_bands

def compute_roc_ci_bands(
    roc_curves: Collection[pd.DataFrame],
    method: str,
    ci_width: float,
    **kwargs
) -> pd.DataFrame:
    # References
    # - "Confidence Bands for ROC Curves" by S. Macskassy, Feb 2004
    # - "ROC Confidence Bands: An Empirical Evaluation" by S. Mackassy, 2005
    # - "On Bootstrapping the ROC Curve" by P. Bertail, 2008
    #   - "On constructing accurate confidence bands for ROC curves through smooth resampling" by P. Bertail, Oct 2008

    if method == 'tpr_averaging':
        # 1a) Vertical averaging (VA)
        return _compute_roc_ci_band_rate_avg(roc_curves, ci_width, sweep_axis='fpr', **kwargs)
    elif method == 'fpr_averaging':
        # 1b) Horizontal averaging (HA)
        return _compute_roc_ci_band_rate_avg(roc_curves, ci_width, sweep_axis='tpr', **kwargs)
    elif method == 'threshold_averaging':
        # 2) Threshold averaging (TA)
        raise NotImplementedError()
    elif method == 'SJR':
        # 3) Simultaneous joint confidence regions (SRJ)
        raise NotImplementedError()
    elif method == 'WHB':
        # 4) Working-Hotelling based bands (WHB)
        raise NotImplementedError()
    elif method == 'FWB':
        # 5) Fixed width confidence bands (FWB)
        raise NotImplementedError()
    raise ValueError(f'Unexpected method: {method}')
