"""Utilities for saving things to disk"""

# Standard library
import logging
from pathlib import Path
import pickle

# 3rd party
import pandas as pd
from sklearn.base import BaseEstimator

# Globals
log = logging.getLogger(__name__)

################################################################################
def dump_model(
    est  : BaseEstimator,
    opath: Path,
    X    : pd.DataFrame | None = None,
) -> None:
    '''Save model to disk in one of several supported formats

    Parameters
    ==========
    est
    opath
    X:
        Example row required for ONNX. All features must be float dtypes
    '''
    # NOTE: Importing 3rd party libraries only if necessary
    fmt = opath.suffix[1:]
    if fmt == 'pkl':
        with opath.open('wb') as ofile:
            pickle.dump(est, ofile, protocol=pickle.HIGHEST_PROTOCOL)
    elif fmt == 'joblib':
        # TODO: Test this works
        # 3rd party
        import joblib
        with opath.open('wb') as ofile:
            joblib.dump(est, ofile, protocol=pickle.HIGHEST_PROTOCOL)
    elif fmt == 'dill':
        # TODO: Test this works
        # 3rd party
        import dill
        with opath.open('wb') as ofile:
            dill.dump(est, ofile, protocol=pickle.HIGHEST_PROTOCOL)
    elif fmt == 'cloudpickle':
        # TODO: Test this works
        # 3rd party
        import cloudpickle
        with opath.open('wb') as ofile:
            cloudpickle.dump(est, ofile, protocol=pickle.HIGHEST_PROTOCOL)
    elif fmt == 'skops':
        # TODO: Test this works
        # 3rd party
        import skops.io as sio
        sio.dump(est, opath)
    elif fmt == 'onnx':
        # TODO: Test this works
        # 3rd party
        from skl2onnx import to_onnx
        onx = to_onnx(est, X.iloc[:1])
        with opath.open('wb') as ofile:
            ofile.write(onx.SerializeToString())
    else:
        raise ValueError(
            f'Unrecognized file extension {fmt!r} from {opath.name}. '
            f'Choose from the following: pkl, joblib, dill, cloudpickle, skops, onnx',
        )

def dump_pandas(data : pd.DataFrame | pd.Series, opath: Path) -> None:
    fmt = opath.suffix[1:]
    # Text format
    if fmt == 'csv':
        data.to_csv(opath)
    elif fmt == 'json':
        data.to_json(opath)
    elif fmt == 'html':
        data.to_html(opath)
    elif fmt == 'xml':
        data.to_xml(opath)
    # Binary format
    elif fmt == 'xlsx':
        data.to_excel(opath)
    elif fmt == 'hdf' or fmt == 'hdf5':
        data.to_hdf(opath)
    elif fmt == 'feather':
        data.to_feather(opath)
    elif fmt == 'parquet':
        data.to_parquet(opath)
    elif fmt == 'orc':
        data.to_orc(opath)
    elif fmt == 'stata':
        data.to_stata(opath)
    else:
        raise ValueError(
            f'Unrecognized file extension {fmt!r} from {opath.name}. '
            'Choose from the following: '
            'csv, json, html, xml, '
            'xlsx, hdf/hdf5, feather, parquet, orc, stata'
        )
