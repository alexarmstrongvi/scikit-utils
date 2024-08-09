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
    suffix = opath.suffix
    if suffix == '.pkl':
        with opath.open('wb') as ofile:
            pickle.dump(est, ofile, protocol=pickle.HIGHEST_PROTOCOL)
    elif suffix == '.joblib':
        # TODO: Test this works
        # 3rd party
        import joblib
        with opath.open('wb') as ofile:
            joblib.dump(est, ofile, protocol=pickle.HIGHEST_PROTOCOL)
    elif suffix == '.dill':
        # TODO: Test this works
        # 3rd party
        import dill
        with opath.open('wb') as ofile:
            dill.dump(est, ofile, protocol=pickle.HIGHEST_PROTOCOL)
    elif suffix == '.cloudpickle':
        # TODO: Test this works
        # 3rd party
        import cloudpickle
        with opath.open('wb') as ofile:
            cloudpickle.dump(est, ofile, protocol=pickle.HIGHEST_PROTOCOL)
    elif suffix == '.skops':
        # TODO: Test this works
        # 3rd party
        import skops.io as sio
        sio.dump(est, opath)
    elif suffix == '.onnx':
        # TODO: Test this works
        # 3rd party
        from skl2onnx import to_onnx
        onx = to_onnx(est, X.iloc[:1])
        with opath.open('wb') as ofile:
            ofile.write(onx.SerializeToString())
    else:
        raise ValueError(
            f'Unrecognized file extension {suffix!r} from {opath.name}. '
            f'Choose from the following: pkl, joblib, dill, cloudpickle, skops, onnx',
        )

def dump_pandas(
    data : pd.DataFrame | pd.Series,
    opath: Path,
    **kwargs,
) -> None:
    suffix = opath.suffix
    # Text format
    if suffix == '.csv':
        data.to_csv(opath, **kwargs)
    elif suffix == '.json':
        data.to_json(opath, **kwargs)
    elif suffix == '.html':
        data.to_html(opath, **kwargs)
    elif suffix == '.xml':
        data.to_xml(opath, **kwargs)
    # Binary format
    elif suffix == '.xlsx':
        data.to_excel(opath, **kwargs)
    elif suffix == '.hdf' or suffix == '.hdf5':
        data.to_hdf(opath, **kwargs)
    elif suffix == '.feather':
        data.to_feather(opath, **kwargs)
    elif suffix == '.parquet':
        data.to_parquet(opath, **kwargs)
    elif suffix == '.orc':
        data.to_orc(opath, **kwargs)
    elif suffix == '.stata':
        data.to_stata(opath, **kwargs)
    else:
        raise ValueError(
            f'Unrecognized file extension {suffix!r} from {opath.name}. '
            'Choose from the following: '
            'csv, json, html, xml, '
            'xlsx, hdf/hdf5, feather, parquet, orc, stata'
        )

def read_pandas(path: Path, **kwargs) -> pd.DataFrame:
    suffix = path.suffix
    # Text format
    if suffix == '.csv':
        data = pd.read_csv(path, **kwargs)
    elif suffix == '.json':
        data = pd.read_json(path, **kwargs)
    elif suffix == '.html':
        data = pd.read_html(path, **kwargs)
    elif suffix == '.xml':
        data = pd.read_xml(path, **kwargs)
    # Binary format
    elif suffix in {'.xls', '.xlsx', '.xlsm', '.xlsb', '.odf', '.ods', '.odt'}:
        data = pd.read_excel(path, **kwargs)
    elif suffix == '.hdf' or suffix == '.hdf5':
        data = pd.read_hdf(path, **kwargs)
    elif suffix == '.feather':
        data = pd.read_feather(path, **kwargs)
    elif suffix == '.parquet':
        data = pd.read_parquet(path, **kwargs)
    elif suffix == '.orc':
        data = pd.read_orc(path, **kwargs)
    elif suffix == '.dta':
        data = pd.read_stata(path, **kwargs)
    elif suffix == '.xpt' or suffix == '.sas7bdat':
        data = pd.read_sas(path, **kwargs)
    elif suffix == '.sav' or suffix == '.zsav':
        data = pd.read_spss(path, **kwargs)
    elif suffix == '.pkl':
        data = pd.read_pickle(path, **kwargs)
    else:
        raise NotImplementedError(
            f'No reader implemented for files of type {path.suffix}'
        )
    return data
