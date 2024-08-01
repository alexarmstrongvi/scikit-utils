# Standard library
import argparse
import logging
from pathlib import Path

# 3rd party
import pandas as pd
from sklearn.datasets import make_classification, make_regression

# 1st party
from skutils import logging_utils, scripting

# Globals
log = logging.getLogger(Path(__file__).stem)

################################################################################
def main():
    args = parse_argv()
    scripting.require_empty_dir(args.odir, overwrite=args.overwrite)
    logging_utils.configure_logging(level = args.log_level)

    # Classification data
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

    opath = args.odir / 'classification_data.feather'
    data.to_feather(opath)
    log.info('Classification data saved: %s', opath)

    # Regression data
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
    opath = args.odir / 'regression_data.feather'
    data.to_feather(opath)
    log.info('Regression data saved: %s', opath)

def parse_argv() -> argparse.Namespace:
    """Parse command line arguments (i.e. sys.argv)"""
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "-l",
        "--log-level",
        choices=logging_utils.LOG_LEVEL_CHOICES,
        help="Root logging level",
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()
