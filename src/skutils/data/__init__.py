# Standard library
from pathlib import Path
import logging

# 3rd party
import yaml

# Globals
log = logging.getLogger(__name__)
DATA_DIR = Path(__file__).parent

def get_default_cfg_supervised():
    path = DATA_DIR / 'fit_supervised_model.yml'
    with path.open('r') as ifile:
        return yaml.safe_load(ifile)