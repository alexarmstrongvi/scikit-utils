# Standard library
from pathlib import Path

# 3rd party
import yaml

# Globals
DATA_DIR = Path(__file__).parent

def get_default_cfg_supervised():
    path = DATA_DIR / 'run_fit_supervised_model.yml'
    with path.open('r') as ifile:
        return yaml.safe_load(ifile)
