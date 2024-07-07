# see User Guide "Column Transformer with Mixed Types"

# Standard library
from pathlib import Path

# 3rd party
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import yaml

# 1st party
from skutils.pipeline import make_pipeline

# Globals
DATA_DIR = Path(__file__).parents[1] / 'data'
np.random.seed(0)

################################################################################
X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)

# Alternatively X and y can be obtained directly from the frame attribute:
# X = titanic.frame.drop('survived', axis=1)
# y = titanic.frame['survived']

with (DATA_DIR/'example_nested_pipeline.yml').open('r') as ifile:
    cfg = yaml.safe_load(ifile)
clf = make_pipeline(**cfg['estimator']['make_pipeline'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf.fit(X_train, y_train)
print("model score: %.3f" % clf.score(X_test, y_test))
