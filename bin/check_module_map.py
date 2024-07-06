# Standard library
from collections import defaultdict
import inspect
from pprint import pprint
import re

# 3rd party
from sklearn.base import (  # OutlierMixin,; ClusterMixin,; RegressorMixin,; ClassifierMixin,; MultiOutputMixin,; OneToOneFeatureMixin,; ClassNamePrefixFeaturesOutMixin,
    BiclusterMixin,
    DensityMixin,
    MetaEstimatorMixin,
    TransformerMixin
)
from sklearn.cluster._bicluster import BaseSpectral
from sklearn.decomposition._base import _BasePCA
from sklearn.feature_selection._base import SelectorMixin
from sklearn.utils import all_estimators

# 1st party
from skutils.import_utils import _MODULE_MAP

estimators = defaultdict(dict)
missing = defaultdict(dict)
all_ests = dict(all_estimators())

def get_estimator_type(est, module):
    global all_ests

    tags = []
    est_type = getattr(est, '_estimator_type', None)
    # is_reg = is_regressor(est)
    # is_cls = is_classifier(est)
    # is_out = is_outlier_detector(est)

    if isinstance(est_type, str):
        pass
    elif issubclass(est, BaseSpectral):
        est_type = 'clusterer Spectral'
    elif issubclass(est, BiclusterMixin):
        est_type = 'clusterer Bicluster'
    elif issubclass(est, DensityMixin):
        est_type = 'Density'
    elif issubclass(est, _BasePCA):
        est_type = 'PCA'
    elif issubclass(est, SelectorMixin):
        est_type = 'Selector'
    elif module in {'sklearn.manifold', 'sklearn.mixture'}:
        est_type = 'Unsupervised'
    elif issubclass(est, TransformerMixin):
        # Lots of classes ultimately inherit from TransformerMixin so check this
        # last
        est_type = 'Transformer'
    else:
        est_type = 'Unknown'

    is_meta1 = issubclass(est, MetaEstimatorMixin)
    is_meta2 = any(
        # Some classes (e.g. ExtraTreesClassifier, IsolationForest) inherit from
        # MetaEstimatorMixin but do not actually take estimators as an
        # argument
        x in inspect.signature(est).parameters
        for x in ('estimator', 'estimators', 'base_estimator', 'regressor', 'classifier')
    )
    if is_meta2 and not is_meta1:
        print('WARNING | Meta estimator that does not inherit from mixin: %s' % est)
    is_meta = is_meta2

    is_cv_est = est.__name__.endswith('CV') and est.__name__[:-2] in all_ests

    tags.append(est_type)
    if is_meta:
        tags.append('Meta')
    if is_cv_est:
        tags.append('CV')
    return tuple(tags)

for name, est in all_ests.items():
    match = re.search(r'(sklearn\..*)\._', str(est))
    if match is None:
        match = re.search(r'(sklearn\..*)\'>', str(est))
    module = match.group(1)
    module = module.replace(name, '').strip('.')
    est_type = get_estimator_type(est, module)
    estimators[est_type][name] = module
    if name not in _MODULE_MAP:
        missing[est_type][name] = module

knowns = {
    est_type : {
        k:v for k,v in sorted(ests.items(), key=lambda kv : kv[1])
    } for est_type, ests in sorted(estimators.items())
    if est_type[0] != 'Unknown'
}
unknowns = {
    est_type : {
        k:v for k,v in sorted(ests.items(), key=lambda kv : kv[1])
    } for est_type, ests in sorted(estimators.items())
    if est_type[0] == 'Unknown'
}
# pprint(knowns, sort_dicts=False)
# pprint(unknowns, sort_dicts=False)
pprint(missing, sort_dicts=False)
