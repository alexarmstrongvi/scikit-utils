# Standard library
import importlib
import logging
import sys
from typing import Type

# Globals
log = logging.getLogger(__name__)

################################################################################
def import_object(name: str) -> Type:
    if name in _MODULE_MAP:
        name = f'{_MODULE_MAP[name]}.{name}'
    # else:
    #     log.debug('Unregistered import name: %s', name)
    module_name, class_name = name.rsplit('.', maxsplit=1)
    if module_name not in sys.modules:
        log.debug('Importing new module: `import %s`', module_name)
    log.debug('Importing object: `from %s import %s`', module_name, class_name)
    return getattr(importlib.import_module(module_name), class_name)

# Map from common objects to their modules to simplify calling import_object
_MODULE_MAP = {
    #### Estimator ####
    # Classifier
    'RidgeClassifier'                : 'sklearn.linear_model',
    'SGDClassifier'                  : 'sklearn.linear_model',
    'PassiveAggressiveClassifier'    : 'sklearn.linear_model',
    'KNeighborsClassifier'           : 'sklearn.neighbors',
    'RadiusNeighborsClassifier'      : 'sklearn.neighbors',
    'GaussianProcessClassifier'      : 'sklearn.gaussian_process',
    'DecisionTreeClassifier'         : 'sklearn.tree',
    'ExtraTreeClassifier'            : 'sklearn.tree',
    'MLPClassifier'                  : 'sklearn.neural_network',
    'AdaBoostClassifier'             : 'sklearn.ensemble',
    'BaggingClassifier'              : 'sklearn.ensemble',
    'ExtraTreesClassifier'           : 'sklearn.ensemble',
    'GradientBoostingClassifier'     : 'sklearn.ensemble',
    'HistGradientBoostingClassifier' : 'sklearn.ensemble',
    'RandomForestClassifier'         : 'sklearn.ensemble',
    # TODO: 'XGBClassifier'                  : 'xgboost',
    # TODO: 'LGBMClassifier'                 : 'lightgbm',
    # Classifier Wrappers
    'StackingClassifier'             : 'sklearn.ensemble',
    'VotingClassifier'               : 'sklearn.ensemble',
    'MultiOutputClassifier'          : 'sklearn.multioutput',
    'OneVsOneClassifier'             : 'sklearn.multiclass',
    'OneVsRestClassifier'            : 'sklearn.multiclass',
    'OutputCodeClassifier'           : 'sklearn.multiclass',
    'SelfTrainingClassifier'         : 'sklearn.semi_supervised',
    # Other
    'DummyClassifier'                : 'sklearn.dummy',

    # Regressor
    'ExtraTreesRegressor'            : 'sklearn.ensemble',

    #### Splitter / CrossValidator ####
    # K-Fold CV - test sets form a partition across splits
    'KFold'                   : 'sklearn.model_selection',
    'RepeatedKFold'           : 'sklearn.model_selection',
    'StratifiedKFold'         : 'sklearn.model_selection',
    'RepeatedStratifiedKFold' : 'sklearn.model_selection',
    'GroupKFold'              : 'sklearn.model_selection',
    'StratifiedGroupKFold'    : 'sklearn.model_selection',
    'LeaveOneOut'             : 'sklearn.model_selection',
    'LeavePOut'               : 'sklearn.model_selection',
    'LeaveOneGroupOut'        : 'sklearn.model_selection',
    'LeavePGroupsOut'         : 'sklearn.model_selection',
    # Shuffle Split - train/test sets might overlap between splits
    'ShuffleSplit'            : 'sklearn.model_selection',
    'StratifiedShuffleSplit'  : 'sklearn.model_selection',
    'GroupShuffleSplit'       : 'sklearn.model_selection',
    # Other
    'PredefinedSplit'         : 'sklearn.model_selection',
    'TimeSeriesSplit'         : 'sklearn.model_selection',
}
