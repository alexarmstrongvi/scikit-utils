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
    elif '.' not in name:
        raise ValueError(
            f'Unrecognized import name {name!r}.'
            ' Check for typo or try full module path.'
        )
    module_name, class_name = name.rsplit('.', maxsplit=1)
    if module_name not in sys.modules:
        log.debug('Importing new module: `import %s`', module_name)
    log.debug('Getting module object: `return %s.%s`', module_name, class_name)
    return getattr(importlib.import_module(module_name), class_name)

# Maps from common objects to their modules so configuration files can specifiy
# estimator without full import path. Also, this serves as a nice reference for
# the available sklearn objects.
# NOTE: Entries organized first into helpful groupings (e.g. _ESTIMATORS).
# Within those groups, they are roughly in the order they are introduced in the
# scikit-learn user guide
_ESTIMATORS = {
    'classifiers' : {
        'base' : {
            # Section 1.1
            'RidgeClassifier'               : 'sklearn.linear_model',
            'LogisticRegression'            : 'sklearn.linear_model',
            'SGDClassifier'                 : 'sklearn.linear_model',
            'Perceptron'                    : 'sklearn.linear_model',
            'PassiveAggressiveClassifier'   : 'sklearn.linear_model',
            # Section 1.2
            'LinearDiscriminantAnalysis'    : 'sklearn.discriminant_analysis',
            'QuadraticDiscriminantAnalysis' : 'sklearn.discriminant_analysis',
            # Section 1.4
            'SVC'                           : 'sklearn.svm',
            'NuSVC'                         : 'sklearn.svm',
            'LinearSVC'                     : 'sklearn.svm',
            # Section 1.6
            'KNeighborsClassifier'          : 'sklearn.neighbors',
            'RadiusNeighborsClassifier'     : 'sklearn.neighbors',
            'NearestCentroid'               : 'sklearn.neighbors',
            # Section 1.7
            'GaussianProcessClassifier'     : 'sklearn.gaussian_process',
            # Section 1.9
            'GaussianNB'                    : 'sklearn.naive_bayes',
            'MultinomialNB'                 : 'sklearn.naive_bayes',
            'ComplementNB'                  : 'sklearn.naive_bayes',
            'BernoulliNB'                   : 'sklearn.naive_bayes',
            'CategoricalNB'                 : 'sklearn.naive_bayes',
            # Section 1.10
            'DecisionTreeClassifier'        : 'sklearn.tree',
            'ExtraTreeClassifier'           : 'sklearn.tree',
            # Section 1.11
            'GradientBoostingClassifier'    : 'sklearn.ensemble',
            'HistGradientBoostingClassifier': 'sklearn.ensemble',
            'ExtraTreesClassifier'          : 'sklearn.ensemble',
            'RandomForestClassifier'        : 'sklearn.ensemble',
            # Section 1.14
            'LabelPropagation'              : 'sklearn.semi_supervised',
            'LabelSpreading'                : 'sklearn.semi_supervised',
            # Section 1.17
            'MLPClassifier'                 : 'sklearn.neural_network',
            # Section 3.4
            'DummyClassifier'               : 'sklearn.dummy',
        },
        'meta' : { # Contruct new classifier from base classifier(s)
            # Section 1.11
            'StackingClassifier'        : 'sklearn.ensemble',
            'BaggingClassifier'         : 'sklearn.ensemble',
            'VotingClassifier'          : 'sklearn.ensemble',
            'AdaBoostClassifier'        : 'sklearn.ensemble',
            # Section 1.12
            'OneVsOneClassifier'        : 'sklearn.multiclass',
            'OneVsRestClassifier'       : 'sklearn.multiclass',
            'OutputCodeClassifier'      : 'sklearn.multiclass',
            'ClassifierChain'           : 'sklearn.multioutput',
            'MultiOutputClassifier'     : 'sklearn.multioutput',
            # Section 1.14
            'SelfTrainingClassifier'    : 'sklearn.semi_supervised',
            # Section 1.16
            'CalibratedClassifierCV'    : 'sklearn.calibration',
        },
        'cv' : { # Builtin support for CV model selection as opposed to using GridSearchCV
            'LogisticRegressionCV'  : 'sklearn.linear_model',
            'RidgeClassifierCV'     : 'sklearn.linear_model',
        },
    },
    'regressors' : {
        'base' : {
            # Section 1.1
            'LinearRegression'             : 'sklearn.linear_model', # 1.1.1
            'Ridge'                        : 'sklearn.linear_model', # 1.1.2
            'Lasso'                        : 'sklearn.linear_model', # 1.1.3
            'MultiTaskLasso'               : 'sklearn.linear_model', # 1.1.4
            'ElasticNet'                   : 'sklearn.linear_model', # 1.1.5
            'MultiTaskElasticNet'          : 'sklearn.linear_model', # 1.1.6
            'Lars'                         : 'sklearn.linear_model', # 1.1.7
            'LassoLars'                    : 'sklearn.linear_model', # 1.1.8
            'LassoLarsIC'                  : 'sklearn.linear_model', # ""
            'OrthogonalMatchingPursuit'    : 'sklearn.linear_model', # 1.1.9
            'BayesianRidge'                : 'sklearn.linear_model', # 1.1.10
            'ARDRegression'                : 'sklearn.linear_model', # ""
            'PoissonRegressor'             : 'sklearn.linear_model', # 1.1.12
            'GammaRegressor'               : 'sklearn.linear_model', # ""
            'TweedieRegressor'             : 'sklearn.linear_model', # ""
            'SGDRegressor'                 : 'sklearn.linear_model', # 1.1.13
            'PassiveAggressiveRegressor'   : 'sklearn.linear_model', # 1.1.15
            'TheilSenRegressor'            : 'sklearn.linear_model', # 1.1.16
            'HuberRegressor'               : 'sklearn.linear_model', # ""
            'QuantileRegressor'            : 'sklearn.linear_model', # ""
            # Section 1.3
            'KernelRidge'                  : 'sklearn.kernel_ridge',
            # Section 1.4
            'SVR'                          : 'sklearn.svm',
            'NuSVR'                        : 'sklearn.svm',
            'LinearSVR'                    : 'sklearn.svm',
            # Section 1.6
            'KNeighborsRegressor'          : 'sklearn.neighbors',
            'RadiusNeighborsRegressor'     : 'sklearn.neighbors',
            # Section 1.7
            'GaussianProcessRegressor'     : 'sklearn.gaussian_process',
            # Section 1.8
            'PLSCanonical'                 : 'sklearn.cross_decomposition',
            'PLSSVD'                       : 'sklearn.cross_decomposition',
            'PLSRegression'                : 'sklearn.cross_decomposition',
            'CCA'                          : 'sklearn.cross_decomposition',
            # Section 1.10
            'DecisionTreeRegressor'        : 'sklearn.tree',
            'ExtraTreeRegressor'           : 'sklearn.tree',
            # Section 1.11
            'GradientBoostingRegressor'    : 'sklearn.ensemble',
            'HistGradientBoostingRegressor': 'sklearn.ensemble',
            'ExtraTreesRegressor'          : 'sklearn.ensemble',
            'RandomForestRegressor'        : 'sklearn.ensemble',
            # Section 1.15
            'IsotonicRegression'           : 'sklearn.isotonic',
            # Section 1.17
            'MLPRegressor'                 : 'sklearn.neural_network',
            # Section 3.4
            'DummyRegressor'               : 'sklearn.dummy',
        },
        'meta' : {
            'AdaBoostRegressor'          : 'sklearn.ensemble',
            'BaggingRegressor'           : 'sklearn.ensemble',
            'StackingRegressor'          : 'sklearn.ensemble',
            'VotingRegressor'            : 'sklearn.ensemble',
            'RANSACRegressor'            : 'sklearn.linear_model',
            'MultiOutputRegressor'       : 'sklearn.multioutput',
            'RegressorChain'             : 'sklearn.multioutput',
            'TransformedTargetRegressor' : 'sklearn.compose',
        },
        'cv' : {
            'RidgeCV'                    : 'sklearn.linear_model',
            'LassoCV'                    : 'sklearn.linear_model',
            'MultiTaskLassoCV'           : 'sklearn.linear_model',
            'ElasticNetCV'               : 'sklearn.linear_model',
            'MultiTaskElasticNetCV'      : 'sklearn.linear_model',
            'LarsCV'                     : 'sklearn.linear_model',
            'LassoLarsCV'                : 'sklearn.linear_model',
            'OrthogonalMatchingPursuitCV': 'sklearn.linear_model',
        },
    },
    'feature_selectors' : {
        'univariate' : {
            # Section 1.13.1
            'VarianceThreshold'        : 'sklearn.feature_selection',
            # Section 1.13.2
            'SelectKBest'              : 'sklearn.feature_selection',
            'SelectPercentile'         : 'sklearn.feature_selection',
            'SelectFdr'                : 'sklearn.feature_selection',
            'SelectFpr'                : 'sklearn.feature_selection',
            'SelectFwe'                : 'sklearn.feature_selection',
            'GenericUnivariateSelect'  : 'sklearn.feature_selection',
        },
        'meta' : {
            # Section 1.13.3
            'RFE'                      : 'sklearn.feature_selection',
            'RFECV'                    : 'sklearn.feature_selection',
            # Section 1.13.4
            'SelectFromModel'          : 'sklearn.feature_selection',
            # Section 1.13.5
            'SequentialFeatureSelector': 'sklearn.feature_selection',
        },
    },
    'unsupervised' : {
        # Section 1.6
        'NearestNeighbors' : 'sklearn.neighbors',
        # Section 2.1 - Gaussian mixture models
        'BayesianGaussianMixture': 'sklearn.mixture',
        'GaussianMixture'        : 'sklearn.mixture',
        # Section 2.2 - Manifold learning
        'Isomap'                : 'sklearn.manifold',
        'LocallyLinearEmbedding': 'sklearn.manifold',
        'SpectralEmbedding'     : 'sklearn.manifold',
        'MDS'                   : 'sklearn.manifold',
        'TSNE'                  : 'sklearn.manifold',
        # Section 2.3 - Clustering
        'clusterers' : {
            'KMeans'                 : 'sklearn.cluster',
            'MiniBatchKMeans'        : 'sklearn.cluster',
            'AffinityPropagation'    : 'sklearn.cluster',
            'MeanShift'              : 'sklearn.cluster',
            'SpectralClustering'     : 'sklearn.cluster',
            'AgglomerativeClustering': 'sklearn.cluster',
            'FeatureAgglomeration'   : 'sklearn.cluster',
            'BisectingKMeans'        : 'sklearn.cluster',
            'DBSCAN'                 : 'sklearn.cluster',
            'HDBSCAN'                : 'sklearn.cluster',
            'OPTICS'                 : 'sklearn.cluster',
            'Birch'                  : 'sklearn.cluster',
        },
        # Section 2.4 - Biclustering
        'SpectralCoclustering'   : 'sklearn.cluster',
        'SpectralBiclustering'   : 'sklearn.cluster',
        # Section 2.5 - Decomposing
        'PCA'                        : 'sklearn.decomposition',
        'IncrementalPCA'             : 'sklearn.decomposition',
        'SparsePCA'                  : 'sklearn.decomposition',
        'MiniBatchSparsePCA'         : 'sklearn.decomposition',
        'KernelPCA'                  : 'sklearn.decomposition',
        'TruncatedSVD'               : 'sklearn.decomposition',
        'SparseCoder'                : 'sklearn.decomposition',
        'DictionaryLearning'         : 'sklearn.decomposition',
        'MiniBatchDictionaryLearning': 'sklearn.decomposition',
        'FactorAnalysis'             : 'sklearn.decomposition',
        'FastICA'                    : 'sklearn.decomposition',
        'NMF'                        : 'sklearn.decomposition',
        'MiniBatchNMF'               : 'sklearn.decomposition',
        'LatentDirichletAllocation'  : 'sklearn.decomposition',
        # Section 2.6 - Covariance
        'EmpiricalCovariance': 'sklearn.covariance',
        'GraphicalLasso'     : 'sklearn.covariance',
        'GraphicalLassoCV'   : 'sklearn.covariance',
        'LedoitWolf'         : 'sklearn.covariance',
        'MinCovDet'          : 'sklearn.covariance',
        'OAS'                : 'sklearn.covariance',
        'ShrunkCovariance'   : 'sklearn.covariance',
    },
    'outlier_detection' : {
        # Section 2.7 - Outlier detection
        'OneClassSVM'       : 'sklearn.svm',
        'SGDOneClassSVM'    : 'sklearn.linear_model',
        'EllipticEnvelope'  : 'sklearn.covariance',
        'IsolationForest'   : 'sklearn.ensemble',
        'LocalOutlierFactor': 'sklearn.neighbors',
    },
    # Section 2.8 - Density estimate
    'KernelDensity'                 : 'sklearn.neighbors',
    # Section 2.9 - NN Models (unsupervised)
    'BernoulliRBM'                  : 'sklearn.neural_network',
    # Section 6.1 - Pipeline
    'Pipeline'                      : 'sklearn.pipeline',
    'FeatureUnion'                  : 'sklearn.pipeline',
    'ColumnTransformer'             : 'sklearn.compose',
    # Section 6.2 - Feature extraction
    'DictVectorizer'                : 'sklearn.feature_extraction',
    'FeatureHasher'                 : 'sklearn.feature_extraction',
    'CountVectorizer'               : 'sklearn.feature_extraction.text',
    'HashingVectorizer'             : 'sklearn.feature_extraction.text',
    'TfidfTransformer'              : 'sklearn.feature_extraction.text',
    'TfidfVectorizer'               : 'sklearn.feature_extraction.text',
    'PatchExtractor'                : 'sklearn.feature_extraction.image',
    # Section 6.3 - Preprocessing
    # Section 6.3.1 - Standardization
    'StandardScaler'                : 'sklearn.preprocessing',
    'MaxAbsScaler'                  : 'sklearn.preprocessing',
    'MinMaxScaler'                  : 'sklearn.preprocessing',
    'RobustScaler'                  : 'sklearn.preprocessing',
    'KernelCenterer'                : 'sklearn.preprocessing',
    # Section 6.3.2 - Non-linear transformer
    'QuantileTransformer'           : 'sklearn.preprocessing',
    'PowerTransformer'              : 'sklearn.preprocessing',
    # Section 6.3.3
    'Normalizer'                    : 'sklearn.preprocessing',
    # Section 6.3.4 - Encoding categorical features
    'OrdinalEncoder'                : 'sklearn.preprocessing',
    'OneHotEncoder'                 : 'sklearn.preprocessing',
    'TargetEncoder'                 : 'sklearn.preprocessing',
    # Section 6.3.5 - Discretization
    'Binarizer'                     : 'sklearn.preprocessing',
    'FunctionTransformer'           : 'sklearn.preprocessing',
    'KBinsDiscretizer'              : 'sklearn.preprocessing',
    'LabelEncoder'                  : 'sklearn.preprocessing',
    'PolynomialFeatures'            : 'sklearn.preprocessing',
    'SplineTransformer'             : 'sklearn.preprocessing',
    # Section 6.4 - Imputation of missing values
    'SimpleImputer'                 : 'sklearn.impute',
    'KNNImputer'                    : 'sklearn.impute',
    'MissingIndicator'              : 'sklearn.impute',
    # Section 6.6
    'GaussianRandomProjection'      : 'sklearn.random_projection',
    'SparseRandomProjection'        : 'sklearn.random_projection',
    # Section 6.7
    'Nystroem'                      : 'sklearn.kernel_approximation',
    'RBFSampler'                    : 'sklearn.kernel_approximation',
    'AdditiveChi2Sampler'           : 'sklearn.kernel_approximation',
    'SkewedChi2Sampler'             : 'sklearn.kernel_approximation',
    'PolynomialCountSketch'         : 'sklearn.kernel_approximation',
    # Section 6.9 - Transforming prediction target
    'LabelBinarizer'                : 'sklearn.preprocessing',
    'MultiLabelBinarizer'           : 'sklearn.preprocessing',
    # section 1.6.6
    'KNeighborsTransformer'         : 'sklearn.neighbors',
    'RadiusNeighborsTransformer'    : 'sklearn.neighbors',
    # section 1.6.7
    'NeighborhoodComponentsAnalysis': 'sklearn.neighbors',
    # section 1.11.2.6
    'RandomTreesEmbedding'          : 'sklearn.ensemble',
}
_SPLITTERS = {
    # Section 3.1
    # K-Fold CV - test sets partition data
    'KFold'                   : 'sklearn.model_selection',
    'RepeatedKFold'           : 'sklearn.model_selection',
    'LeaveOneOut'             : 'sklearn.model_selection',
    'LeavePOut'               : 'sklearn.model_selection',
    'StratifiedKFold'         : 'sklearn.model_selection',
    'RepeatedStratifiedKFold' : 'sklearn.model_selection',
    'GroupKFold'              : 'sklearn.model_selection',
    'StratifiedGroupKFold'    : 'sklearn.model_selection',
    'LeaveOneGroupOut'        : 'sklearn.model_selection',
    'LeavePGroupsOut'         : 'sklearn.model_selection',
    # Shuffle Split - train/test sets might overlap between splits
    'ShuffleSplit'            : 'sklearn.model_selection',
    'StratifiedShuffleSplit'  : 'sklearn.model_selection',
    'GroupShuffleSplit'       : 'sklearn.model_selection',
    # Other splits
    'PredefinedSplit'         : 'sklearn.model_selection',
    'TimeSeriesSplit'         : 'sklearn.model_selection',
    # Section 3.2
    'GridSearchCV'       : 'sklearn.model_selection',
    'RandomizedSearchCV' : 'sklearn.model_selection'
    # 'HalvingGridSearchCV'       : 'sklearn.model_selection' # Experimental
    # 'HalvingRandomizedSearchCV' : 'sklearn.model_selection' # Experimental
}

def flatten_dict(d):
    flat_d = {}
    for k, v in d.items():
        if isinstance(v, dict):
            flat_d.update(flatten_dict(v))
        else:
            flat_d[k] = v
    return flat_d

_MODULE_MAP = {
    **flatten_dict(_ESTIMATORS),
    **flatten_dict(_SPLITTERS),
}
