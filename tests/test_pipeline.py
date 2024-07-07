# 3rd party
import pandas as pd
import pytest
from sklearn import (
    compose,
    decomposition,
    feature_extraction,
    pipeline,
    preprocessing,
    svm
)

# 1st party
from skutils.pipeline import make_column_transformer, make_pipeline, make_union


################################################################################
def test_make_pipeline():
    # see User Guide 6.1.1.1.1
    cfg = { # make_pipeline config format
        'steps' : [
            'PCA', # No kwargs
            ['SVC', {'C' : 0.9}]] # With kwargs
    }
    pipe = make_pipeline(**cfg)
    assert len(pipe.steps) == 2
    name, obj = pipe.steps[0] # Access via .steps list
    assert name == 'pca' and isinstance(obj, decomposition.PCA)
    assert isinstance(pipe['svc'], svm.SVC) # Access via __getitem__

    cfg = { # Pipeline config format
        'steps' : [
            ['reduce_dim', 'PCA'],
            ['clf', 'SVC', {'C' : 0.9}],
        ],
    }
    pipe = make_pipeline(**cfg)
    assert len(pipe.steps) == 2
    assert isinstance(pipe['reduce_dim'], decomposition.PCA)
    assert isinstance(pipe['clf'], svm.SVC)

    # Test error handling
    # Mixing make_pipeline and Pipeline formats
    cfg = {
        'steps' : [
            'PCA',
            ['clf', 'SVC', {'C' : 0.9}],
        ],
    }
    with pytest.raises(ValueError, match='^Mixing.*configurations'):
        make_pipeline(**cfg)


def test_make_union():
    # see User Guide 6.1.3.1
    cfg = { # make_union config format
        'transformers' : [
            'PCA',
            ['KernelPCA', {'kernel' : 'poly'}],
        ]
    }
    union = make_union(**cfg)
    assert len(union.named_transformers) == 2
    assert isinstance(union['pca'], decomposition.PCA)
    assert isinstance(union['kernelpca'], decomposition.KernelPCA)

    cfg = { # FeatureUnion config format
        'transformers' : [
            ['linear_pca', 'PCA'],
            ['kernel_pca', 'KernelPCA', {'kernel' : 'poly'}],
        ]
    }
    union = make_union(**cfg)
    assert len(union.named_transformers) == 2
    assert isinstance(union['linear_pca'], decomposition.PCA)
    assert isinstance(union['kernel_pca'], decomposition.KernelPCA)

    # Test error handling
    # Mixing make_union and FeatureUnion formats
    cfg = {
        'transformers' : [
            'PCA',
            ['kernel_pca', 'KernelPCA', {'kernel' : 'poly'}],
        ]
    }
    with pytest.raises(ValueError, match='^Mixing.*configurations'):
        make_union(**cfg)



def test_make_column_transformers():
    # see User Guide 6.1.4
    X = pd.DataFrame({
        'city': ['London', 'London', 'Paris', 'Sallisaw'],
        'title': [
            "His Last Bow",
            "How Watson Learned the Trick",
            "A Moveable Feast",
            "The Grapes of Wrath",
        ],
        'expert_rating': [5, 3, 4, 5],
        'user_rating': [4, 5, 4, 3]
    })

    cfg = { # make_column_transformer config format
        'transformers' : [
            ['OneHotEncoder', {'dtype' : 'int', 'sparse_output' : False}, ['city']],
            ['CountVectorizer', 'title'],
        ],
        'remainder' : 'drop',
        'verbose_feature_names_out' : False,
    }
    column_transformer = make_column_transformer(**cfg)
    assert len(column_transformer.transformers) == 2
    name, obj, cols = column_transformer.transformers[0]
    assert name == 'onehotencoder'
    assert isinstance(obj, preprocessing.OneHotEncoder)
    assert cols == ['city']

    column_transformer.fit(X)
    assert column_transformer.get_feature_names_out().tolist() == [
        'city_London', 'city_Paris', 'city_Sallisaw', 'bow', 'feast', 'grapes',
        'his', 'how', 'last', 'learned', 'moveable', 'of', 'the', 'trick',
        'watson', 'wrath'
    ]

    cfg = { # ColumnTransformer config format
        'transformers' : [
            ['categories', 'OneHotEncoder', {'dtype' : 'int', 'sparse_output': False}, ['city']],
            ['title_bow', 'CountVectorizer', 'title'],
        ],
        'remainder' : 'drop',
        'verbose_feature_names_out' : False,
    }
    column_transformer = make_column_transformer(**cfg)
    assert len(column_transformer.transformers) == 2
    name, obj, cols = column_transformer.transformers[1]
    assert name == 'title_bow'
    assert isinstance(obj, feature_extraction.text.CountVectorizer)
    assert cols == 'title'

    # Test error handling
    # Mixing make_union and FeatureUnion formats
    cfg = {
        'transformers' : [
            ['categories', 'OneHotEncoder', {'dtype' : 'int', 'sparse_output': False}, ['city']],
            ['CountVectorizer', 'title'],
        ],
    }
    with pytest.raises(ValueError, match='^Mixing.*configurations'):
        make_column_transformer(**cfg)


def test_nested_pipelines():
    cfg_pipeline = {
        'steps' : [
            'PCA', # No kwargs
            ['SVC', {'C' : 0.9}]] # With kwargs
    }
    cfg_union = {
        'transformers' : [
            'PCA',
            ['KernelPCA', {'kernel' : 'poly'}],
        ]
    }
    cfg_column_transformer = {
        'transformers' : [
            ['OneHotEncoder', {'dtype' : 'int', 'sparse_output' : False}, ['city']],
            ['CountVectorizer', 'title'],
        ],
        'remainder' : 'drop',
        'verbose_feature_names_out' : False,
    }

    # Pipeline of FeatureUnion and/or ColumnTransformer
    cfg = { # make_pipeline config format
        'steps' : [
            ['MyColTransformer', 'make_column_transformer', cfg_column_transformer],
            ['MyFeatureUnion', 'make_union', cfg_union],
        ]
    }
    pipe = make_pipeline(**cfg)
    assert len(pipe.steps) == 2
    assert isinstance(pipe['MyColTransformer'], compose.ColumnTransformer)
    assert isinstance(pipe['MyFeatureUnion'], pipeline.FeatureUnion)

    # FeatureUnion of Pipeline and/or ColumnTransformer
    cfg = {
        'transformers' : [
            ['MyColTransformer', 'make_column_transformer', cfg_column_transformer],
            ['MyPipeline', 'make_pipeline', cfg_pipeline],
        ]
    }
    union = make_union(**cfg)
    assert len(union.named_transformers) == 2
    assert isinstance(union.named_transformers['MyColTransformer'], compose.ColumnTransformer)
    assert isinstance(union.named_transformers['MyPipeline'], pipeline.Pipeline)

    # ColumnTransformer of Pipeline and/or FeatureUnion
    cfg = {
        'transformers' : [
            ['MyPipeline', 'make_pipeline', cfg_pipeline, 'column1'],
            ['MyFeatureUnion', 'make_union', cfg_union, ['column2', 'column3']],
        ]
    }
    column_transformer = make_column_transformer(**cfg)
    assert len(column_transformer.transformers) == 2
    name, obj, cols = column_transformer.transformers[0]
    assert name == 'MyPipeline'
    assert isinstance(obj, pipeline.Pipeline)
    assert cols == 'column1'
    name, obj, cols = column_transformer.transformers[1]
    assert name == 'MyFeatureUnion'
    assert isinstance(obj, pipeline.FeatureUnion)
    assert cols == ['column2', 'column3']
