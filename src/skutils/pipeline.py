# Standard library
import logging
from pathlib import Path

# 3rd party
from sklearn import compose, pipeline

# 1st party
from skutils.import_utils import import_object

# Globals
log = logging.getLogger(Path(__file__).stem)

################################################################################
def make_pipeline(steps: list[str | dict],  **kwargs) -> pipeline.Pipeline:
    use_make_func, step_list = _make_pipeline_steps(steps)
    if use_make_func:
        pipe = pipeline.make_pipeline(*step_list, **kwargs)
    else:
        pipe = pipeline.Pipeline(step_list, **kwargs)
    return pipe


def make_union(transformers: list[str | dict], **kwargs) -> pipeline.FeatureUnion:
    use_make_func, transformer_list = _make_pipeline_steps(transformers)
    if use_make_func:
        union = pipeline.make_union(*transformer_list, **kwargs)
    else:
        union = pipeline.FeatureUnion(transformer_list, **kwargs)
    return union


def make_column_transformer(transformers: list[list[str | list | dict]], **kwargs) -> compose.ColumnTransformer:
    transformer_list = []
    use_make_func = None
    for cfg in transformers:
        name, class_, kw, columns = _parse_column_transformer_config(cfg)

        # Check
        if use_make_func is None:
            use_make_func = name is None
        elif use_make_func != (name is None):
            raise ValueError(f'Mixing of transformer configurations: {transformers}')

        # Build objects from configuration strings
        if isinstance(columns, dict):
            columns = compose.make_column_selector(**columns)
        transformer = _make_transformer(class_, **kw)

        # Record transformer
        transformer_list.append(
            (transformer, columns) if use_make_func else (name, transformer, columns)
        )

    # Create column transformer
    if use_make_func:
        transformer = compose.make_column_transformer(*transformer_list, **kwargs)
    else:
        transformer = compose.ColumnTransformer(transformer_list, **kwargs)
    return transformer

################################################################################
# Support functions
def _make_pipeline_steps(steps: list[str | list[str | dict]]) -> tuple[bool | None, list]:
    step_tuples = []
    use_make_func = None
    for cfg in steps:
        name, class_, kw = _parse_pipeline_config(cfg)

        # Check
        if use_make_func is None:
            use_make_func = name is None
        elif use_make_func != (name is None):
            raise ValueError(f'Mixing of pipeline step configurations: {steps}')

        transformer = _make_transformer(class_, **kw)

        # Record transformer
        step_tuples.append(transformer if use_make_func else (name, transformer))

    return use_make_func, step_tuples


def _parse_pipeline_config(cfg: str | list[str | dict]) -> tuple[str | None, str, dict]:
    match cfg:
        # make_pipeline format
        case [class_] | class_ if isinstance(class_, str):
            name, kw = None, {}
        case [class_, kw] if isinstance(kw, dict):
            name = None
        # Pipeline format
        case [name, class_]:
            kw = {}
        case [name, class_, kw]:
            pass
        case _:
            raise ValueError(f'Unexpected transformer configuration: {cfg!r}')

    if not (
        (name is None or isinstance(name, str))
        and isinstance(class_, str)
        and isinstance(kw, dict)
    ):
        raise TypeError(f'Invalid pipeline step configuration: {cfg!r}')

    return name, class_, kw


def _parse_column_transformer_config(cfg: list[str | list | dict]) -> tuple[str | None, str, dict, str | list[str] | dict]:
    match cfg:
        # make_column_transformer format
        case [class_, columns]:
            name, kw = None, {}
        case [class_, kw, columns] if isinstance(kw, dict):
            name = None
        # ColumnTransformer format
        case [name, class_, columns]:
            kw = {}
        case [name, class_, kw, columns]:
            pass
        case _:
            raise ValueError(f'Unexpected transformer configuration: {cfg}')

    # Check
    if not (
        (name is None or isinstance(name, str))
        and isinstance(class_, str)
        and isinstance(kw, dict)
        and isinstance(columns, (str, list, dict))
    ):
        raise TypeError(f'Invalid transformer configuration: {cfg}')

    return name, class_, kw, columns

def _make_transformer(name: str, **kwargs):
    if name == 'make_pipeline' or name == 'Pipeline':
        transformer = make_pipeline(**kwargs)
    elif name == 'make_union' or name == 'FeatureUnion':
        transformer = make_union(**kwargs)
    elif name == 'make_column_transformer' or name == 'ColumnTransformer':
        transformer = make_column_transformer(**kwargs)
    else:
        # TODO: handle kwargs containing an object that needs to be imported. It
        # could be a function or class with kwargs of its own. Need to decide on
        # the best way for the user to specify this in the configuration.
        # new_kwargs = {}
        # for k, v in kwargs.items():
        #     if v.startswith('sklearn.') or v.startswith('skutils.'):
        #         v = import_object(name)
        #     new_kwargs[k] = v
        # kwargs = new_kwargs
        transformer = import_object(name)(**kwargs)

    return transformer
