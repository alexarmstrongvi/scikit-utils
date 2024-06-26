# Standard library
from collections.abc import Iterable
from typing import Union

# 3rd party
import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import BaseCrossValidator, BaseShuffleSplit

# Type aliases
NDArrayInt = NDArray[np.int64]

################################################################################
TrainTestIterable = Union[
    BaseCrossValidator,
    Iterable[tuple[NDArrayInt, NDArrayInt]],
    BaseShuffleSplit,
]