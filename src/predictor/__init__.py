# __all__ = ['predictor']

from numpy import ndarray
from typing import Tuple, Union

TrainTestVectors = Tuple[ndarray, ndarray, ndarray, ndarray]
TrainVectors = Union[TrainTestVectors, Tuple[ndarray, ndarray]]
