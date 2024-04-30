"""
Module containing abstract ParameterTransformation class
"""

from abc import ABC, abstractmethod
from typing import Union

from numpy.typing import NDArray

float_or_array = Union[float, NDArray]


class ParameterTransformation(ABC):
    """
    Transformation that contains function, first order and inverse information
    """

    @abstractmethod
    def __call__(self, x: float_or_array):
        """
        Calls transformation function
        """

    @abstractmethod
    def inverse(self, z: float_or_array):
        """
        Calls inverse of transformation function
        """

    @abstractmethod
    def diff(self, x: float_or_array):
        """
        Calls derivative of transformation
        """
