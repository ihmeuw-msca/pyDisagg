"""
Module containing abstract ParameterTransformation class and some ParameterTransformation subclasses
"""
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root_scalar

float_or_array = Union[float, NDArray]


class ParameterTransformation(ABC):
    """
    Transformation that contains function, first order and inverse information
    """

    @abstractmethod
    def __call__(self, x: float_or_array):
        '''
        Calls transformation function
        '''

    @abstractmethod
    def inverse(self, z: float_or_array):
        '''
        Calls inverse of transformation function
        '''

    @abstractmethod
    def diff(self, x: float_or_array):
        '''
        Calls derivative of transformation
        '''


class LogTransformation(ParameterTransformation):
    '''
    Logarithmic transformation of parameter
    Since we fit models additively in beta the log makes it multiplicative
    '''

    def __call__(self, x: float_or_array):
        return np.log(x)

    def inverse(self, z: float_or_array):
        return np.exp(z)

    def diff(self, x: float_or_array):
        return 1/x


class LogModifiedOddsTransformation(ParameterTransformation):
    '''
    Log Modified odds transformation
    T(x)=log(x/(1-x**a))
    '''

    def __init__(self, a:float):
        self.a = a

    def __call__(self, x: float_or_array):
        '''
        Calls transformation function
        '''
        return np.log(x/(1-(x**self.a)))

    def _inverse_single(self, z:float):
        def root_func(x):
            return np.exp(z)*(1-x**self.a)-x
        return root_scalar(root_func, bracket=[0, 1], method='toms748').root

    def inverse(self, z: float_or_array):
        return np.vectorize(self._inverse_single)(z)

    def diff(self, x: float_or_array):
        numerator = (self.a-1)*(x**self.a)+1
        denominator = ((x**self.a)-1)*x
        return -1*numerator/denominator


class LogOddsTransformation(ParameterTransformation):
    '''
    Log-odds transformation
    T(x)=log(x/(1-x))
    '''

    def __call__(self, x: float_or_array):
        '''
        Calls transformation function
        '''
        return np.log(x/(1-x))

    def inverse(self, z: float_or_array):
        expz = np.exp(z)
        return expz/(1+expz)

    def diff(self, x: float_or_array):
        return 1/(x-x**2)
