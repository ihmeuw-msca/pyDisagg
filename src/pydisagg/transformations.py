from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import root_scalar


class ParameterTransformation(ABC):
    """
    Transformation that contains function, first order and inverse information
    """

    @abstractmethod
    def __call__(self, x):
        '''
        Calls transformation function
        '''
        pass

    @abstractmethod
    def inverse(self, x):
        '''
        Calls inverse of transformation function
        '''
        pass

    @abstractmethod
    def diff(self, x):
        '''
        Calls derivative of transformation
        '''
        pass


class LogTransformation(ParameterTransformation):
    '''
    Log Transformation here results in a model multiplicative in the baseline rate
    as we fit the model additively in beta
    '''

    def __call__(self, x):
        '''
        Calls transformation function
        '''
        return np.log(x)

    def inverse(self, z):
        return np.exp(z)

    def diff(self, x):
        return 1/x


class LogModifiedOddsTransformation(ParameterTransformation):
    '''
    Log Modified odds transformation
    T(x)=log(x/(1-x**a))
    '''

    def __init__(self, a):
        self.a = a

    def __call__(self, x):
        '''
        Calls transformation function
        '''
        return np.log(x/(1-(x**self.a)))

    def inverse_single(self, z):
        def root_func(x):
            return np.exp(z)*(1-x**self.a)-x
        try:
            return root_scalar(root_func, bracket=[0, 1], method='toms748').root
        except:
            print(z)

    def inverse(self, z):
        return np.vectorize(self.inverse_single)(z)

    def diff(self, x):
        numerator = (self.a-1)*(x**self.a)+1
        denominator = ((x**self.a)-1)*x
        return -1*numerator/denominator


class LogOddsTransformation(ParameterTransformation):
    '''
    Log-odds transformation
    T(x)=log(x/(1-x))
    '''

    def __call__(self, x):
        '''
        Calls transformation function
        '''
        return np.log(x/(1-x))

    def inverse(self, z):
        expz = np.exp(z)
        return expz/(1+expz)

    def diff(self, x):
        return 1/(x-x**2)
