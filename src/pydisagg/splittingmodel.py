"""Module with main SplittingModel class and algorithmic implementation"""
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root_scalar
from scipy.stats import norm

from pydisagg.transformations import ParameterTransformation


class SplittingModel:
    '''
    A possibly confusing code and notation choice is that we enforce 
    multiplicativity in some space, but fit an additive parameter beta throughout

    We do this because it works better with the delta method calculations, and we assume that we
    have a log the transormations so additive factors become multiplicative
    '''

    def __init__(
        self,
        parameter_transformation: ParameterTransformation,
        rate_pattern: Optional[NDArray] = None,
        beta_parameter: Optional[float] = None,
        error_inflation: Optional[float] = None,
        beta_standard_error: Optional[float] = None
    ):
        self.rate_pattern = rate_pattern
        self.beta_parameter = beta_parameter
        self.beta_standard_error = beta_standard_error
        self.error_inflation = error_inflation

        self.parameter_transformation = parameter_transformation
        self.T = self.parameter_transformation
        self.T_inverse = self.parameter_transformation.inverse
        self.T_diff = self.parameter_transformation.diff

    def pull_beta(
        self,
        beta
    ):
        '''
        Checks whether beta parameter is available in input, or if it is null
        and returns beta if it is not none. If beta is none, then this will try and return 
        self.beta_parameter. If neither are available, this will raise an exception
        '''
        if beta is not None:
            return beta
        elif self.beta_parameter is not None:
            return self.beta_parameter
        else:
            raise Exception("Not fitted, No Beta Parameter Available")

    def pull_set_rate_pattern(
        self,
        rate_pattern: NDArray
    ):
        '''
        Checks whether rate_pattern parameter is available in input, or if it is None
            if rate_pattern is not none, it will return it and set it as self.rate_pattern
        If rate_pattern is none, then this will try and return 
        self.rate_pattern. 
        If neither are available, this will raise an exception
        '''
        if rate_pattern is not None:
            self.rate_pattern = rate_pattern
            return rate_pattern
        elif self.rate_pattern is not None:
            return self.rate_pattern
        else:
            raise Exception("No Rate Pattern Available")

    def predict_rates(
        self,
        beta: Optional[float] = None,
        rate_pattern: Optional[NDArray] = None
    ):
        '''
        Generates a predicted rate within each bucket assuming 
            multiplicativity in the T-transformed space with the additive parameter
        '''
        beta_val = self.pull_beta(beta)
        rate_pattern_val = self.pull_set_rate_pattern(rate_pattern)

        return self.T_inverse(beta_val + self.T(rate_pattern_val))

    def _predict_rates_SE(self, beta_val, SE_val):
        '''
        Computes the standard error of the predicted rate in each bucket
            using the delta method, propogating the given standard error on beta
        '''
        return self._rate_derivative(beta_val, self.rate_pattern)*SE_val

    def _H_func(self, beta, bucket_populations):
        '''
        Function outputs expected deaths in a population given a value for beta and 
            the age density of the population in the form of an array (grid of densities)
        '''
        return np.sum(bucket_populations*self.predict_rates(beta))

    def _rate_derivative(self, beta, rate_pattern_value):
        '''
        Computes the derivative with respect to beta
        of the predicted rate in a bucket with a fixed rate_pattern value
        '''
        return 1/self.T_diff(self.T_inverse(beta + self.T(rate_pattern_value)))

    def _H_diff(self, beta, bucket_populations):
        '''
        Function outputs the derivative of the above _H_func with respect to beta
        '''
        return np.sum(bucket_populations *
                      self._rate_derivative(beta, self.rate_pattern)
                      )

    def fit_beta(
        self,
        bucket_populations: NDArray,
        measured_count: float,
        measured_count_se: Optional[float] = None,
        rate_pattern: Optional[NDArray] = None,
        lower_guess: Optional[float] = -50,
        upper_guess: Optional[float] = 50,
        verbose: Optional[int] = 0
    ):
        '''
        Fits a value for beta from the age density of a population and a measured count
        Will attempt to generate a standard error using delta method if a standard error for
        for age density is given. 
        '''
        _ = self.pull_set_rate_pattern(rate_pattern)

        def beta_misfit(beta):
            return self._H_func(beta, bucket_populations)-measured_count

        beta_results = root_scalar(beta_misfit, bracket=[
                                   lower_guess, upper_guess], method='toms748')

        if verbose == 2:
            print(beta_results)
        elif verbose == 1:
            print(f'beta={beta_results.root}')

        self.beta_parameter = beta_results.root
        self.error_inflation = 1 / \
            self._H_diff(self.beta_parameter, bucket_populations)
        if measured_count_se is not None:
            self.beta_standard_error = measured_count_se*self.error_inflation
            if verbose >= 1:
                print(
                    f"Delta Method Standard Error for Beta: {self.beta_standard_error}")
        else:
            # Reset beta_standard_error to none if we refit
            # Old SE is no longer relevant
            self.beta_standard_error = None

    def predict_rates_SE(self):
        '''
        Computes the standard error of the predicted rate in each bucket
            using the delta method, propogating the given standard error on beta
        '''
        if (self.beta_parameter is None):
            raise Exception("Not fitted, No Beta Parameter Available")
        if (self.beta_standard_error is None):
            raise Exception("No Beta Standard Error is available")
        return self._predict_rates_SE(self.beta_parameter, self.beta_standard_error)

    def predict_rates_CI(
            self,
            alpha: Optional[float] = 0.05,
            method: Optional[str] = 'delta-wald'):
        '''
        Computes a 1-alpha confidence interval on the rate function 
            from the standard error on beta

        Method "delta-wald" propogates the standard error on beta through to the
            predicted rate using delta method of Tinv(beta + T(rate_pattern))
        This gives a symmetric confidence interval that will be self consistent with 
            any confidence interval on your original measurement

        Method "pushforward" first computes a confidence interval on beta and then 
            pushes forward this confidence interval directly into the predicted rate
        This gives a possibly assymetric confidence interval that may be inconsistent 
            with your measurement's original confidence interval, but which is likely
            more realistic
        '''
        if (self.beta_parameter is None):
            raise Exception("Not fitted, No Beta Parameter Available")

        if (self.beta_standard_error is None):
            raise Exception("No Beta Standard Error is available")

        l = norm.ppf(alpha/2)
        u = norm.ppf(1-alpha/2)

        if method == 'delta-wald':
            prediction = self.predict_rates()
            rate_SE = self.predict_rates_SE()
            lower_rate = prediction+l*rate_SE
            upper_rate = prediction+u*rate_SE

        if method == 'pushforward':
            beta_lower = self.beta_parameter+l*self.beta_standard_error
            beta_upper = self.beta_parameter+u*self.beta_standard_error
            lower_rate = self.predict_rates(beta_lower)
            upper_rate = self.predict_rates(beta_upper)

        return (lower_rate, upper_rate)

    def predict_count(
        self,
        bucket_populations: NDArray
    ):
        return self.predict_rates()*bucket_populations

    def predict_total_count_SE(
        self,
        bucket_populations: NDArray
    ):
        '''
        Computes the standard error of the total number of events given an age density
        using delta method on H

        This is basically for testing as this should be self consistent with the original 
        standard error if everything working correctly, since everything 
        is just getting expanded out to first order
        '''
        if (self.beta_parameter is None):
            raise Exception("Not fitted, No Beta Parameter Available")

        if (self.beta_standard_error is None):
            raise Exception("No Beta Standard Error is available")

        return self._H_diff(self.beta_parameter, bucket_populations)*self.beta_standard_error

    def predict_count_SE(
        self,
        bucket_populations: NDArray
    ):
        '''
        Computes the standard error of the number events in each bucket given an age density
        using delta method on H
        '''
        if (self.beta_parameter is None):
            raise Exception("Not fitted, No Beta Parameter Available")

        if (self.beta_standard_error is None):
            raise Exception("No Beta Standard Error is available")

        return self.predict_rates_SE()*bucket_populations

    def predict_count_CI(
            self,
            bucket_populations: NDArray,
            alpha: Optional[float] = 0.05,
            method: Optional[str] = 'delta-wald'):
        '''
        Computes a (one minus alpha) confidence interval on the events occuring in a population
            given an an age density from the standard error on beta

        Method "delta-wald" propogates the standard error on beta through to the
            predicted rate using delta method of Tinv(beta + T(rate_pattern))
        This gives a symmetric confidence interval that will be self consistent with 
            any confidence interval on your original measurement

        Method "pushforward" first computes a confidence interval on beta and then 
            pushes forward this confidence interval directly into the predicted rate
        This gives a possibly assymetric confidence interval that may be inconsistent 
            with your measurement's original confidence interval
        '''
        if (self.beta_parameter is None):
            raise Exception("Not fitted, No Beta Parameter Available")

        if (self.beta_standard_error is None):
            raise Exception("No Beta Standard Error is available")

        lower_rate, upper_rate = self.predict_rates_CI(
            alpha=alpha, method=method)
        return (
            bucket_populations*lower_rate,
            bucket_populations*upper_rate
        )

    def split_groups(
        self,
        bucket_populations: NDArray,
        measured_count: Optional[float] = None,
        measured_count_se: Optional[float] = None,
        rate_pattern: Optional[NDArray] = None,
        CI_method: Optional[str] = 'delta-wald',
        alpha: Optional[float] = 0.05
    ):
        '''
        Splits measured_count into the given bucket populations
        If a measured_count and measured_count_se argument is given, 
        then we refit the model to the bucket_populations and the measured_count first before predicting
        '''
        _ = self.pull_set_rate_pattern(rate_pattern)

        if measured_count is not None:
            self.fit_beta(bucket_populations, measured_count,
                          measured_count_se, verbose=0)

        elif (self.beta_parameter is None):
            raise Exception("Not fitted, No Beta Parameter Available")

        if self.beta_standard_error is not None:
            return (
                self.predict_count(bucket_populations),
                self.predict_count_SE(bucket_populations),
                self.predict_count_CI(bucket_populations, alpha=alpha, method=CI_method))

        else:
            return self.predict_count(bucket_populations)

    # def summarize(self,CI_method='delta-wald',title=''):
    #     if (self.beta_parameter is None):
    #         raise Exception("Not fitted, No Beta Parameter Available")

    #     if (self.beta_standard_error is None):
    #         raise Exception("No Beta Standard Error is available")

    #     CI_lower,CI_upper=self.predict_rates_CI(alpha=0.05,method=CI_method)
    #     rates=self.predict_rates()
    #     plt.figure(figsize=(10,7))
    #     plt.fill_between(self.age_grid,CI_lower,CI_upper,color='red',alpha=0.5,label='95% CI')
    #     plt.plot(self.age_grid,rates,c='blue',label='Predicted rate')
    #     beta_print=np.around(self.beta_parameter,2)
    #     SE_print=np.around(self.beta_standard_error,2)
    #     plt.title(title+f"beta={beta_print}, SE={SE_print}")
    #     plt.legend()
