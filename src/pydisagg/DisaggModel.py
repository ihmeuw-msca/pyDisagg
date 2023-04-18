"""Module with main DisaggModel class and algorithmic implementation"""
from typing import Optional,Tuple,Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root_scalar
from scipy.stats import norm

from pydisagg.transformations import ParameterTransformation


class DisaggModel:
    '''
    Model for solving splitting/disaggregation problems

    Notes
    -----
    A possibly confusing code and notation choice is that we enforce
    multiplicativity in some space, but fit an additive parameter beta throughout

    We do this because it works better with the delta method calculations, and we assume that we
    have a log in the transormations so additive factors become multiplicative
    '''

    def __init__(
        self,
        parameter_transformation: ParameterTransformation,
        rate_pattern: Optional[NDArray] = None,
        beta_parameter: Optional[float] = None,
        beta_standard_error: Optional[float] = None,
        rate_pattern_cov:Optional[NDArray]=None,
        full_parameter_cov:Optional[NDArray]=None,
    ):
        """

        Parameters
        ----------
        parameter_transformation : ParameterTransformation
            Transformation to apply to rate pattern values
        rate_pattern : Optional[NDArray], optional
            Rate Pattern to use, should be an estimate of the rates in each bucket
            that we want to rescale, by default None
        beta_parameter : Optional[float], optional
            Additive parameter in log space (acts as multiplier)., by default None
        beta_standard_error : Optional[float], optional
            Standard error of beta parameter, by default None
        """

        self.beta_parameter = beta_parameter
        self.beta_standard_error = beta_standard_error

        self.parameter_transformation = parameter_transformation
        self.T = self.parameter_transformation
        self.T_inverse = self.parameter_transformation.inverse
        self.T_diff = self.parameter_transformation.diff
        self._rate_pattern = rate_pattern
        self._rate_pattern_cov=rate_pattern_cov
        self.full_parameter_cov=full_parameter_cov

    @property
    def rate_pattern_cov(self) -> NDArray:
        '''Model rate pattern point estimate'''
        if self._rate_pattern_cov is None:
            raise ValueError("No Rate Pattern Covariance Available")
        return self._rate_pattern_cov
    
    @rate_pattern_cov.setter
    def rate_pattern_cov(self,rate_pattern_cov:Optional[NDArray]):
        if rate_pattern_cov is not None:
            self._rate_pattern_cov = rate_pattern_cov

    @property
    def rate_pattern(self) -> NDArray:
        '''Model rate pattern point estimate'''
        if self._rate_pattern is None:
            raise ValueError("No Rate Pattern Available")
        return self._rate_pattern

    @rate_pattern.setter
    def rate_pattern(self, rate_pattern: Optional[NDArray]):
        if rate_pattern is not None:
            self._rate_pattern = rate_pattern

    def pull_beta(
        self,
        beta: float
    )->float:
        """Checks whether beta parameter is available in input, or if it is null
        and returns beta if it is not none. If beta is none, then this will try and return
        self.beta_parameter. If neither are available, this will raise an exception

        Parameters
        ----------
        beta : float or None
            fitted model parameter or None, as this is a check to see if its passed

        Returns
        -------
        float
            the value for beta

        Raises
        ------
        Exception
            "Not fitted, No Beta Parameter Available" if no Beta parameter is available
        """
        if beta is not None:
            return beta
        if self.beta_parameter is not None:
            return self.beta_parameter
        raise Exception("Not fitted, No Beta Parameter Available")

    def predict_rates(
        self,
        beta: Optional[float] = None,
        rate_pattern: Optional[NDArray] = None
    )->NDArray:
        """
        Generate a predicted rate within each bucket assuming
            multiplicativity in the T-transformed space with the additive parameter

        Parameters
        ----------
        beta : Optional[float], optional
            fitted beta parameter, by default None
        rate_pattern : Optional[NDArray], optional
            by default None

        Returns
        -------
        NDArray
            Predicted rates in each bucket
        """
        beta_val = self.pull_beta(beta)
        self.rate_pattern = rate_pattern

        return self.T_inverse(beta_val + self.T(self.rate_pattern))

    def _predict_rates_SE(self, beta_val, SE_val):
        '''
        Computes the standard error of the predicted rate in each bucket
            using the delta method, propogating the given standard error on beta
        '''
        return self._rate_derivative_beta(beta_val, self.rate_pattern)*SE_val

    def _H_func(self, beta, bucket_populations):
        '''
        Function outputs expected deaths in a population given a value for beta and
            the age density of the population in the form of an array (grid of densities)
        '''
        return np.sum(bucket_populations*self.predict_rates(beta))

    def _rate_derivative_beta(self, beta, rate_pattern_value):
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
                      self._rate_derivative_beta(beta, self.rate_pattern)
                      )
    
    def _fgrad_H(self,beta,bucket_populations,rate_pattern=None):
        '''
        Computes the gradient of H with respect to the pattern, at the point beta, self.rate_pattern
        '''
        self.rate_pattern = rate_pattern
        return bucket_populations*self.T_diff(self.rate_pattern)/self.T_diff(self.beta_parameter + self.T(self.rate_pattern))
    

    def _fgrad_Hinv(self,beta,bucket_populations,rate_pattern=None):
        """
        Computes the gradient of 
        the beta partial inverse of H with respect to the pattern, at the point 
            (H(beta,self.rate_pattern), self.rate_pattern)

        Parameters
        ----------
        fitted_beta : float
            _description_
        bucket_populations : _type_
            _description_
        """
        self.rate_pattern = rate_pattern
        denominator = self._H_diff(beta,bucket_populations)
        ## Call this at the fitted beta for the inverse derivative of H at the observed_total 
        # If observed_total is H(beta), the total that beta would give

        return self._fgrad_H(beta,bucket_populations,self.rate_pattern)/denominator

    def _build_parameter_covariance(
            self,
            beta,
            bucket_populations,
            rate_pattern,
            rate_pattern_cov,
            observed_total_se,
            ):
        #Assumes that beta is the fitted beta
        self.rate_pattern = rate_pattern
        self.rate_pattern_cov = rate_pattern_cov
        beta_pattern_cov = (
            self.rate_pattern_cov @ 
            self._fgrad_Hinv(beta,bucket_populations,self.rate_pattern)
        ).reshape(-1,1)

        beta_error_inflation = (1 / \
                self._H_diff(self.beta_parameter, bucket_populations)
                )
        
        beta_var = np.array([[(observed_total_se*beta_error_inflation)**2]])
        full_parameter_cov = np.block(
            [
            [beta_var,beta_pattern_cov.T],
            [beta_pattern_cov,rate_pattern_cov]
            ]
        )
        self.full_parameter_cov = full_parameter_cov
        return full_parameter_cov

    def fit_beta(
        self,
        bucket_populations: NDArray,
        observed_total: float,
        observed_total_se: Optional[float] = None,
        rate_pattern: Optional[NDArray] = None,
        lower_guess: Optional[float] = -50,
        upper_guess: Optional[float] = 50,
        verbose: Optional[int] = 0
    )->None:
        """
        Fit a value for beta from the age density of a population and a measured count
        Will attempt to generate a standard error using delta method if a standard error for
        for age density is given.

        Parameters
        ----------
        bucket_populations : NDArray
            numpy array of population sizes for each bucket
        observed_total : float
            total of observed quantity across bucket_populations
        observed_total_se : Optional[float], optional
            se of observed_total, by default None
        rate_pattern : Optional[NDArray], optional
            , by default None
        lower_guess : Optional[float], optional
            Lower bound for rootfinding (we use bracketing), by default -50
        upper_guess : Optional[float], optional
            Upper bound for rootfinding (we use bracketing), by default 50
        verbose : Optional[int], optional
            how much to print, 1 prints the root value,
            2 prints the entire rootfinding output, by default 0

        Returns
        -------
        None
            Doesn't return anything, updates model in place.
        """
        self.rate_pattern = rate_pattern
        def beta_misfit(beta):
            return self._H_func(beta, bucket_populations)-observed_total

        beta_results = root_scalar(beta_misfit, bracket=[
                                   lower_guess, upper_guess], method='toms748')

        if verbose == 2:
            print(beta_results)
        elif verbose == 1:
            print(f'beta={beta_results.root}')

        self.beta_parameter = beta_results.root

        if observed_total_se is not None:
            error_inflation = (1 / \
                self._H_diff(self.beta_parameter, bucket_populations)
                )
            self.beta_standard_error = observed_total_se*error_inflation
            if verbose >= 1:
                print(
                    f"Delta Method Standard Error for Beta: {self.beta_standard_error}")
        else:
            # Reset beta_standard_error to none if we refit
            # Old SE is no longer relevant
            self.beta_standard_error = None

    def predict_rates_SE(self):
        """
        Computes the standard error of the predicted rate in each bucket
        using the delta method, propogating the given standard error on beta

        Returns
        -------
        NDArray
            An array of each of the standard errors in each bucket

        Raises
        ------
        Exception
            Raises an exception when the model's not been fitted, or there's not enough info given
        """
        if self.beta_parameter is None:
            raise Exception("Not fitted, No Beta Parameter Available")
        if self.beta_standard_error is None:
            raise Exception("No Beta Standard Error is available")
        return self._predict_rates_SE(self.beta_parameter, self.beta_standard_error)

    def predict_rates_CI(
            self,
            alpha: Optional[float] = 0.05,
            method: Optional[str] = 'delta-wald')->Tuple[NDArray]:
        """
        Computes a 1-alpha confidence interval on the rate function
        from the standard error on beta

        Parameters
        ----------
        alpha : Optional[float], optional
            1 - confidence level, by default 0.05
        method : Optional[str], optional
            method to use, see notes for more details, by default 'delta-wald'

        Returns
        -------
        Tuple[float]
            Returns a tuple (CI_lower_values,CI_upper_values) 
            where CI_lower_values is a numpy array with the lower end of the confidence interval for each bucket

        Raises
        ------
        Exception
            Raises an exception when not enough info is available

        Notes
        -----
        Method "delta-wald" propogates the standard error on beta through to the
            predicted rate using delta method of Tinv(beta + T(rate_pattern))
        This gives a symmetric confidence interval that will be self consistent with
            any confidence interval on your original measurement

        Method "pushforward" first computes a confidence interval on beta and then
            pushes forward this confidence interval directly into the predicted rate
        This gives a possibly assymetric confidence interval that may be inconsistent
            with your measurement's original confidence interval, but which is likely
            more realistic

        """
        if self.beta_parameter is None:
            raise Exception("Not fitted, No Beta Parameter Available")

        if self.beta_standard_error is None:
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
    )->NDArray:
        """
        Predicts the count in each bucket given the population in each bucket

        Parameters
        ----------
        bucket_populations : NDArray
            population size of each bucket

        Returns
        -------
        NDArray
            estimated occurrences in each bucket

        Notes
        -----
        This is just taking the size of each bucket, and multiplying it by the predicted rate
        """
        return self.predict_rates()*bucket_populations

    def predict_total_count_SE(
        self,
        bucket_populations: NDArray
    )->float:
        """Compute the standard error of the total number of events given an age density
        using delta method on H

        Parameters
        ----------
        bucket_populations : NDArray
            population size of each bucket

        Returns
        -------
        float
            The estimated standard error of the reaggregated estimate

        Raises
        ------
        Exception
            Raises an exception if the model doesn't have enough info

        Notes
        -----
        This is basically for testing as this should be self consistent with the original
        standard error if everything working correctly, since everything
        is just getting expanded out to first order
        """
        if self.beta_parameter is None:
            raise Exception("Not fitted, No Beta Parameter Available")

        if self.beta_standard_error is None:
            raise Exception("No Beta Standard Error is available")

        return self._H_diff(self.beta_parameter, bucket_populations)*self.beta_standard_error

    def predict_count_SE(
        self,
        bucket_populations: NDArray
    )->NDArray:
        """Compute the standard error of the number of events in each bucket given an age density
        using delta method on H

        Parameters
        ----------
        bucket_populations : NDArray
            population size of each bucket

        Returns
        -------
        NDArray
            The standard error of the estimated disaggregated estimate in each bucket

        Raises
        ------
        Exception
            Raises an exception if the model doesn't have enough info
        """
        if self.beta_parameter is None:
            raise Exception("Not fitted, No Beta Parameter Available")

        if self.beta_standard_error is None:
            raise Exception("No Beta Standard Error is available")

        return self.predict_rates_SE()*bucket_populations

    def predict_count_CI(
            self,
            bucket_populations: NDArray,
            alpha: Optional[float] = 0.05,
            method: Optional[str] = 'delta-wald')->Tuple[NDArray]:
        """
        Computes a (one minus alpha) confidence interval on the events occuring in a population
            given an an age density from the standard error on beta

        Parameters
        ----------
        bucket_populations : NDArray
            population size of each bucket
        alpha : Optional[float], optional
            1 - confidence level, by default 0.05
        method : Optional[str], optional
            method to use, see Notes for more details, by default 'delta-wald'

        Returns
        -------
        Tuple[NDArray]
            Returns tuple of numpy arrays with the lower and upper confidence bounds in each bucket
            (CI_lower,CI_upper)

        Raises
        ------
        Exception
            Raises an exception if the model doesn't have enough info

        Notes
        -----
        Method "delta-wald" propogates the standard error on beta through to the
            predicted rate using delta method of Tinv(beta + T(rate_pattern))
        This gives a symmetric confidence interval that will be self consistent with
            any confidence interval on your original measurement

        Method "pushforward" first computes a confidence interval on beta and then
            pushes forward this confidence interval directly into the predicted rate
        This gives a possibly assymetric confidence interval that may be inconsistent
            with your measurement's original confidence interval
        """
        if self.beta_parameter is None:
            raise Exception("Not fitted, No Beta Parameter Available")

        if self.beta_standard_error is None:
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
        observed_total: Optional[float] = None,
        observed_total_se: Optional[float] = None,
        rate_pattern: Optional[NDArray] = None,
        CI_method: Optional[str] = 'delta-wald',
        alpha: Optional[float] = 0.05
    )->Union[Tuple,NDArray]:
        """Splits observed_total into the given bucket populations
        If a observed_total and observed_total_se argument is given,
        then we refit the model to the bucket_populations and
        the observed_total first before predicting

        Parameters
        ----------
        bucket_populations : NDArray
            population size of each bucket
        observed_total : Optional[float], optional
            total of observed quantity across bucket_populations, by default None
        observed_total_se : Optional[float], optional
            se of observed_total, by default None
        rate_pattern : Optional[NDArray], optional
            Rate Pattern to use, should be an estimate of the rates in each bucket
            that we want to rescale. If given, replaces the model's attribute rate pattern, by default None
            None default uses model's rate pattern attribute
        CI_method : Optional[str], optional
            method to use for standard errors, by default 'delta-wald'
        alpha : Optional[float], optional
            1 - (confidence level) for confidence intervals, by default 0.05

        Returns
        -------
        Union[Tuple,NDArray]
            If standard errors are available, this will return the tuple
                (
                    estimate_in_each_bucket,
                    se_of_estimate_bucket,
                    (CI_lower_in_each_bucket,CI_upper_in_each_bucket)
                )
            Otherwise, if standard errors are not available, 
            this will return a numpy array of the disaggregated estimates

        Raises
        ------
        Exception
            Raises an exception if the model doesn't have enough info.
        """
        self.rate_pattern = rate_pattern

        if observed_total is not None:
            self.fit_beta(bucket_populations, observed_total,
                          observed_total_se, verbose=0)

        elif self.beta_parameter is None:
            raise Exception("Not fitted, No Beta Parameter Available")

        if self.beta_standard_error is not None:
            return (
                self.predict_count(bucket_populations),
                self.predict_count_SE(bucket_populations),
                self.predict_count_CI(bucket_populations, alpha=alpha, method=CI_method))

        return self.predict_count(bucket_populations)
    
    def split_groups_rate(
        self,
        bucket_populations: NDArray,
        observed_total: Optional[float] = None,
        observed_total_se: Optional[float] = None,
        rate_pattern: Optional[NDArray] = None,
        CI_method: Optional[str] = 'delta-wald',
        alpha: Optional[float] = 0.05
    )->Union[Tuple,NDArray]:
        """Splits observed_total into the given bucket populations
        and returns the estimated rate in each bucket (instead of the estimated count)
        If a observed_total and observed_total_se argument is given,
        then we refit the model to the bucket_populations and
        the observed_total first before predicting

        Parameters
        ----------
        bucket_populations : NDArray
            population size of each bucket
        observed_total : Optional[float], optional
            total of observed quantity across bucket_populations, by default None
        observed_total_se : Optional[float], optional
            se of observed_total, by default None
        rate_pattern : Optional[NDArray], optional
            Rate Pattern to use, should be an estimate of the rates in each bucket
            that we want to rescale. If given, replaces the model's attribute rate pattern, by default None
            None default uses model's rate pattern attribute
        CI_method : Optional[str], optional
            method to use for standard errors, by default 'delta-wald'
        alpha : Optional[float], optional
            1 - (confidence level) for confidence intervals, by default 0.05

        Returns
        -------
        Union[Tuple,NDArray]
            If standard errors are available, this will return the tuple
                (
                    rate_estimate_in_each_bucket,
                    se_of_rate_estimate_bucket,
                    (CI_lower_in_each_bucket,CI_upper_in_each_bucket)
                )
            Otherwise, if standard errors are not available, 
            this will return a numpy array of the disaggregated estimates

        Raises
        ------
        Exception
            Raises an exception if the model doesn't have enough info.
        """
        self.rate_pattern = rate_pattern

        if observed_total is not None:
            self.fit_beta(bucket_populations, observed_total,
                          observed_total_se, verbose=0)

        elif self.beta_parameter is None:
            raise Exception("Not fitted, No Beta Parameter Available")

        if self.beta_standard_error is not None:
            return (
                self.predict_rates(),
                self.predict_rates_SE(),
                self.predict_rates_CI(alpha=alpha, method=CI_method))

        return self.predict_rates()


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
