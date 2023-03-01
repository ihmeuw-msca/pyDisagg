"""Module containing specific splitting models with transformations built in"""
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from pydisagg import transformations
from pydisagg.DisaggModel import DisaggModel


class RateMultiplicativeModel(DisaggModel):
    """
    Produces a DisaggModel using the log(rate) transformation with the exponent m.
    This assumes that log(rate)=log(rate_pattern)+beta
    resulting in the current multiplicative model after exponentiating
    Take exp(beta) to recover the multiplier in the model.
    """

    def __init__(
        self,
        rate_pattern: Optional[NDArray] = None,
        beta_parameter: Optional[float] = None,
        error_inflation: Optional[float] = None,
        beta_standard_error: Optional[float] = None
    ):
        super().__init__(
            parameter_transformation=transformations.LogTransformation(),
            rate_pattern=rate_pattern,
            beta_parameter=beta_parameter,
            error_inflation=error_inflation,
            beta_standard_error=beta_standard_error
        )
    
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
        Custom fit_beta for this model, as we can do it without rootfinding. 
        """
        self.rate_pattern = rate_pattern


        beta_val = np.log(observed_total/np.sum(bucket_populations*self.rate_pattern))

        if verbose>0:
            print(beta_val)

        self.beta_parameter = beta_val
        self.error_inflation = (1 / \
            self._H_diff(self.beta_parameter, bucket_populations))
        if observed_total_se is not None:
            self.beta_standard_error = observed_total_se*self.error_inflation
            if verbose >= 1:
                print(
                    f"Delta Method Standard Error for Beta: {self.beta_standard_error}")
        else:
            # Reset beta_standard_error to none if we refit
            # Old SE is no longer relevant
            self.beta_standard_error = None



class LMO_model(DisaggModel):
    """
    DisaggModel using the log-modified odds transformation with the exponent m.
    """

    def __init__(
        self,
        m: float,
        rate_pattern: Optional[NDArray] = None,
        beta_parameter: Optional[float] = None,
        error_inflation: Optional[float] = None,
        beta_standard_error: Optional[float] = None
    ):
        super().__init__(
            parameter_transformation=transformations.LogModifiedOddsTransformation(
                m),
            rate_pattern=rate_pattern,
            beta_parameter=beta_parameter,
            error_inflation=error_inflation,
            beta_standard_error=beta_standard_error
        )


class LogOdds_model(DisaggModel):
    '''
    Produces an DisaggModel assuming multiplicativity in the odds
    '''

    def __init__(
        self,
        rate_pattern: Optional[NDArray] = None,
        beta_parameter: Optional[float] = None,
        error_inflation: Optional[float] = None,
        beta_standard_error: Optional[float] = None
    ):
        super().__init__(
            parameter_transformation=transformations.LogOddsTransformation(),
            rate_pattern=rate_pattern,
            beta_parameter=beta_parameter,
            error_inflation=error_inflation,
            beta_standard_error=beta_standard_error
        )
