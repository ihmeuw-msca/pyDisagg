from typing import Optional,List
from numpy.typing import NDArray
from pandas import DataFrame

from pydisagg import transformations
from pydisagg.splittingmodel import SplittingModel


class RateMultiplicativeModel(SplittingModel):
    '''
    Produces a AgeSplittingModel using the log(rate) transformation with the exponent m.
    This assumes that log(rate)=log(rate_pattern)+beta
    resulting in the current multiplicative model after exponentiating
    Take exp(beta) to recover the multiplier in the model.
    '''

    def __init__(
        self,
        rate_pattern:Optional[NDArray]=None,
        beta_parameter:Optional[float]=None,
        error_inflation:Optional[float]=None,
        beta_standard_error:Optional[float]=None
    ):
        super().__init__(
            parameter_transformation=transformations.LogTransformation(),
            rate_pattern=rate_pattern,
            beta_parameter=beta_parameter,
            error_inflation=error_inflation,
            beta_standard_error=beta_standard_error
        )


class LMO_model(SplittingModel):
    '''
    Produces a AgeSplittingModel using the log-modified odds transformation with the exponent m. 
    '''

    def __init__(
        self,
        m:float,
        rate_pattern:Optional[NDArray]=None,
        beta_parameter:Optional[float]=None,
        error_inflation:Optional[float]=None,
        beta_standard_error:Optional[float]=None
    ):
        super().__init__(
            parameter_transformation=transformations.LogModifiedOddsTransformation(m),
            rate_pattern=rate_pattern,
            beta_parameter=beta_parameter,
            error_inflation=error_inflation,
            beta_standard_error=beta_standard_error
        )


class LogOdds_model(SplittingModel):
    '''
    Produces an AgeSplittingModel assuming multiplicativity in the odds
    '''

    def __init__(
        self,
        rate_pattern:Optional[NDArray]=None,
        beta_parameter:Optional[float]=None,
        error_inflation:Optional[float]=None,
        beta_standard_error:Optional[float]=None
    ):
        super().__init__(
            parameter_transformation=transformations.LogOddsTransformation(),
            rate_pattern=rate_pattern,
            beta_parameter=beta_parameter,
            error_inflation=error_inflation,
            beta_standard_error=beta_standard_error
        )