from pydisagg import transformations
from pydisagg.splittingmodel import SplittingModel


class RateMultiplicativeModel(SplittingModel):
    '''
    Produces a AgeSplittingModel using the log(prev) transformation with the exponent m.
    This assumes that log(prev)=log(baseline_prev)+beta
    resulting in the current multiplicative model after exponentiating
    Take exp(beta) to recover the multiplier in the model.
    '''

    def __init__(
        self,
        baseline_prevalence=None,
        beta_parameter=None,
        error_inflation=None,
        beta_standard_error=None
    ):
        super().__init__(
            parameter_transformation=transformations.LogTransformation(),
            baseline_prevalence=baseline_prevalence,
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
        m,
        baseline_prevalence=None,
        beta_parameter=None,
        error_inflation=None,
        beta_standard_error=None
    ):
        super().__init__(
            parameter_transformation=transformations.LogModifiedOddsTransformation(m),
            baseline_prevalence=baseline_prevalence,
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
        baseline_prevalence=None,
        beta_parameter=None,
        error_inflation=None,
        beta_standard_error=None
    ):
        super().__init__(
            parameter_transformation=transformations.LogOddsTransformation(),
            baseline_prevalence=baseline_prevalence,
            beta_parameter=beta_parameter,
            error_inflation=error_inflation,
            beta_standard_error=beta_standard_error
        )
