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
    ):
        super().__init__(
            parameter_transformation=transformations.LogTransformation(),
        )

    def fit_beta(
        self,
        observed_total: float,
        rate_pattern: NDArray,
        bucket_populations: NDArray,
        lower_guess: float = -50,
        upper_guess: float = 50,
        verbose: Optional[int] = 0,
    ) -> None:
        """
        Custom fit_beta for this model, as we can do it without rootfinding.
        """
        beta_val = np.log(
            observed_total / np.sum(bucket_populations * rate_pattern)
        )
        return beta_val


class LMO_model(DisaggModel):
    """
    DisaggModel using the log-modified odds transformation with the exponent m.
    """

    def __init__(self, m: float):
        super().__init__(
            parameter_transformation=transformations.LogModifiedOddsTransformation(
                m
            ),
        )


class LogOdds_model(DisaggModel):
    """
    Produces an DisaggModel assuming multiplicativity in the odds
    """

    def __init__(
        self,
    ):
        super().__init__(
            parameter_transformation=transformations.LogOddsTransformation(),
        )
