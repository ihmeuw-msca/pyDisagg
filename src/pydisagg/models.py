"""Module containing specific splitting models with transformations built in"""

import numpy as np
from numpy.typing import NDArray

from pydisagg.DisaggModel import DisaggModel
from pydisagg.transformations import Log, LogModifiedOdds, LogOdds


class RateMultiplicativeModel(DisaggModel):
    """Produces a DisaggModel using the log(rate) transformation with the
    exponent m. This assumes that log(rate)=log(rate_pattern)+beta
    resulting in the current multiplicative model after exponentiating
    Take exp(beta) to recover the multiplier in the model.

    """

    def __init__(self) -> None:
        super().__init__(transformation=Log())

    def fit_beta(
        self,
        observed_total: float,
        rate_pattern: NDArray,
        bucket_populations: NDArray,
        lower_guess: float = -50,
        upper_guess: float = 50,
        verbose: int = 0,
    ) -> None:
        """
        Custom fit_beta for this model, as we can do it without rootfinding.
        """
        beta_val = np.log(
            observed_total / np.sum(bucket_populations * rate_pattern)
        )
        return beta_val


class LMOModel(DisaggModel):
    """DisaggModel using the log-modified odds transformation with the exponent m."""

    def __init__(self, m: float) -> None:
        super().__init__(transformation=LogModifiedOdds(m))


class LogOddsModel(DisaggModel):
    """Produces an DisaggModel assuming multiplicativity in the odds"""

    def __init__(self) -> None:
        super().__init__(transformation=LogOdds())
