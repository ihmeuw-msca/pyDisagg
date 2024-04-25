# TODO Fix the disaggregate.py file and then fill this in to make this work
# TODO Also add dataframe splitting stuff
import numpy as np
import pytest
from numpy.testing import assert_approx_equal

from pydisagg import models
from pydisagg.disaggregate import split_datapoint

model_list = [
    models.RateMultiplicativeModel(),
    models.LogOdds_model(),
    models.LMO_model(5),
]


@pytest.mark.parametrize("model", model_list)
def test_count_model_consistency(model):
    populations = np.array([2, 5])
    measured_total = 4.8
    measurement_SE = 1
    rate_pattern = np.array([0.2, 0.4])

    result, SE = split_datapoint(
        measured_total,
        populations,
        rate_pattern,
        measurement_SE,
        model,
        output_type="total",
    )
    assert_approx_equal(measured_total, np.sum(result))
    assert_approx_equal(measurement_SE, np.sum(SE))


@pytest.mark.parametrize("model", model_list)
def test_rate_model_consistency(model):
    populations = np.array([2, 5])
    measured_total = 4.8
    measurement_SE = 1
    rate_pattern = np.array([0.2, 0.4])

    result, SE = split_datapoint(
        measured_total,
        populations,
        rate_pattern,
        measurement_SE,
        model,
        output_type="rate",
    )
    assert_approx_equal(measured_total, np.sum(result * populations))
