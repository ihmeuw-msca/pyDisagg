import numpy as np
import pytest
from numpy.testing import assert_approx_equal

from pydisagg import models
from pydisagg.disaggregate import split_datapoint

model_list = [
    models.RateMultiplicativeModel(),
    models.LMO_model(1),
    models.LMO_model(5)
]


@pytest.mark.parametrize('model', model_list)
def test_model_consistency(model):
    populations = np.array([2, 5])
    measured_total = 4.8
    measurement_SE = 1
    rate_pattern = np.array([0.2, 0.4])

    result, SE, CI = split_datapoint(measured_total, populations,
                                     rate_pattern, measurement_SE,
                                     model)
    assert_approx_equal(measured_total, np.sum(result))
    assert_approx_equal(measurement_SE, np.sum(SE))
