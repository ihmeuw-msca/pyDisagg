from splitting.disaggregate import split_datapoint
from splitting import models
import numpy as np
from numpy.testing import assert_approx_equal


def test_model_consistency(model):
    populations=np.array([2,5])
    measured_total=4.8
    measurement_SE=1
    baseline_prevalence=np.array([0.2,0.4])
    
    result,SE,CI=split_datapoint(measured_total,populations,baseline_prevalence,measurement_SE,model)
    assert_approx_equal(measured_total,np.sum(result))
    assert_approx_equal(measurement_SE,np.sum(SE))

model_list=[
    models.RateMultiplicativeModel(),
    models.LMO_model(1),
    models.LMO_model(5)
    ]
for model in model_list:
    test_model_consistency(model)