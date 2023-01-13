import splitting.transformations as tr
from numpy.testing import assert_approx_equal


def test_inverse_consistency(T,test_values):
    for x in test_values:
        assert_approx_equal(x,T.inverse(T(x)))

def test_approximate_derivative(T,test_values,h=0.001):
    '''
    Sanity check that the derivatives are are close to correct with a finite difference
    It would have been a good idea to just do everything with Jax from the start
    '''
    for x in test_values:
        assert_approx_equal(T.diff(x),(T(x+h)-T(x-h))/(2*h),2)

transformations=[
    tr.LogTransformation(),
    tr.LogModifiedOddsTransformation(1),
    tr.LogModifiedOddsTransformation(4),
    tr.LogOddsTransformation(),
    ]

test_values=[0.02,0.01,0.5,0.98]
for T in transformations:
    test_inverse_consistency(T,test_values)
    test_approximate_derivative(T,test_values)

