import numpy as np
from model import mixture_m_step_poisson, m_step_poisson, poisson_likelihood


@pytest.fixture(scope=module)
def global_vars():
    num_rows = 11

    x1 = np.linspace(0, 10, num_rows).reshape(num_rows, 1)

    x2 = np.linspace(10, 20, num_rows).reshape(num_rows, 1)

    X = np.concatenate((x1, x2), axis=1)
    return {"num_rows": num_rows, 
            "X": X
    }


def test_mixture_m_step_poisson(global_vars):
    """
    Test for the M step Poisson parameter.

    \lambda is the fraction of weighted x and
    the weighted number of observations
    (equation 1.6). Set the weights to a constant
    and \lambda must be equal to the mean of X.
    This is the value assigned to lambda_expected.
    """
    X = global_vars["X"]

    X_mean = X.mean(axis=0)

    # q is a two-dimensional array of 0.5
    # i.e. each component distribution is a constant
    q_constant = 0.5

    q = np.ones(shape=(global_vars["num_rows"], 2)) * q_constant

    result = mixture_m_step_poisson(X=X, q=q, C=q.shape[1], D=X.shape[1])

    lambda_actual = result[0]

    # Mean of X is (5, 15) so lambda must be
    # [5, 15] for each component
    lambda_expected = np.array([X_mean, X_mean])

    assert np.alltrue(lambda_actual == lambda_expected)


def test_m_step_poisson(global_vars):
    """
    Test for the prior probability.

    \pi is the fraction of the sum of ones
    weighted by q and the number of observations 
    (equ 1.7). If weights q are a constant then 
    \pi must be equal to that constant.
    
    Constant is (1/3, 2/3) for the components.
    

    """
    # Constant is (1/3, 2/3) for each observation.
    constant = np.array([1 / 3, 2 / 3])

    q = np.ones(shape=(global_vars["num_rows"], 2)) * constant

    # So \pi must be equal to (1/3, 2/3)
    pi_expected = constant

    result = m_step_poisson(X=global_vars["X"], q=q)

    pi_actual = result[0]

    assert np.allclose(pi_actual, pi_expected)


def test_poisson_likelihood(global_vars):
    """
    Test for the poisson likelihood function.

    Based on the special case when \lambda is 1
    i.e. one event per time period, and k=0 i.e. 
    compute proba that no event happens.
    https://en.wikipedia.org/wiki/Poisson_distribution#Once_in_an_interval_events:_The_special_case_of_%CE%BB_=_1_and_k_=_0
    """
    # Compute the probability of no event occuring
    # i.e. x=0 hence use an array of zeros.
    X = np.zeros(shape=(global_vars["num_rows"], 2))
    
    lambda_test = np.array([1, 1])

    pi = np.array([0.5, 0.5])

    mixture_params = (pi, lambda_test)

    pmf_actual = poisson_likelihood(c=0, mixture_params=mixture_params, X=X)

    # The expected result is c.0.368.
    expected_proba = 1 / np.exp(1)

    pmf_expected = np.ones(shape=(global_vars["num_rows"], 2)) * expected_proba

    assert np.allclose(pmf_actual, pmf_expected)


def test_lower_bound_poisson():
    """
    https://en.wikipedia.org/wiki/Entropy_(information_theory)

    """
    num_rows = 11

    q_constant = 0.5

    X = np.zeros((num_rows, 2))

    lambda_special_case = np.array([1, 1])

    pi = np.array([0.5, 0.5])

    mixture_params = (pi, lambda_special_case)

    q = np.ones((num_rows, 2)) * q_constant
