import numpy as np
from model import mixture_m_step_poisson, m_step_poisson, poisson_likelihood

def test_mixture_m_step_poisson():
    '''
    Test for the M step Poisson parameter.

    \lambda is the fraction of weighted x and
    the weighted number of observations
    (equation 1.6). Set the weights to a constant
    and \lambda must be equal to the mean of X.
    This is the value assigned to lambda_expected.

    constant is 0.5 for each component.
    Mean of X is (5, 15).
    So lambda must be [5, 15] for each component.
    '''
    num_rows = 11

    q_constant = 0.5

    x1 = np.linspace(0,10,num_rows).reshape(num_rows,1)

    x2 = np.linspace(10,20,num_rows).reshape(num_rows,1)

    X = np.concatenate((x1, x2), axis=1)

    q = np.ones((num_rows,2))*q_constant

    result = mixture_m_step_poisson(X, q, q.shape[1], X.shape[1])

    lambda_actual = result[0]

    lambda_expected = np.array([X.mean(axis=0), X.mean(axis=0)])

    assert np.alltrue(lambda_actual==lambda_expected)

def test_m_step_poisson():
    '''
    Test for the prior probability.

    \pi is the fraction of weighted x and
    the number of observations (equ 1.7).
    Set weights to a constant and \pi must
    be equal to that constant.
    
    Constant is (1/3, 2/3) for the components.
    So \pi must be equal to (1/3, 2/3)

    '''

    num_rows = 11

    constant = np.array([1/3,2/3])

    x1 = np.linspace(0,10,num_rows).reshape(num_rows,1)

    x2 = np.linspace(10,20,num_rows).reshape(num_rows,1)

    X = np.concatenate((x1, x2), axis=1)

    q = np.ones((num_rows,2))*constant

    result = m_step_poisson(X, q)

    pi_actual = result[0]

    pi_expected = constant

    assert np.allclose(pi_actual, pi_expected)

def test_poisson_likelihood():
    '''

    https://en.wikipedia.org/wiki/Poisson_distribution#Once_in_an_interval_events:_The_special_case_of_%CE%BB_=_1_and_k_=_0
    '''
    num_rows = 11
  
    proba_special_case = 1/np.exp(1)

    X = np.zeros((num_rows, 2))

    lambda_special_case = np.array([1, 1])

    pi = np.array([0.5, 0.5])

    mixture_params = (pi, lambda_special_case)

    pmf_actual = poisson_likelihood(0, mixture_params,X)

    pmf_expected = np.ones((num_rows, 2))*proba_special_case

    assert np.allclose(pmf_actual, pmf_expected)


def test_lower_bound_poisson():
    '''
    https://en.wikipedia.org/wiki/Entropy_(information_theory)

    '''
    num_rows = 11

    q_constant = 0.5
    
    X = np.zeros((num_rows, 2))

    lambda_special_case = np.array([1, 1])

    pi = np.array([0.5, 0.5])

    mixture_params = (pi, lambda_special_case)

    q = np.ones((num_rows,2)) * q_constant

