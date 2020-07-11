import pytest
import numpy as np
from model import lower_bound_poisson, mixture_m_step_poisson, m_step_poisson, poisson_likelihood


@pytest.fixture(scope="session")
def global_vars():
    num_rows = 11

    X_nozero = np.linspace(0, 10, num_rows).reshape(num_rows, 1)

    X_zero = np.zeros((num_rows, 1))

    pi = np.array([0.5, 0.5])

    return {"num_rows": num_rows,
            "pi": pi,
            "X_nozero": X_nozero,
            "X_zero": X_zero
    }

class TestPoisson:

    @pytest.fixture(autouse=True)
    def set_vars(self, global_vars):
        self.global_vars = global_vars

    def test_mixture_m_step_poisson(self):
        """
        Test for the M step Poisson parameter.

        \lambda is the fraction of weighted x and
        the weighted number of observations
        (equation 1.6). Set the weights to a constant
        and \lambda must be equal to the mean of X.
        This is the value assigned to lambda_expected.
        """
        X = self.global_vars["X_nozero"]

        # q is a two-dimensional array of 0.5
        # i.e. each component distribution is a constant
        q_constant = 0.5

        q = np.ones(shape=(self.global_vars["num_rows"], 2)) * q_constant

        result = mixture_m_step_poisson(X=X, q=q, C=q.shape[1], D=X.shape[1])

        lambda_actual = result[0]

        # Mean of X is (5, 15) so lambda must be
        # [5, 15] for each component
        X_mean = X.mean(axis=0)

        lambda_expected = np.array([X_mean, X_mean])

        assert np.alltrue(lambda_actual == lambda_expected)


    def test_m_step_poisson(self):
        """
        Test for the prior probability.

        \pi is the fraction of the sum of ones
        weighted by q and the number of observations 
        (equ 1.7). If weights q are a constant then 
        \pi must be equal to that constant.
        
        Constant is (1/3, 2/3) for the components.
        

        """
        # Posterior q is (1/3, 2/3) for each observation.
        constant = np.array([1 / 3, 2 / 3])

        q = np.ones(shape=(self.global_vars["num_rows"], 2)) * constant

        result = m_step_poisson(X=self.global_vars["X_nozero"], q=q)

        pi_actual = result[0]

        # So \pi must be equal to (1/3, 2/3)
        pi_expected = constant

        assert np.allclose(pi_actual, pi_expected)


    def test_poisson_likelihood(self):
        """
        Test for the poisson likelihood function.

        Based on the special case when \lambda is 1
        i.e. one event per time period, and k=0 i.e. 
        compute proba that no event happens.
        https://en.wikipedia.org/wiki/Poisson_distribution#Once_in_an_interval_events:_The_special_case_of_%CE%BB_=_1_and_k_=_0
        """
        # Compute the probability of no event occuring
        # i.e. x=0 hence use an array of zeros.
        X = self.global_vars["X_zero"]
        
        lambda_test = np.array([1, 1])

        mixture_params = (self.global_vars["pi"], lambda_test)

        pmf_actual = poisson_likelihood(c=0, 
                                        mixture_params=mixture_params, 
                                        X=X)

        # The expected result is 1/e ~ 0.368.
        expected_proba = 1 / np.exp(1)

        pmf_expected = np.ones(shape=(self.global_vars["num_rows"], 1)) * expected_proba

        assert np.allclose(pmf_actual, pmf_expected)


    def test_lower_bound_poisson(self):
        """
        Test for the Poisson lower bound function.

        Choose special parameters to get a simple analytical
        solution, used to test the function.
        """
        q_constant = 0.5

        X = self.global_vars["X_zero"]

        lambda_test = np.array([1, 1])

        pi = np.array([0.5, 0.5])

        q = np.ones((self.global_vars["num_rows"], 1)) * q_constant

        mixture_params = (pi, lambda_test)

        actual_ll = lower_bound_poisson(X=X, 
                                        pi=pi, 
                                        mixture_params=mixture_params, 
                                        q=q)    

        # With q and \pi both a constant equal to 0.5
        # they cancel each other in equation 2.2 which
        # becomes -1*\sum_1^N i.e. minus the number of rows.
        expected_ll = -1 * self.global_vars["num_rows"]

        assert actual_ll == expected_ll