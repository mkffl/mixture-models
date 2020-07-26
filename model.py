'''
Discrete mixture model functionalities used for the
analyses in my blog articles on Expectation Maximization.
https://mkffl.github.io/

Code is from Martin Krasser's fantastic notebook on GMM.
I only made it a bit more modular to add more distributions
and wrote a few tests.
https://nbviewer.jupyter.org/github/krasserm/bayesian-machine-learning/blob/master/latent_variable_models_part_1.ipynb
'''

from plot_utils import aic, bic
from data.make_wine_dataset import get_X_data_wine
from scipy.stats import poisson, multivariate_normal as mvn
from typing import Any, Tuple, Callable
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def e_step(likelihood: Callable) -> Callable:
    """ 
    Implements the E step of the EM algorithm.
    
    Args:
    likelihood: The mixture probability function
                e.g. Poisson, binomial or multivarite normal

    Returns: 
        The posterior distribution for observations X using 
        the mixture parameters estimated in the M step
    """
    def general_e_step(X, pi, distribution_params):
        N = X.shape[0]
        C = pi.shape[0]

        q = np.zeros((N, C))

        for c in range(C):
            q[:, c] = likelihood(c, distribution_params, X) * pi[c]
        return q / np.sum(q, axis=-1, keepdims=True)
    return general_e_step


def poisson_likelihood(c: int, mixture_params: Tuple[Any], X: np.array) -> np.array:
    """

    Args:
    c: Component index
    mixture_params: Distribution parameters i.e. prior proba 
                    and Poisson rate
    X: Observations
    
    Returns the Poisson probability mass for X
    """
    lambda_param = mixture_params[1]
    return poisson(lambda_param[c]).pmf(X).flatten()

def gaussian_likelihood(c: int, mixture_params: Tuple[Any], X: np.array) -> np.array:
    """
    Multivariate normal function using the mixture parameters.
    Implements equation 3.2.

    Args:
      c: Component index
      mixture_params: Distribution parameters i.e. prior proba, mean and variance
      X: Observations
    
    Returns:
         Gaussian probability density for X
    """
    mu = mixture_params[1]
    sigma = mixture_params[2]
    return mvn(mu[c], sigma[c]).pdf(X)


def m_step(mixture_m_step):
    def general_m_step(X: np.array, q: np.array) -> Callable:
        """
        Computes parameters from data and posterior probabilities.

        Args:
            X: data (N, D).
            q: posterior probabilities (N, C).

        Returns:
            mixture_params, a tuple of
            - prior probabilities (C,).
            - mixture component lambda (C, D).
        """    
        
        N, D = X.shape
        C = q.shape[1]    
        
        # Equation 1.7
        pi = np.sum(q, axis=0) / N

        mixture_params = mixture_m_step(X, q, C, D)
            
        return (pi, ) + mixture_params
    return general_m_step

def mixture_m_step_poisson(X: np.array, q: np.array, C: int, D: int) -> Tuple[np.array]:
    '''
        M step for a Poisson mixture. Implements equation 1.6.
        
        Returns: 
            The updated lambda parameter (C, D).
    '''
    lambda_poisson = q.T.dot(X) / np.sum(q.T, axis=1, keepdims=True)
    return (lambda_poisson, )

def mixture_m_step_gaussian(X, q, C, D):
    # Equation (16)
    sigma = np.zeros((C, D, D))

    # Equation (17)
    mu = q.T.dot(X) / np.sum(q.T, axis=1, keepdims=True)
    
    # Equation (18)
    for c in range(C):
        delta = (X - mu[c])
        sigma[c] = (q[:, [c]] * delta).T.dot(delta) / np.sum(q[:, c])
    return (mu, sigma)

m_step_poisson = m_step(mixture_m_step_poisson)

m_step_gaussian = m_step(mixture_m_step_gaussian)

def lower_bound(likelihood):
    def general_lower_bound(X, pi, mixture_params, q):
        """
        Computes lower bound from data, parameters and posterior probabilities.

        Args:
            X: observed data (N, D).
            pi: prior probabilities (C,).
            mu: mixture component means (C, D).
            sigma: mixture component covariances (C, D, D).
            q: posterior probabilities (N, C).

        Returns:
            The lower bound.
        """    

        N, C = q.shape
        ll = np.zeros((N, C))
        
        # Equation 2.2
        for c in range(C):
            ll[:,c] = np.log(likelihood(c, mixture_params, X))
        return np.sum(q * (ll + np.log(pi) - np.log(np.maximum(q, 1e-8))))
    return general_lower_bound

def random_init_params(mixture_init_params):
    '''
        Initialise mixture distribution parameters.

        \pi_c is initialised as 1/C i.e. components 
        have the same prior probability to be drawn.
    '''
    def general_random_init_params(X, C):
        D = X.shape[1]
        pi = np.ones(C) / C
        mixture_params = mixture_init_params(X, C, D)
        return (pi, ) + mixture_params
    return general_random_init_params

def mixture_init_params_poisson(X, C, D):
    '''
        Initialise Poisson mixture param by drawing 
        samples from a Poisson RV with rate equals to the 
        sample mean.
    '''
    return (poisson(mu=np.mean(X, axis=0)).rvs(C).reshape(C, 1), )

def mixture_init_params_gaussian(X, C, D):
    '''
        Initialise the mixture gaussian mean by drawing 
        samples from a gaussian with shape parameters equal to the
        data set mean and variance.
        The mixture gaussian variance initialisation is the 
        identity matrix (no randomness).
    '''
    rv_loc = np.mean(X, axis=0)
    rv_scale = [np.var(X[:, 0]), np.var(X[:, 1])]
    mu_random_samples = mvn(mean=rv_loc, cov=rv_scale).rvs(C).reshape(C, D)
    sigma = np.tile(np.eye(2), (C, 1, 1))
    return (mu_random_samples, sigma)

# Poisson
random_init_params_poisson = random_init_params(mixture_init_params=mixture_init_params_poisson)

e_step_poisson = e_step(likelihood=poisson_likelihood)

m_step_poisson = m_step(mixture_m_step=mixture_m_step_poisson)

lower_bound_poisson = lower_bound(likelihood=poisson_likelihood)

# Gaussian
random_init_params_gaussian = random_init_params(mixture_init_params=mixture_init_params_gaussian)

e_step_gaussian = e_step(likelihood=gaussian_likelihood)

m_step_gaussian = m_step(mixture_m_step=mixture_m_step_gaussian)

lower_bound_gaussian = lower_bound(likelihood=gaussian_likelihood)


def train(X, 
        C,
        random_init_params,
        e_step, 
        m_step, 
        lower_bound, 
        n_restarts=10, max_iter=50, rtol=1e-3):

    q_best = None
    mixture_params_best = None
    lb_best = -np.inf

    for _ in range(n_restarts):
        mixture_params = random_init_params(X, C)
        pi = mixture_params[0]
        prev_lb = None

        try:
            for st in range(max_iter):
                q = e_step(X, pi, mixture_params)
                mixture_params = m_step(X, q)
                lb = lower_bound(X, pi, mixture_params, q)

                if lb > lb_best:
                    q_best = q
                    mixture_params_best = mixture_params
                    lb_best = lb

                if prev_lb and np.abs((lb - prev_lb) / prev_lb) < rtol:
                    break

                prev_lb = lb
            print(f"{st} steps.")
        except np.linalg.LinAlgError:
            # Singularity. One of the components collapsed
            # onto a specific data point. Start again ...
            pass
    print(f"Total restarts: {_}.")
    return mixture_params_best + (q_best, lb_best)

def run_gmm():
    X = get_X_data_wine().values
    
    C = 2
    
    pi_best, mu_best, sigma_best, q_best, lb_best = train(X, 
                                                    C, 
                                                    random_init_params_gaussian,
                                                    e_step_gaussian,
                                                    m_step_gaussian,
                                                    lower_bound_gaussian,
                                                    n_restarts=10)

    print(f'Lower bound = {lb_best:.2f}')
    
    print(f'BIC = {bic(lb_best, X.shape[1], C):.2f}')

    print(f'AIC = {aic(lb_best, X.shape[1], C, X.shape[0]):.2f}')


if __name__ == "__main__":
    run_gmm()