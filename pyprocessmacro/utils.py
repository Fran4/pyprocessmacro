from numpy.linalg import inv, LinAlgError
from numpy import dot
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import re


def z_score(conf: float) -> float:
    """
    :param conf: Desired level of confidence
    :return: The Z-score corresponding to the level of confidence desired.
    """
    return norm.ppf((100 - (100 - conf) / 2) / 100)


def bias_corrected_ci(estimate: float, samples: np.array, conf: float = 95) -> (float, float):
    """
    Return the bias-corrected bootstrap confidence interval for an estimate
    :param estimate: Numerical estimate in the original sample
    :param samples: Nx1 array of bootstrapped estimates
    :param conf: Level of the desired confidence interval
    :return: Bias-corrected bootstrapped LLCI and ULCI for the estimate.
    """
    ptilde = (samples < estimate).mean()
    Z = norm.ppf(ptilde)
    Zci = z_score(conf)
    Zlow, Zhigh = -Zci + 2 * Z, Zci + 2 * Z
    plow, phigh = norm._cdf(Zlow), norm._cdf(Zhigh)
    llci = np.percentile(samples, plow * 100, interpolation='lower')
    ulci = np.percentile(samples, phigh * 100, interpolation='higher')
    return llci, ulci


def percentile_ci(samples: np.array, conf=95) -> np.array:
    """
    Based on an array of values, returns the lower and upper percentile bound for a desired level of confidence
    :param samples: NxK array of samples
    :param conf: Desired level of confidence
    :return: 2xK array corresponding to the lower and upper percentile bounds for K estimates.
    """
    lower = (100 - conf) / 2
    upper = 100 - lower
    return np.percentile(samples, [lower, upper])


def fast_OLS(endog: np.array, exog: np.array) -> np.array:
    """
    A simple function for (X'X)^(-1)X'Y
    :return: The Kx1 array of estimated coefficients.
    """
    try:
        return dot(dot(inv(dot(exog.T, exog)), exog.T), endog).squeeze()
    except LinAlgError:
        raise LinAlgError


def logit_cdf(X):
    """
    The CDF of the logistic function.
    :param X: Value at which to estimate the CDF
    :return: The logistic function CDF, evaluated at X
    """
    idx = X > 0
    out = np.empty(X.size, dtype=float)
    out[idx] = 1 / (1 + np.exp(-X[idx]))
    exp_X = np.exp(X[~idx])
    out[~idx] = exp_X / (1 + exp_X)
    return out


def logit_score(endog: np.array, exog: np.array, params: np.array, n_obs: int) -> np.array:
    """
    The score of the logistic function.
    :param endog: Nx1 vector of endogenous predictions
    :param exog: NxK vector of exogenous predictors
    :param params: Kx1 vector of parameters for the predictors
    :param n_obs: Number of observations
    :return: The score, a Kx1 vector, evaluated at `params'
    """
    return dot(endog - logit_cdf(dot(exog, params)), exog) / n_obs


def logit_hessian(exog: np.array, params: np.array, n_obs: int) -> np.array:
    """
    The hessian of the logistic function.
    :param exog: NxK vector of exogenous predictors
    :param params: Kx1 vector of parameters for the predictors
    :param n_obs: Number of observations
    :return: The Hessian, a KxK matrix, evaluated at `params'
    """
    L = logit_cdf(np.dot(exog, params))
    return -dot(L * (1 - L) * exog.T, exog) / n_obs


def fast_optimize(endog: np.array, exog: np.array, n_obs: int = 0, n_vars: int = 0, max_iter: int = 10000,
                  tolerance: float = 1e-10):
    """
    A convenience function for the Newton-Raphson method to evaluate a logistic model.
    :param endog: Nx1 vector of endogenous predictions
    :param exog: NxK vector of exogenous predictors
    :param n_obs: Number of observations N
    :param n_vars: Number of exogenous predictors K
    :param max_iter: Maximum number of iterations
    :param tolerance: Margin of error for convergence
    :return: The error-minimizing parameters for the model.
    """
    iterations = 0
    oldparams = np.inf
    newparams = np.repeat(0, n_vars)
    while iterations < max_iter and np.any(np.abs(newparams - oldparams) > tolerance):
        oldparams = newparams
        try:
            H = logit_hessian(exog, oldparams, n_obs)
            newparams = oldparams - dot(inv(H), logit_score(endog, exog, oldparams, n_obs))
        except LinAlgError:
            raise LinAlgError
        iterations += 1
    return newparams


def bootstrap_sampler(n_obs: int, seed: int = None):
    """
    A generator of bootstrapped indices. Samples with repetition a list of indices.
    :param n_obs: Number of observations
    :return: Bootstrapped indices of size n_obs
    """
    seeder = np.random.RandomState(seed)
    seeder.seed(seed)
    while True:
        yield seeder.randint(n_obs, size=n_obs)


def eigvals(exog):
    """
    Return the eigenvalues of a matrix of endogenous predictors.
    :param exog: NxK matrix of exogenous predictors.
    :return: Kx1 vector of eigenvalues, sorted in decreasing order of magnitude.
    """
    return np.sort(np.linalg.eigvalsh(dot(exog.T, exog)))[::-1]


def eval_expression(expr: np.array, values: dict = None) -> np.array:
    """
    Evaluate a symbolic expression and returns a numerical array.
    :param expr: A symbolic expression to evaluate, in the form of a N_terms * N_Vars matrix
    :param values: None, or a dictionary of variable:value pairs, to substitute in the symbolic expression.
    :return: An evaled expression, in the form of an N_terms array.
    """
    n_coeffs = expr.shape[0]
    evaled_expr = np.zeros(n_coeffs)
    for (i, term) in enumerate(expr):
        if values:
            evaled_term = np.array([values.get(elem, 0) if isinstance(elem, str) else elem for elem in term])
        else:
            evaled_term = np.array(
                [0 if isinstance(elem, str) else elem for elem in term])  # All variables at 0
        evaled_expr[i] = np.product(evaled_term.astype(float))  # Gradient is the product of values
    return evaled_expr


def plot_errorbars(x, y, yerrlow: float, yerrhigh: float, plot_kws: dict = None,
                   err_kws: dict = None, *args, **kwargs):
    yerr = [yerrlow, yerrhigh]
    err_kws_final = kwargs.copy()
    err_kws_final.update(err_kws)
    err_kws_final.update({'marker': "", 'fmt': 'none', 'label': '', "zorder": 3})
    plot_kws_final = kwargs.copy()
    plot_kws_final.update(plot_kws)
    plt.plot(x, y, *args, **plot_kws_final)
    plt.errorbar(x, y, yerr, *args, **err_kws_final)
    return None


def plot_errorbands(x, y, llci: float, ulci: float, plot_kws: dict = None, err_kws: dict = None,
                    *args, **kwargs):
    err_kws_final = kwargs.copy()
    err_kws_final.update(err_kws)
    err_kws_final.update({'label': ''})
    plot_kws_final = kwargs.copy()
    plot_kws_final.update(plot_kws)
    plt.plot(x, y, *args, **plot_kws_final)
    plt.fill_between(x, llci, ulci, *args, **err_kws_final)
    return None


def list_moderators(terms: list, of="x"):
    """
    :param vars: A list of regression terms
    :param of: The variable that is moderated
    :return: A set of all moderators of the variable "of".
    """
    pattern = re.compile(r"^(?:x\*)([a-z])$")

