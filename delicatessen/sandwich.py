#####################################################################################################################
# Functionality to compute the sandwich
#   This script allows for computation of the empirical sandwich variance estimator with just the
#   parameter values and estimating equations. This is to allow computing the sandwich quickly without
#   called the MEstimator procedure itself.
#####################################################################################################################

import warnings

import numpy as np
from scipy.optimize import approx_fprime

from delicatessen.derivative import auto_differentiation, approx_differentiation


def compute_sandwich(stacked_equations, theta, deriv_method='approx', dx=1e-9, allow_pinv=True):
    r"""Compute the empirical sandwich variance estimator given a set of estimating equations and parameter estimates.
    Note that this functionality does not solve for the parameter estimates (unlike ``MEstimator``). Instead, it
    only computes the sandwich for the provided value.

    The empirical sandwich variance estimator is defined as

    .. math::

        V_n(O_i; \theta) = B_n(O_i; \theta)^{-1} F_n(O_i; \theta) \left[ B_n(O_i; \theta)^{-1} \right]^{T}

    where :math:`\psi(O_i; \theta)` is the estimating function,

    .. math::

        B_n(O_i; \theta) = \sum_{i=1}^n \frac{\partial}{\partial \theta} \psi(O_i; \theta),

    and

    .. math::

        F_n(O_i; \theta) = \sum_{i=1}^n \psi(O_i; \theta) \psi(O_i; \theta)^T .

    To compute the bread matrix, :math:`B_n`, the matrix of partial derivatives is computed by using either finite
    difference methods or automatic differentiation. For finite differences, the default is to use SciPy's
    ``approx_fprime`` functionality, which uses forward finite differences. However, you can also use the delicatessen
    homebrew version that allows for forward, backward, and center differences. Automatic differentiation is also
    supported by a homebrew version.

    To compute the meat matrix, :math:`F_n`, only linear algebra methods, implemented through NumPy, are necessary.
    The sandwich is then constructed from these pieces using linear algebra methods from NumPy.

    Parameters
    ----------
    stacked_equations : function, callable
        Function that returns a `v`-by-`n` NumPy array of the estimating equations. See provided examples in the
        documentation for how to construct a set of estimating equations.
    theta : list, set, array
        Parameter estimates to compute the empirical sandwich variance estimator at. Note that this function assumes
        that you have solved for the ``theta`` that correspond to the root of the input estimating equations.
    deriv_method : str, optional
        Method to compute the derivative of the estimating equations for the bread matrix. Options include numerical
        approximation via the forward difference method via SciPy (``'approx'``), forward difference implemented by-hand
        (`'fapprox'`), backward difference implemented by-hand (`'bapprox'`),  central difference implemented by-hand
        (`'capprox'`), or forward-mode automatic differentiation (``'exact'``). Default is ``'approx'``.
    dx : float, optional
        Spacing to use to numerically approximate the partial derivatives of the bread matrix. Here, a small value
        for ``dx`` should be used, since some large values can result in poor approximations. This argument is only
        used with numerical approximation methods. Default is ``1e-9``.
    allow_pinv : bool, optional
        Whether to allow for the pseudo-inverse (via ``numpy.linalg.pinv``) if the bread matrix is determined to be
        non-invertible. If you want to disallow the pseudo-inverse (i.e., use ``numpy.linalg.inv``), set this
        argument to ``False``. Default is ``True``, which  is more robust to the possible bread matrices.

    Returns
    -------
    array :
        Returns a `p`-by-`p` NumPy array for the input ``theta``, where ``p = len(theta)``

    Examples
    --------
    Loading necessary functions and building a generic data set for estimation of the mean

    >>> import numpy as np
    >>> from delicatessen import MEstimator
    >>> from delicatessen import compute_sandwich
    >>> from delicatessen.estimating_equations import ee_mean_variance

    >>> y_dat = [1, 2, 4, 1, 2, 3, 1, 5, 2]

    The following is an illustration of how to compute sandwich covariance using only an estimating equation and the
    parameter values. The mean and variance (that correspond to ``ee_mean_variance``) can be computed using NumPy by

    >>> mean = np.mean(y_dat)
    >>> var = np.var(y_dat, ddof=0)

    For the corresponding estimating equation, we can use the built-in functionality as done below

    >>> def psi(theta):
    >>>     return ee_mean_variance(theta=theta, y=y_dat)

    Calling the sandwich computation procedure

    >>> sandwich_asymp = compute_sandwich(stacked_equations=psi, theta=[mean, var])

    The output sandwich is the *asymptotic* variance (or the variance that corresponds to the standard deviation). To
    get the variance (or the variance that corresponds to the standard error), we rescale ``sandwich`` by the number of
    observations

    >>> sandwich = sandwich_asymp / len(y_dat)

    The standard errors are then

    >>> se = np.sqrt(np.diag(sandwich))

    References
    ----------
    Boos DD, & Stefanski LA. (2013). M-estimation (estimating equations). In Essential Statistical Inference
    (pp. 297-337). Springer, New York, NY.

    Ross RK, Zivich PN, Stringer JSA, & Cole SR. (2024). M-estimation for common epidemiological measures: introduction
    and applied examples. *International Journal of Epidemiology*, 53(2), dyae030.

    Stefanski LA, & Boos DD. (2002). The calculus of M-estimation. The American Statistician, 56(1), 29-38.
    """
    # Evaluating at provided theta values
    evald_theta = np.asarray(stacked_equations(theta=theta))        # Evaluating EE at theta-hat
    if len(theta) == 1:                                             # Number of parameters
        n_obs = evald_theta.shape[0]                                # ... to get number of obs
    else:                                                           # Number of parameters
        n_obs = evald_theta.shape[1]                                # ... to get number of obs

    # Step 1: Compute the bread matrix
    bread = compute_bread(stacked_equations=stacked_equations,      # Call the bread matrix function
                          theta=theta,                              # ... at given theta-hat
                          deriv_method=deriv_method,                # ... with derivative method
                          dx=dx)                                    # ... and approximation
    bread = bread / n_obs                                           # Scale bread by number of obs

    # Step 2: Compute the meat matrix
    meat = compute_meat(stacked_equations=stacked_equations,        # Call the meat matrix function
                        theta=theta)                                # ... at given theta-hat
    meat = meat / n_obs                                             # Scale meat by number of obs

    # Step 3: Construct sandwich from the bread and meat matrices
    sandwich = build_sandwich(bread=bread,                          # Call the sandwich constructor
                              meat=meat,                            # ... with bread and meat matrices above
                              allow_pinv=allow_pinv)                # ... and whether to allow pinv

    # Return the constructed empirical sandwich variance estimator
    return sandwich


def compute_bread(stacked_equations, theta, deriv_method, dx=1e-9):
    r"""Function to compute the bread matrix. The bread matrix is defined as

    .. math::

        B_n(O_i; \theta) = \sum_{i=1}^n \frac{\partial }{\partial \theta} \psi(O_i; \theta)

    where :math:`\psi(O_i; \theta)` is the estimating function.
    To compute the bread matrix, :math:`B_n`, the matrix of partial derivatives is computed by using either finite
    difference methods or automatic differentiation. For finite differences, the default is to use SciPy's
    ``approx_fprime`` functionality, which uses forward finite differences. However, you can also use the delicatessen
    homebrew version that allows for forward, backward, and center differences. Automatic differentiation is also
    supported by a homebrew version.

    Parameters
    ----------
    stacked_equations : function, callable
        Function that returns a `v`-by-`n` NumPy array of the estimating equations. See provided examples in the
        documentation for how to construct a set of estimating equations.
    theta : list, set, array
        Parameter estimates to compute the empirical sandwich variance estimator at. Note that this function assumes
        that you have solved for the ``theta`` that correspond to the root of the input estimating equations.
    deriv_method : str, optional
        Method to compute the derivative of the estimating equations for the bread matrix. Options include numerical
        approximation via the forward difference method via SciPy (``'approx'``), forward difference implemented by-hand
        (`'fapprox'`), backward difference implemented by-hand (`'bapprox'`),  central difference implemented by-hand
        (`'capprox'`), or forward-mode automatic differentiation (``'exact'``). Default is ``'approx'``.
    dx : float, optional
        Spacing to use to numerically approximate the partial derivatives of the bread matrix. Here, a small value
        for ``dx`` should be used, since some large values can result in poor approximations. This argument is only
        used when numerical approximation methods. Default is ``1e-9``.

    Returns
    -------
    array :
        Returns a `p`-by-`p` NumPy array for the input ``theta``, where ``p = len(theta)``
    """
    def estimating_equation(input_theta):
        ef = np.asarray(stacked_equations(theta=input_theta))
        if ef.ndim == 1:
            return np.sum(ef)
        else:
            return np.sum(ef, axis=1)

    # Compute the derivative
    if deriv_method.lower() == 'approx':
        bread_matrix = approx_fprime(xk=theta,
                                     f=estimating_equation,
                                     epsilon=dx)
        if bread_matrix.ndim == 1:
            bread_matrix = np.asarray([bread_matrix, ])
    elif deriv_method.lower() == 'capprox':
        bread_matrix = approx_differentiation(xk=theta,
                                              f=estimating_equation,
                                              method='capprox',
                                              epsilon=dx)
    elif deriv_method.lower() == 'fapprox':
        bread_matrix = approx_differentiation(xk=theta,
                                              f=estimating_equation,
                                              method='fapprox',
                                              epsilon=dx)
    elif deriv_method.lower() == 'bapprox':
        bread_matrix = approx_differentiation(xk=theta,
                                              f=estimating_equation,
                                              method='bapprox',
                                              epsilon=dx)
    elif deriv_method.lower() == "exact":  # Automatic Differentiation
        bread_matrix = auto_differentiation(xk=theta,  # Compute the exact derivative at theta
                                            f=estimating_equation)  # ... for the given estimating equations
    else:
        raise ValueError("The input for deriv_method was "
                         + str(deriv_method)
                         + ", but only 'approx', 'fapprox', 'capprox', 'bapprox' "
                           "and 'exact' are available.")

    # Checking for an issue when trying to invert the bread matrix
    if np.isnan(bread_matrix).any():
        warnings.warn("The bread matrix contains at least one np.nan, so it cannot be inverted. The variance will "
                      "not be calculated. This may be an issue with the provided estimating equations or the "
                      "evaluated theta.",
                      UserWarning)

    # Returning the constructed bread matrix according to SB 2002
    return -1 * bread_matrix


def compute_meat(stacked_equations, theta):
    r"""Function to compute the meat matrix. The meat matrix is defined as

    .. math::

        F_n(O_i; \theta) = \sum_{i=1}^n \psi(O_i; \theta) \psi(O_i; \theta)^T

    where :math:`\psi(O_i; \theta)` is the estimating function.
    Rather than summing over all the individual contributions, this implementation takes a single dot product of the
    stacked estimating functions. This implementation is much faster than summing over :math:`n` matrices.

    Parameters
    ----------
    stacked_equations : function, callable
        Function that returns a `v`-by-`n` NumPy array of the estimating equations. See provided examples in the
        documentation for how to construct a set of estimating equations.
    theta : list, set, array
        Parameter estimates to compute the empirical sandwich variance estimator at. Note that this function assumes
        that you have solved for the ``theta`` that correspond to the root of the input estimating equations.

    Returns
    -------
    array :
        Returns a `p`-by-`p` NumPy array for the input ``theta``, where ``p = len(theta)``
    """
    evald_theta = np.asarray(stacked_equations(theta=theta))  # Evaluating EE at theta-hat
    return np.dot(evald_theta, evald_theta.T)                 # Return the fast dot product calculation


def build_sandwich(bread, meat, allow_pinv=True):
    r"""Function to combine the sandwich elements together. This function takes the bread and meat matrices, does the
    inversions, and then combines them together. This function is separate from ``compute_sandwich`` as it is called
    by both ``compute_sandwich`` and ``MEstimator``.

    Parameters
    ----------
    bread : ndarray
        The bread matrix. The expected input is the output from the ``compute_bread`` function
    meat : ndarray
        The meat matrix. The expected input is the output from the ``compute_meat`` function
    allow_pinv : bool, optional
        Whether to allow for the pseudo-inverse (via ``numpy.linalg.pinv``) if the bread matrix is determined to be
        non-invertible. If you want to disallow the pseudo-inverse (i.e., use ``numpy.linalg.inv``), set this
        argument to ``False``. Default is ``True``, which  is more robust to the possible bread matrices.

    Returns
    -------
    array :
        Returns a `p`-by-`p` NumPy array for the input ``theta``, where ``p = len(theta)``
    """
    # Check if there is an issue with the bread matrix
    if np.any(np.isnan(bread)):                                   # If bread contains NaN, breaks
        return None                                               # ... so give back a NaN

    # Compute the bread inversion
    if allow_pinv:                                                 # Allowing the pseudo-inverse
        bread_invert = np.linalg.pinv(bread)                       # ... then call pinv
    else:                                                          # Only allowing the actual inverse
        bread_invert = np.linalg.inv(bread)                        # ... then call inv

    # Compute the sandwich variance
    sandwich = np.dot(np.dot(bread_invert, meat), bread_invert.T)

    # Return the sandwich covariance matrix
    return sandwich
