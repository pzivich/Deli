import warnings
import numpy as np
import scipy as sp
from scipy.stats import norm
from delicatessen.derivative import PrimalTangentPairs as PTPair


def logit(prob):
    r"""Logistic transformation. Used to transform probabilities into log-odds.

    .. math::

        \log \left( \frac{p}{1-p} \right)

    Parameters
    ----------
    prob : float, ndarray
        A single probability or an array of probabilities

    Returns
    -------
    array :
        logit-transformed values
    """
    return np.log(prob / (1 - prob))


def inverse_logit(logodds):
    r"""Inverse logistic transformation. Used to transform log-odds into probabilities.

    .. math::

        \frac{1}{1 + \exp(o)}

    Parameters
    ----------
    logodds : float, ndarray
        A single log-odd or an array of log-odds

    Returns
    -------
    array :
        inverse-logit transformed values
    """
    return 1 / (1 + np.exp(-logodds))


def identity(value):
    """Identity transformation. Used to transform input into itself (i.e., no transformation in applied).

    Note
    ----
    This function doesn't actually apply any transformation. It is used for arbitrary function calls that apply
    transformations, and this is called when no transformation is to be applied

    Parameters
    ----------
    value : float, ndarray
        A single value or an array of values

    Returns
    -------
    value
    """
    return value


def polygamma(n, x):
    """Polygamma function. This is a wrapper function of ``scipy.special.polygamma`` meant to enable automatic
    differentation with ``delicatessen``. When the input is a ``PrimalTangentPairs`` object, then an internal function
    that implements the polygamma function is called. Otherwise, ``scipy.special.polygamma`` is called for the input
    object.

    Parameters
    ----------
    n : int
        Order of the derivative of the digamma function
    x : int, float, ndarray
        Real valued input

    Returns
    -------
    Return type depends on the input type (``PrimalTangentPairs`` will return ``PrimalTangentPairs``, otherwise will
    return ``ndarray``).
    """
    if isinstance(x, np.ndarray):
        storage = []
        for xi in x:
            pgnxi = polygamma(n=n, x=xi)
            storage.append(pgnxi)
        return np.array(storage)
    elif isinstance(x, PTPair):
        return x.polygamma(n=n)
    else:
        return sp.special.polygamma(n=n, x=x)


def digamma(z):
    """Digamma function. This is a wrapper function of ``scipy.special.digamma`` meant to enable automatic
    differentation with ``delicatessen``. When the input is a ``PrimalTangentPairs`` object, then an internal function
    that implements the digamma function is called. Otherwise, ``scipy.special.digamma`` is called for the input
    object.

    Parameters
    ----------
    x : int, float, ndarray
        Real valued input

    Returns
    -------
    Return type depends on the input type (``PrimalTangentPairs`` will return ``PrimalTangentPairs``, otherwise will
    return ``ndarray``).
    """
    return polygamma(n=0, x=z)


def standard_normal_cdf(x):
    """Cumulative distribution function for the standard normal distribution. This is a wrapper function of
    ``scipy.stats.norm.cdf`` meant to enable automatic differentation with ``delicatessen``. When the input is a
    ``PrimalTangentPairs`` object, then an internal function that implements the CDF function is called. Otherwise,
    ``scipy.stats.norm.cdf`` is called for the input object.

    Parameters
    ----------
    x : int, float, ndarray
        Real valued input

    Returns
    -------
    Return type depends on the input type (``PrimalTangentPairs`` will return ``PrimalTangentPairs``, otherwise will
    return ``ndarray``).
    """
    if isinstance(x, np.ndarray):
        storage = []
        for xi in x:
            normxi = standard_normal_cdf(x=xi)
            storage.append(normxi)
        return np.array(storage)
    elif isinstance(x, PTPair):
        return x.normal_cdf()
    else:
        return norm.cdf(x=x)


def standard_normal_pdf(x):
    """Probability density function for the standard normal distribution. This is a wrapper function of
    ``scipy.stats.norm.pdf`` meant to enable automatic differentation with ``delicatessen``. When the input is a
    ``PrimalTangentPairs`` object, then an internal function that implements the PDF function is called. Otherwise,
    ``scipy.stats.norm.pdf`` is called for the input object.

    Parameters
    ----------
    x : int, float, ndarray
        Real valued input

    Returns
    -------
    Return type depends on the input type (``PrimalTangentPairs`` will return ``PrimalTangentPairs``, otherwise will
    return ``ndarray``).
    """
    if isinstance(x, np.ndarray):
        storage = []
        for xi in x:
            normxi = standard_normal_pdf(x=xi)
            storage.append(normxi)
        return np.array(storage)
    elif isinstance(x, PTPair):
        return x.normal_pdf()
    else:
        return norm.pdf(x=x)


def robust_loss_functions(residual, loss, k, a=None, b=None):
    r"""Loss functions for robust mean and robust regression estimating equations. This function is called internally
    for ``ee_mean_robust`` and ``ee_robust_regression``. This function can also be accessed, so user's can easily adapt
    their own regression models into robust regression models using the pre-defined loss functions.

    Note
    ----
    The loss functions here are technically the first-order derivatives of the loss functions you see in the literature.


    The following score of the loss functions, :math:`f_k()`, are available.

    Andrew's Sine

    .. math::

        f_k(x) = I(k \pi \le x \le k \pi) \times \sin(x/k)

    Huber

    .. math::

        f_k(x) = x I(-k < x < k) + \text{sign}(x) k (1 - I(-k < x < k))

    Tukey's biweight

    .. math::

        f_k(x) = x I(-k < x < k) + x \left( 1 - (x/k)^2 \right)^2

    Hampel (Hampel's add two additional parameters, :math:`a` and :math:`b`)

    .. math::

        f_{k,a,b}(x) =
        \begin{bmatrix}
            I(-a < x < a) \times x \\
            + I(a \le |x| < b) \times a \times \text{sign}(x) \\
            + I(b \le x < k) \times a \frac{k - x}{k - b} \\
            + I(-k \ge x > -b) \times -a \frac{-k + x}{-k + b} \\
            + I(|x| \ge k) \times 0
        \end{bmatrix}

    Parameters
    ----------
    residual : ndarray, vector, list
        1-dimensional vector of n observed values. Input should consists of the residuals (the difference between the
        observed value and the predicted value). For the robust mean, this is :math:`Y_i - \mu`. For robust regression,
        this is :math:`Y_i - X_i^T \beta`
    loss : str
        Loss function to use. Options include: 'andrew', 'hampel', 'huber', 'minimax', 'tukey'
    k : int, float
        Tuning parameter for the corresponding loss function. Note: no default is provided, since each loss function
        has different recommendations.
    a : int, float, None, optional
        Lower parameter for the 'hampel' loss function
    b : int, float, None, optional
        Upper parameter for the 'hampel' loss function

    Returns
    -------
    array :
        Returns a 1-by-n NumPy array evaluated for the input theta and residual

    Examples
    --------
    Using the robust loss function

    >>> import numpy as np
    >>> from delicatessen.utilities import robust_loss_functions

    Some generic data to stand-in for the residuals

    >>> residuals = np.random.standard_cauchy(size=20)

    Huber's loss function

    >>> robust_loss_functions(residuals, loss='huber', k=1.345)

    Andrew's Sine

    >>> robust_loss_functions(residuals, loss='andrew', k=1.339)

    Tukey's biweight

    >>> robust_loss_functions(residuals, loss='tukey', k=4.685)

    Hampel's loss function

    >>> robust_loss_functions(residuals, loss='hampel', k=8, a=2, b=4)

    References
    ----------
    Andrews DF. (1974). A robust method for multiple linear regression. *Technometrics*, 16(4), 523-531.

    Beaton AE & Tukey JW (1974). The fitting of power series, meaning polynomials, illustrated on band-spectroscopic
    data. *Technometrics*, 16(2), 147-185.

    Hampel FR. (1971). A general qualitative definition of robustness. *The Annals of Mathematical Statistics*,
    42(6), 1887-1896.

    Huber PJ. (1964). Robust Estimation of a Location Parameter. *The Annals of Mathematical Statistics*, 35(1), 73–101.

    Huber PJ, Ronchetti EM. (2009) Robust Statistics 2nd Edition. Wiley. pgs 98-100
    """
    # Checking type for later .lower() calls so informative error
    if not isinstance(loss, str):
        raise ValueError("The provided loss function should be a string.")

    # Huber function
    elif loss.lower() == "huber":
        xr = np.clip(residual, a_min=-k, a_max=k)

    # Tukey's biweight
    elif loss.lower() == "tukey":
        xr = np.where(np.abs(residual) <= k, residual * (1-(residual/k)**2)**2, 0)

    # Andrew's Sine
    elif loss.lower() == "andrew":
        xr = np.where(np.abs(residual) <= k*np.pi,
                      np.sin(residual/k), np.nan)
        xr = np.where(residual > k*np.pi, 0, xr)
        xr = np.where(residual < -k*np.pi, 0, xr)

    # Hampel
    elif loss.lower() == "hampel":
        if a is None or b is None:
            raise ValueError("The 'hampel' loss function requires the optional `a` and `b` arguments")
        if not a < b < k:
            raise ValueError("The 'hampel' loss function requires that a < b < k")
        xr = np.where(np.abs(residual) < a, residual, np.nan)
        xr = np.where((a <= residual) & (residual < b), a, xr)
        xr = np.where((-a >= residual) & (residual > -b), -a, xr)
        xr = np.where((b <= residual) & (residual < k), (k - residual)/(k - b)*a, xr)
        xr = np.where((-b >= residual) & (residual > -k), (-k - residual)/(-k + b)*-a, xr)
        xr = np.where(np.abs(residual) >= k, 0, xr)

    # Catch-all value error for the provided loss function
    else:
        raise ValueError("The loss function "+str(loss)+" is not available.")

    # Returning the updated values
    return xr


def regression_predictions(X, theta, covariance, offset=None, alpha=0.05):
    r"""Generate predicted values of the outcome given a design matrix, point estimates, and covariance matrix.
    This functionality computes :math:`\hat{Y}`, :math:`\hat{Var}\left(\hat{Y}\right)`, and corresponding Wald-type
    :math:`(1 - \alpha) \times` 100% confidence intervals from estimated coefficients and covariance of a regression
    model given a set of specific covariate values.

    This function is a helper function to compute the predictions from a regression model for a set of given :math:`X`
    values. Importantly, this method allows for the variance of :math:`\hat{Y}` to be estimated without having to expand
    the estimating equations. As such, this functionality is meant to be used after ``MEstimator`` has been used to
    estimate the coefficients (i.e., this function is for use after the M-estimator has computed the results for the
    chosen regression model). Therefore, this function should be viewed as a post-processing functionality for
    generating plots or tables.

    Note
    ----
    No tranformations are applied by this function. So, input from a logistic model will generate the *log-odds* of the
    outcome (not probability).


    Parameters
    ----------
    X : ndarray, list, vector
        2-dimensional vector of values to generate predicted variances for. The number of columns must match the number
        of coefficients / parameters in ``theta``.
    theta : ndarray
        Estimated coefficients from ``MEstimator.theta``.
    covariance : ndarray
        Estimated covariance matrix from ``MEstimator.variance``.
    offset : ndarray, None, optional
        A 1-dimensional offset to be included in the model. Default is None, which applies no offset term.
    alpha : float, optional
        The :math:`\alpha` level for the corresponding confidence intervals. Default is 0.05, which calculate the
        95% confidence intervals. Notice that :math:`0<\alpha<1`.

    Returns
    -------
    array :
        Returns a n-by-4 NumPy array with the 4 columns correspond to the predicted outcome, variance of the predictied
        outcome, lower confidence limit, and upper confidence limit.

    Examples
    --------
    The following is a simple example demonstrating how this function can be used to plot a regression line and
    corresponding 95% confidence intervals.

    >>> import numpy as np
    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt
    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_regression
    >>> from delicatessen.utilities import regression_predictions

    Some generic data to estimate the regression model with

    >>> n = 50
    >>> data = pd.DataFrame()
    >>> data['X'] = np.random.normal(size=n)
    >>> data['Z'] = np.random.normal(size=n)
    >>> data['Y'] = 0.5 + 2*data['X'] - 1*data['Z'] + np.random.normal(loc=0, size=n)
    >>> data['C'] = 1

    Estimating the corresponding regression model parameters

    >>> def psi(theta):
    >>>     return ee_regression(theta=theta, X=data[['C', 'X', 'Z']], y=data['Y'], model='linear')

    >>> estr = MEstimator(stacked_equations=psi, init=[0., 0., 0.,])
    >>> estr.estimate(solver='lm')

    To create a line plot of our regression line, we need to first create a new set of covariate values that are evenly
    spaced across the range of the predictor values. Here, we will plot the relationship between ``Z`` and ``Y`` while
    holding ``X=0``.

    >>> pred = pd.DataFrame()
    >>> pred['Z'] = np.linspace(np.min(data['Z']), np.max(data['Z']), 200)
    >>> pred['X'] = 0
    >>> pred['C'] = 1

    Now the predicted values of the outcome, and confidence intervals to plot

    >>> Xp = pred[['C', 'X', 'Z']]
    >>> yhat = regression_predictions(X=Xp, theta=estr.theta, covariance=estr.variance)

    Finally, the predicted values can be plotted (using matplotlib)

    >>> plt.plot(pred['Z'], yhat[:, 0], '-', color='blue')
    >>> plt.fill_between(pred['Z'], yhat[:, 2], yhat[:, 3], alpha=0.25, color='blue')
    >>> plt.show()

    For predicting with a Poisson or logistic model, one may want to transform the predicted values and confidence
    intervals to another measure. For the logistic model, the predicted log-odds can easily be transformed using
    ``delicatessen.utilities.inverse_logit``. For the Poisson model, predictions can easily be transformed using
    ``numpy.exp``.
    """
    # Check valid alpha value is being provided
    if not 0 < alpha < 1:
        raise ValueError("`alpha` must be 0 < a < 1")

    # Process offset term
    if offset is None:                                 # When offset is None
        offset = 0                                     # ... modify by adding a zero (i.e., no mod)
    else:                                              # Otherwise
        offset = np.asarray(offset)[:, None]           # ... ensure that a NumPy array is passed forward

    # Setup inputs for matrix multiplications
    x = np.asarray(X)
    b = np.asarray(theta)
    c = np.asarray(covariance)

    # Predicted Y
    yhat = np.dot(x, b) + offset

    # Predicted Y variance / standard error
    yhat_var = np.sum(np.dot(x, c) * x,
                      axis=1)

    # Confidence limit of predictions
    yhat_se = np.sqrt(yhat_var)                        # Taking square root to get SE
    z_alpha = norm.ppf(1 - alpha/2, loc=0, scale=1)    # Z_alpha value for CI
    lower_ci = yhat - z_alpha*yhat_se                  # Lower CI
    upper_ci = yhat + z_alpha*yhat_se                  # Upper CI

    # Return estimates and variance
    return np.vstack([yhat, yhat_var, lower_ci, upper_ci]).T


def spline(variable, knots, power=3, restricted=True, normalized=False):
    r"""Generate generic polynomial spline terms for a given NumPy array and pre-specified knots. Default is restricted
    cubic splines but unrestricted splines to different polynomial terms can also be specified.

    Unrestricted splines for knot :math:`k` are generated using the following formula

    .. math::

        s_k(X) = I(X > k) \left\{ X - k \right\}^a

    where :math:`a` is the power (:math:`a=3` for cubic splines).

    Restricted splines are generated via

    .. math::

        r_k(X) = I(X > k) \left\{ X - k \right\}^a - s_K(X)

    where :math:`K` is largest knot value.

    Splines are normalized by the upper knot minus the lower knot to the corresponding power. Normalizing the splines
    can be helpful for the root-finding procedure.

    Parameters
    ----------
    variable : ndarray, vector, list
        1-dimensional vector of observed values. Input should consists of the variable to generate spline terms for
    knots : ndarray, vector, list
        1-dimensional vector of pre-specified knot locations. All knots should be between the minimum and maximum
        values of the input variable
    power : int, float, optional
        Power or polynomial term to use for the splines. Default is 3, which corresponds to cubic splines
    restricted : bool, optional
        Whether to generate restricted or unrestricted splines. Default is True, which corresponds to restricted
        splines. Restricted splines return one less column than the number of knots, whereas unrestricted splines
        return the same number of columns as knots
    normalized : bool, optional
        Whether to normalize, or divide, the spline terms by the difference between the upper and lower knots. Default
        is ``False``, which corresponds to unnormalized splines.

    Returns
    -------
    ndarray :
        A 2-dimensional array of the spline terms in ascending order of the knots.

    Examples
    --------
    Construction of spline variables should be done similar to the following

    >>> import numpy as np
    >>> import pandas as pd
    >>> from delicatessen.utilities import spline

    Some generic data to estimate a generalized additive model

    >>> x = np.random.normal(size=200)

    A restricted quadratic spline with 3 knots (at -1, 0, 1) can be generated using the following function call

    >>> spline(variable=x, knots=[-1, 0, 1], power=2, restricted=True)

    This function will return a 2 by 200 array here. Other knot specifications, other powers, and unrestricted splines
    can also be generated by updating the corresponding arguments.

    References
    ----------
    Mulla ZD (2007). Spline regression in clinical research. *West Indian Med J*, 56(1), 77.
    """
    # Processing input and output arrays
    knots = sorted(knots)                           # Sorting the order of the knots
    n_cols = len(knots)                             # Column number is based on the number of provided knots
    x = np.asarray(variable)                        # Converting input variable to another array
    spline_terms = np.empty((x.shape[0], n_cols))   # Creating the output spline array

    # Determining normalization
    if normalized:                                     # Rescale spline terms so less extreme ranges (helps root-finder)
        if n_cols == 1:                                # When only a single knot
            divisor = knots[0]**power                  # ... scale by knot to that power
        else:                                          # Otherwise
            divisor = (knots[-1] - knots[0])**power    # ... range of the knots to that power
    else:                                              # If not rescaling (only option up to v2.0)
        divisor = 1                                    # ... divisor is 1

    # Generating each spline with it's corresponding term
    for i in range(n_cols):
        spline_terms[:, i] = np.where(x > knots[i], (x - knots[i])**power, 0)
        spline_terms[:, i] = np.where(np.isnan(x), np.nan, spline_terms[:, i])

    # Logic for unrestricted and restricted splines
    if restricted is False:
        return spline_terms / divisor
    else:
        for i in range(n_cols - 1):
            spline_terms[:, i] = np.where(x > knots[i], spline_terms[:, i] - spline_terms[:, -1], 0)
            spline_terms[:, i] = np.where(np.isnan(x), np.nan, spline_terms[:, i])
        return spline_terms[:, :-1] / divisor


def additive_design_matrix(X, specifications, return_penalty=False):
    r"""Generate an additive design matrix for generalized additive models (GAM) given a set of spline specifications to
    apply.

    Note
    ----
    This function is interally called by ``ee_additive_regression``. This function can also be called to aid in easily
    generating predicted values.

    Parameters
    ----------
    X : ndarray, vector, list
        Input independent variable data.
    specifications : ndarray, vector, list
        A list of dictionaries that define the hyperparameters for the spline (e.g., number of knots, strength of
        penalty). For terms that should not have splines, ``None`` should be specified instead (see examples below).
        Each dictionary supports the following parameters:
        "knots", "natural", "power", "penalty"
        knots (list): controls the position of the knots, with knots are placed at given locations. There is no
        default, so must be specified by the user.
        natural (bool): controls whether to generate natural (restricted) or unrestricted splines.
        Default is ``True``, which corresponds to natural splines.
        power (float): controls the power to raise the spline terms to. Default is 3, which corresponds to cubic
        splines.
        penalty (float): penalty term (:math:`\lambda`) applied to each corresponding spline basis term. Default is 0,
        which applies no penalty to the spline basis terms.
        normalized (bool): whether to normalize the spline terms. Default is ``False``, with a default change coming
        with v3.0 release.
    return_penalty : bool, optional
        Whether the list of the corresponding penalty terms should also be returned. This functionality is used
        internally to create the list of penalty terms to provide the Ridge regression model, where only the spline
        terms are penalized. Default is False.

    Returns
    -------
    array :
        Returns a (b+k)-by-n design matrix as a NumPy array, where b is the number of columns in the input array and k
        is determined by the specifications of the spline basis functions.

    Examples
    --------
    Construction of a design matrix for an additive model should be done similar to the following

    >>> import numpy as np
    >>> import pandas as pd
    >>> from delicatessen.utilities import additive_design_matrix

    Some generic data to estimate a generalized additive model

    >>> n = 200
    >>> d = pd.DataFrame()
    >>> d['X'] = np.random.normal(size=n)
    >>> d['Z'] = np.random.normal(size=n)
    >>> d['W'] = np.random.binomial(n=1, p=0.5, size=n)
    >>> d['C'] = 1

    To begin, consider the simple input design matrix of ``d[['C', 'X']]``. This initial design matrix consists of an
    intercept term and a continuous term. Here, we will specify a natural spline with 20 knots for the second term only

    >>> x_knots = np.linspace(np.min(d['X'])+0.1, np.max(d['X'])-0.1, 20)
    >>> specs = [None, {"knots": x_knots, "penalty": 10}]
    >>> Xa_design = additive_design_matrix(X=d[['C', 'X']], specifications=specs)

    Other optional specifications are also available. Here, we will specify an unrestricted quadratic spline with a
    penalty of 5.5 for the second column of the design matrix.

    >>> specs = [None, {"knots": [-2, -1, 0, 1, 2], "natural": False, "power": 2, "penalty": 5.5}]
    >>> Xa_design = additive_design_matrix(X=d[['C', 'X']], specifications=specs)

    Now consider the input design matrix of ``d[['C', 'X', 'Z', 'W']]``. This initial design matrix consists of an
    intercept, two continuous, and a categorical term. Here, we will specify splines for both continuous terms

    >>> x_knots = np.linspace(np.min(d['X'])+0.1, np.max(d['X'])-0.1, 20)
    >>> z_knots = np.linspace(np.min(d['Z'])+0.1, np.max(d['Z'])-0.1, 10)
    >>> specs = [None,                              # Intercept term
    >>>          {"knots": x_knots, "penalty": 25}, # X (continuous)
    >>>          {"knots": z_knots, "penalty": 15}, # Z (continuous)
    >>>          None]                              # W (categorical)
    >>> Xa_design = additive_design_matrix(X=d[['C', 'X', 'Z', 'W']], specifications=specs)

    Notice that the two continuous terms have different spline specifications.

    Finally, we could opt to only generate a spline basis for one of the continuous variables

    >>> specs = [None,                              # Intercept term
    >>>          {"knots": x_knots, "penalty": 25}, # X (continuous)
    >>>          None,                              # Z (continuous)
    >>>          None]                              # W (categorical)
    >>> Xa_design = additive_design_matrix(X=d[['C', 'X', 'Z', 'W']], specifications=specs)

    Specification of splines can be modified and paired in a variety of ways. These are determined by the object type
    in the specification list, and the input dictionary for the spline terms.
    """
    def generate_spline(variable, specification):
        """Internal function to call the spline functionality. This function merely calls the spline function with the
        corresponding specifications. This was built as an internal function to simply future maintenance.

        Parameters
        ----------
        variable : ndarray
            Column of variables to generate the splines for
        specification : dict
            Dictionary of the processed spline specification for the corresponding variable.

        Returns
        -------
        ndarray :
            Returns the object returned by the spline (the basis matrix of the spline terms for the column)
        """
        return spline(variable=variable,                        # Pass variable to spline function
                      knots=specification["knots"],             # ... with knot locations
                      power=specification["power"],             # ... to the power
                      restricted=specification["natural"],      # ... whether to restrict
                      normalized=specification["normalized"])   # ... and whether to normalize by knot range

    def generate_default_spline_parameters(specification):
        """Internal function to process the input specification dictionary of spline parameters. Namely, ensure that
        'knots' is specified, fill in any other empty parameters, and check for additional keys given.

        Parameters
        ----------
        specification : dict
            Dictionary of the input spline specifications to check and process.

        Returns
        -------
        dict :
            Processed dictionary of the spline hyperparameters
        """
        # Setup for dict processing
        keys = specification.keys()                                   # Extract keys from the input dictionary
        defaults = {"knots": None,                                    # Default values (knots must be provided)
                    "natural": True,                                  # ... default to restricted splines
                    "power": 3,                                       # ... default to cubic splines
                    "penalty": 0,                                     # ... default to NO penalty on spline terms
                    "normalized": False}                              # ... default to NO normalized splines

        # Checking the keys in the input dictionary against the expected keys
        expected_keys = ["knots", "natural", "power", "penalty", "normalized"]  # List of keys expected to occur
        extra_keys = [param for param in keys if param not in expected_keys]
        if len(extra_keys) != 0:
            warnings.warn("The following keys were found in the specification: " + str(extra_keys)
                          + ". These keys are not supported and are being ignored.",
                          UserWarning)

        # Managing knot keyword
        if "knots" not in keys:
            raise ValueError("`knots` must be specified.")
        # Managing spline keywords with defaults
        for kw in ["natural", "power", "penalty", "normalized"]:
            if kw not in keys:
                specification[kw] = defaults[kw]

        # Returning processed spline parameters dictionary
        return specification

    # Extract meta-data from input
    X = np.asarray(X)                               # Convert to NumPy array (when user interacts directly with func)
    n_cols = X.shape[1]                             # Number of columns in the input data
    n_obs = X.shape[0]                              # Number of observations in the input data
    if isinstance(specifications, dict):            # Expanding to list if only a single dict is provided
        specifications = [specifications, ]*n_cols
    elif specifications is None:                    # When None is given as specification, ignore splines (linear reg)
        specifications = [specifications, ]*n_cols
    else:                                           # Otherwise check dimensions of specifications and cols match
        if len(specifications) != n_cols:
            raise ValueError("The number of input specifications (" + str(len(specifications)) +
                             ") and the number of columns (" + str(n_cols) + ") do not match")

    # Generate spline terms for each column
    Xt = []                                              # Placeholder storage for generated spline terms
    penalties = []                                       # Placeholder storage for corresponding penalty terms
    for col_id in range(n_cols):                         # Loop through all the columns by their index number
        xvar = X[:, col_id]                              # ... extract corresponding column by ID
        xspec = specifications[col_id]                   # ... extract specification by ID

        # Linear term
        Xt.append(xvar.reshape(n_obs, 1))                # ... always add the corresponding linear term
        penalties.append(0)                              # ... always have penalty of zero for the linear term

        # Spline generation
        if xspec is not None:                            # ... when given a specification != None, generate splines
            # Processing input spline specifications to adhere to conventions
            spec_i = generate_default_spline_parameters(specification=xspec)
            # Generate the spline matrix using the spline function and specification
            spline_matrix = generate_spline(variable=X[:, col_id],
                                            specification=spec_i)
            # Add spline terms to storage
            Xt.append(spline_matrix)                                      # ... add matrix of splines to list
            penalties += [spec_i["penalty"], ]*spline_matrix.shape[1]     # ... add penalty terms from specifications

    # Return the transformed design matrix
    if return_penalty:                      # If return penalty
        return np.hstack(Xt), penalties     # ... return additive design matrix AND list of penalty terms
    else:                                   # Otherwise
        return np.hstack(Xt)                # ... only return the additive design matrix
