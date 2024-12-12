####################################################################################################################
# Tests for differentiation procedures
####################################################################################################################

import pytest
import numpy as np
import numpy.testing as npt
import pandas as pd
import scipy as sp
from scipy.stats import logistic
from scipy.optimize import approx_fprime

from delicatessen.derivative import auto_differentiation, approx_differentiation
from delicatessen.utilities import inverse_logit, identity, polygamma, standard_normal_cdf, standard_normal_pdf
from delicatessen import MEstimator
from delicatessen.data import load_inderjit
from delicatessen.estimating_equations import (ee_mean_variance, ee_mean_robust,
                                               ee_regression, ee_glm, ee_robust_regression, ee_ridge_regression,
                                               ee_additive_regression,
                                               ee_weibull_model, ee_aft_weibull,
                                               ee_loglogistic, ee_loglogistic_ed,
                                               ee_gformula, ee_ipw, ee_ipw_msm, ee_aipw, ee_gestimation_snmm,
                                               ee_mean_sensitivity_analysis)

np.random.seed(20230704)


class TestAutoDifferentiation:

    def test_compare_elementary_operators(self):
        # Defining the functions to check
        def f(x):
            return [10,
                    10 - 5 + 32*5 - 6**2,
                    5 + x[0] + x[1] - x[2] - x[3],
                    -32 + x[0]*x[2] + x[1]*x[3],
                    x[1]**2 + x[0] + x[3] - 30,
                    -32 + x[0]**x[2] + x[1]**x[3],
                    (x[0] + x[1])**(x[2] + x[3]) + 6,
                    5*x[1]**2 + (x[2]**2)*5,
                    x[1] / 10 + (10/x[2])**2,
                    x[0]**x[1],
                    0.9**x[2],
                    (x[3] + 0.9)**(x[1] * x[0] - 0.1),
                    ]

        # Points to Evaluate at
        xinput = [0.5, 1.9, -2.3, 2]

        # Evaluating the derivatives at the points
        dx_exact = auto_differentiation(xinput, f)
        dx_approx = approx_fprime(xinput, f, epsilon=1e-9)

        # Checking
        npt.assert_allclose(dx_approx, dx_exact, atol=1e-5)

    def test_single_evaluation(self):
        def f(x):
            return -32 + 4*x - 10*x**2

        # Points to Evaluate at
        xinput = [2.754, ]

        # Evaluating the derivatives at the points
        dx_exact = auto_differentiation(xinput, f)
        dx_approx = approx_fprime(xinput, f, epsilon=1e-9)

        # Checking
        npt.assert_allclose(dx_approx, dx_exact, atol=1e-5)

    def test_compare_equality1_operators(self):
        # Defining the functions to check
        def f(x):
            return [(x[0] <= 0.1)*x[0] + x[1]**x[2],
                    (x[0] < 0.1)*x[0] + x[1]**x[2],
                    (x[0] >= 0.1) * x[0] + x[1] ** x[2],
                    (x[0] > 0.1) * x[0] + x[1] ** x[2],
                    (x[0] >= 5.0)*x[0] + x[1]**x[2],
                    (x[0] > 5.0)*x[0] + x[1]**x[2],
                    (x[0] <= 5.0)*x[0] + x[1]**x[2],
                    (x[0] < 5.0)*x[0] + x[1]**x[2],
                    (x[0] <= 5.1)*(x[0] <= 7.0)*(x[0] ** 2.5)*(x[0] + 3)**0.5 + 27*x[0]**3,
                    (x[0] < 5.1) * (x[0] + x[1] ** 2) ** 3,
                    ]

        # Points to Evaluate at
        xinput = [0.5, 1.9, -2.3, 2]

        # Evaluating the derivatives at the points
        dx_exact = auto_differentiation(xinput, f)
        dx_approx = approx_fprime(xinput, f, epsilon=1e-9)

        # Checking
        npt.assert_allclose(dx_approx, dx_exact, atol=1e-5)

    def test_compare_equality2_operators(self):
        # Defining the functions to check
        def f(x):
            return [(x[0] == 0.5) * (x[0] + x[1]**2),
                    (x[0] != 0.5) * (x[0] + x[1]**2),
                    (x[0] == 5.5) * (x[0] + x[1]**2),
                    (x[0] != 5.5) * (x[0] + x[1]**2),
                    ]

        # Points to Evaluate at
        xinput = [0.5, 1.9, -2.3, 2]

        # Truth by-hand (approx_fprime breaks here due to shifts)
        dx_true = np.array([[1., 3.8, 0., 0., ],
                            [0., 0., 0., 0., ],
                            [0., 0., 0., 0., ],
                            [1., 3.8, 0., 0., ]])

        # Evaluating the derivatives at the points
        dx_exact = auto_differentiation(xinput, f)

        # Checking
        npt.assert_allclose(dx_true, dx_exact, atol=1e-9)

    def test_compare_numpy_operators(self):
        # Defining the functions to check
        def f(x):
            return [np.sum(x),
                    np.prod(x),
                    np.dot(np.array([1, -2, 3]), x[0:3].T),
                    np.power(x[1], 0) + np.power(x[2], 2),
                    -np.power(x[0], 2) + 3*np.power(x[3], 2),
                    np.sqrt(x[1]),

                    np.abs(x[0] + x[2]),
                    np.abs(x[0] + (x[1] + x[0])**2),
                    np.sign(x[2]),
                    np.sign(x[2])*x[0] + x[1]**x[2],

                    5 + np.exp(1) + x[1] - x[2],
                    np.exp(x[0] - x[1]) + 1.5*x[2]**3,
                    5 + np.exp(x[0] + x[1]) + x[2] + 5 * x[3],

                    np.log(x[0] + x[1]) + x[3]**1.5,
                    np.log2(x[1]) + np.log10(x[3]) - x[2]**2,
                    ]

        # Points to Evaluate at
        xinput = [0.5, 1.9, -2.3, 2]

        # Evaluating the derivatives at the points
        dx_exact = auto_differentiation(xinput, f)
        dx_approx = approx_fprime(xinput, f, epsilon=1e-9)

        # Checking
        npt.assert_allclose(dx_approx, dx_exact, atol=1e-5)

    def test_compare_numpy_trig(self):
        # Defining the functions to check
        def f(x):
            return [np.sin(x[0]) + x[0]**2,
                    np.cos(x[1]**2) + x[2],
                    np.tan(x[2])**2 - 5*x[0]*x[1]**2,
                    np.arcsin(x[0]) - 2*np.arccos(0.1*x[1]) + np.arctan(0.1*x[2]) + x[3]**3,
                    np.arctan(x[0] + np.arcsin(0.5*x[1] + 0.1*x[2])),
                    np.sinh(1.1*x[0]) + np.cosh(0.5*x[1]) + np.tanh(0.9*x[2] + x[3]),
                    np.arcsinh(x[0]) - 2*np.arccosh(x[1]) + np.arctanh(0.1*x[2]) + x[3]**3,
                    ]

        # Points to Evaluate at
        xinput = [0.5, 1.9, -2.3, 2]

        # Evaluating the derivatives at the points
        dx_exact = auto_differentiation(xinput, f)
        dx_approx = approx_fprime(xinput, f, epsilon=1e-9)

        # Checking
        npt.assert_allclose(dx_approx, dx_exact, atol=1e-5)

    def test_compare_numpy_comparisons(self):
        # Defining the functions to check
        def f(x):
            return [np.where(x[0] <= 0.1, 1, 0),
                    np.where(x[0] <= 5.1, 1, 0),
                    np.where(x[0] <= 5.1, 1, 0) * x[1],
                    np.where(x[0] <= 0.1, 1, 0) * x[0],
                    np.where(x[0] <= 5.1, 1, 0) * (x[0] ** 2.5),
                    np.where(x[0] <= 5.1, x[0] ** 2.5, 0),
                    np.where(x[0] < 5.1, 1, 0)*(x[3]**1.5),
                    np.where(x[0] < 5.1, x[3] ** 1.5, 0),
                    ]

        # Points to Evaluate at
        xinput = [0.5, 1.9, -2.3, 2]

        # Evaluating the derivatives at the points
        dx_exact = auto_differentiation(xinput, f)
        dx_approx = approx_fprime(xinput, f, epsilon=1e-9)

        # Checking
        npt.assert_allclose(dx_approx, dx_exact, atol=1e-5)

    def test_compare_numpy_truncate(self):
        # Defining the functions to check
        def f(x):
            return [np.floor(x[0]),
                    1 + np.floor(x[0] + x[1]),
                    np.floor(x[3] - 0.1),
                    np.ceil(x[1]),
                    np.clip(x[0], -10, 10),
                    np.clip(x[0], 8, 10),
                    np.clip(x[0], -10, -8),
                    ]

        # Points to Evaluate at
        xinput = [0.5, 1.9, -2.3, 2]

        # Evaluating the derivatives at the points
        dx_exact = auto_differentiation(xinput, f)
        dx_approx = approx_fprime(xinput, f, epsilon=1e-9)

        # Checking
        npt.assert_allclose(dx_approx, dx_exact, atol=1e-5)

    def test_numpy_hard_comparisons(self):
        # Defining the functions to check
        def f(x):
            return [x[1] ** 2 + 1 * np.where(x[2] == -2.3, 1, 0) * (x[0] + x[3]),  # Equality causes bad approx
                    x[1] ** 2 + 1 * np.where(x[2] == -2.3, x[0] + x[3], 0),        # Equality causes bad approx
                    ]

        # Points to Evaluate at
        xinput = [0.5, 1.9, -2.3, 2]

        # True Derivatives
        dx_true = np.array([[1., 3.8, 0., 1.],
                            [1., 3.8, 0., 1.],
                            ])

        # Evaluating the derivatives at the points
        dx_exact = auto_differentiation(xinput, f)

        # Checking
        npt.assert_allclose(dx_true, dx_exact, atol=1e-5)

    def test_numpy_transpose(self):
        d = pd.DataFrame()
        d['X'] = np.log([5, 10, 15, 20, 25])
        d['Y'] = [118, 58, 5, 30, 58]
        d['I'] = 1
        y = np.asarray(d['Y'])[:, None]
        X = np.asarray(d[['I', 'X']])

        def psi(theta):
            beta, alpha = theta[:-1], np.exp(theta[-1])
            beta = np.asarray(beta)[:, None]
            pred_y = np.exp(np.dot(X, beta))
            ee_beta = ((y - pred_y) * X).T

            # This is the simplest example of the issue
            ee_alpha = (alpha * y).T
            return np.vstack([ee_beta, ee_alpha])

        def internal_sum(theta):
            return np.sum(psi(theta), axis=1)

        # Evaluating the derivatives at the points
        dx_exact = auto_differentiation([5.503, -0.6019, 0.0166], internal_sum)
        dx_approx = approx_fprime([5.503, -0.6019, 0.0166], internal_sum, epsilon=1e-9)

        # Checking
        npt.assert_allclose(dx_approx, dx_exact, atol=1e-4)

    def test_numpy_operator_lead(self):
        d = pd.DataFrame()
        d['X'] = [1, 2, 3]
        d['Y'] = [2, 4, 6]
        d['I'] = 1
        y = np.asarray(d['Y'])[:, None]
        X = np.asarray(d[['X', ]])

        def psi(theta):
            beta, alpha = theta[:-1], np.exp(theta[-1])
            beta = np.asarray(beta)[:, None]
            pred_y = np.dot(X, beta)
            ee_beta = ((y - pred_y) * X).T

            # Order of operations issues that can happen
            ee_alpha1 = ((y + alpha) * alpha).T
            ee_alpha2 = ((y - alpha) * alpha).T
            ee_alpha3 = ((y + alpha) / alpha).T
            ee_alpha4 = ((y * alpha) + alpha).T
            ee_alpha5 = ((y / alpha) + alpha).T
            ee_alpha6 = ((y ** alpha) + alpha).T

            return np.vstack([ee_beta,
                              ee_alpha1, ee_alpha2, ee_alpha3,
                              ee_alpha4, ee_alpha5, ee_alpha6
                              ])

        def internal_sum(theta):
            return np.sum(psi(theta), axis=1)

        # Evaluating the derivatives at the points
        dx_approx = approx_fprime([1.503, 0.2], internal_sum, epsilon=1e-9)
        dx_exact = auto_differentiation([1.503, 0.2], internal_sum)

        # Checking
        npt.assert_allclose(dx_approx, dx_exact, rtol=1e-6)

    def test_numpy_operator_tail(self):
        d = pd.DataFrame()
        d['Y'] = [2, -4, -3, 0]
        d['I'] = 1
        y = np.asarray(d['Y'])[:, None]

        def psi(theta):
            alpha = np.log(theta[0])
            ee_alpha1 = (alpha + (y * alpha)).T
            ee_alpha2 = (alpha - (y * alpha)).T
            ee_alpha3 = (alpha * (y + alpha)).T
            ee_alpha4 = (alpha / (y + alpha)).T
            ee_alpha5 = (alpha ** y).T
            ee_alpha6 = np.where(alpha + (alpha*y) > 0, alpha*y, alpha**2 + y).T

            return np.vstack([ee_alpha1, ee_alpha2,
                              ee_alpha3, ee_alpha4,
                              ee_alpha5, ee_alpha6])

        def internal_sum(theta):
            return np.sum(psi(theta), axis=1)

        # Evaluating the derivatives at the point less than zero
        dx_approx = approx_fprime([0.2, ], internal_sum, epsilon=1e-9)
        dx_exact = auto_differentiation([0.2, ], internal_sum)

        # Checking
        npt.assert_allclose(dx_approx, dx_exact, rtol=1e-6)

        # Evaluating the derivatives at the point greater than zero
        #   We do both, since it matters for the __pow__ check
        dx_approx = approx_fprime([1.2, ], internal_sum, epsilon=1e-9)
        dx_exact = auto_differentiation([1.2, ], internal_sum)

        # Checking
        npt.assert_allclose(dx_approx, dx_exact, rtol=1e-6)

    def test_scipy_special(self):
        def f(x):
            return [polygamma(n=1, x=x[0]),
                    polygamma(n=1, x=x[1]),
                    polygamma(n=1, x=x[2]),
                    polygamma(n=1, x=x[3]),
                    ]

        # Points to Evaluate at
        xinput = [0.5, 1.9, -2.3, 2]

        # True Derivatives
        dx_true = sp.special.polygamma(n=2, x=xinput)

        # Evaluating the derivatives at the points
        dx_exact = auto_differentiation(xinput, f)

        # Checking
        npt.assert_allclose(dx_true, np.diag(dx_exact), atol=1e-7)

    def test_scipy_special_numapprox(self):
        def f(x):
            return [polygamma(n=1, x=x[0]),
                    polygamma(n=2, x=x[1]) + x[1]**2,
                    polygamma(n=3, x=x[2]*x[3] + x[1]),
                    polygamma(n=4, x=np.log(x[3] + x[1]) + x[0]**2) - x[3],
                    standard_normal_cdf(x=x[1]),
                    standard_normal_pdf(x=x[2]),
                    ]

        # Points to Evaluate at
        xinput = [0.5, 1.9, -2.3, 2]

        # Approximate Derivatives
        dx_approx = approx_fprime(xinput, f, epsilon=1e-9)

        # Evaluating the derivatives at the points
        dx_exact = auto_differentiation(xinput, f)

        # Checking
        npt.assert_allclose(dx_approx, dx_exact, atol=1e-5)

    def test_scipy_special_arrays(self):
        d = pd.DataFrame()
        d['Y'] = [2, 4, 6]
        y = np.asarray(d['Y'])[:, None]

        def f(x):
            alpha = np.exp(x[0])
            ee_alpha1 = (y + polygamma(n=1, x=alpha)).T
            ee_alpha2 = (polygamma(n=2, x=y*alpha)).T
            stack = np.vstack([ee_alpha1,
                               ee_alpha2])
            return np.sum(stack, axis=1)

        # Points to Evaluate at
        xinput = [0.5, ]

        # Approximate Derivatives
        dx_approx = approx_fprime(xinput, f, epsilon=1e-9)

        # Evaluating the derivatives at the points
        dx_exact = auto_differentiation(xinput, f)

        # Checking
        npt.assert_allclose(dx_approx, dx_exact, atol=1e-5)

    def test_compare_deli_utilities(self):
        # Defining the functions to check
        def f(x):
            return [inverse_logit(-0.75 + 0.5*x[0] - 0.78*x[1] + 1.45*x[2] + 1.2*x[3]),
                    identity(-0.75 + 0.5*x[0] - 0.78*x[1] + 1.45*x[2]) + 1.2*x[3], ]

        # Points to Evaluate at
        xinput = [0.5, 1.9, -2.3, 3]

        # Evaluating the derivatives at the points
        dx_exact = auto_differentiation(xinput, f)
        dx_approx = approx_fprime(xinput, f, epsilon=1e-9)

        # Checking
        npt.assert_allclose(dx_approx, dx_exact, atol=1e-7)

    def test_large_gradient(self):
        # Function to evaluate (with 500 parameters)
        def f(x):
            x_init = 1
            storage = []
            for xi in x:
                y = np.sin(x_init - xi**2 - 0.01*xi)
                storage.append(y)
                x_init = xi
            return storage

        # Points to Evaluate at
        xinput = np.linspace(-10, 10, 500)

        # Evaluating the derivatives at the points
        dx_exact = auto_differentiation(xinput, f)
        dx_approx = approx_fprime(xinput, f, epsilon=1e-9)

        # Checking
        npt.assert_allclose(dx_approx, dx_exact, atol=1e-4)

    def test_undefined_derivs(self):
        # Defining the functions to check
        def f(x):
            return [np.floor(x[3]) + x[0],                 # Floor is not defined if no remain
                    np.ceil(x[3]) + x[1],                  # Ceil is not defined if no remain
                    np.abs(x[0] - 1/x[3]) + x[2],          # Absolute value not defined at 0
                    ]

        # Points to Evaluate at
        xinput = [0.5, 1.9, -2.3, 2]

        # True Derivatives
        dx_true = np.array([[np.nan, np.nan, np.nan, np.nan],
                            [np.nan, np.nan, np.nan, np.nan],
                            [np.nan, np.nan, np.nan, np.nan],
                            ])

        # Evaluating the derivatives at the points
        dx_exact = auto_differentiation(xinput, f)

        # Checking
        npt.assert_allclose(dx_true, dx_exact, atol=1e-5)


class TestApproxDifferentiation:

    def test_single_evaluation(self):
        def f(x):
            return -32 + 4*x - 10*x**2

        # Points to Evaluate at
        xinput = [2.754, ]

        # Evaluating the derivatives at the points
        dx_byhand = approx_differentiation(xinput, f, method='fapprox')
        dx_approx = approx_fprime(xinput, f, epsilon=1e-9)

        # Checking
        npt.assert_allclose(dx_approx, dx_byhand[0][0], atol=1e-5)

    def test_doc_example(self):
        # First function to check
        def f(x):
            return x ** 2 - x + np.sin(x + np.sqrt(x))

        dy_approx = approx_differentiation(xk=[1, ], f=f)
        dy_exact = auto_differentiation(xk=[1, ], f=f)
        npt.assert_allclose(dy_exact, dy_approx[0][0], atol=1e-5)

        # Second function to check
        def f(x):
            return [x[0]**2 - x[1], np.sin(np.sqrt(x[1]) + x[2]) + x[2]*(x[1]**2)]

        dy_approx = approx_differentiation(xk=[0.7, 1.2, -0.9], f=f)
        dy_exact = auto_differentiation(xk=[0.7, 1.2, -0.9], f=f)
        npt.assert_allclose(dy_exact, dy_approx, atol=1e-5)

    def test_compare_elementary_operators_f(self):
        # Defining the functions to check
        def f(x):
            return [10,
                    10 - 5 + 32*5 - 6**2,
                    5 + x[0] + x[1] - x[2] - x[3],
                    -32 + x[0]*x[2] + x[1]*x[3],
                    x[1]**2 + x[0] + x[3] - 30,
                    -32 + x[0]**x[2] + x[1]**x[3],
                    (x[0] + x[1])**(x[2] + x[3]) + 6,
                    5*x[1]**2 + (x[2]**2)*5,
                    x[1] / 10 + (10/x[2])**2,
                    x[0]**x[1],
                    0.9**x[2],
                    (x[3] + 0.9)**(x[1] * x[0] - 0.1),
                    ]

        # Points to Evaluate at
        xinput = [0.5, 1.9, -2.3, 2]

        # Evaluating the derivatives at the points
        dx_byhand = approx_differentiation(xinput, f, epsilon=1e-9, method='fapprox')
        dx_approx = approx_fprime(xinput, f, epsilon=1e-9)

        # Checking
        npt.assert_allclose(dx_approx, dx_byhand, atol=1e-5)

    def test_compare_elementary_operators_b(self):
        # Defining the functions to check
        def f(x):
            return [10,
                    10 - 5 + 32*5 - 6**2,
                    5 + x[0] + x[1] - x[2] - x[3],
                    -32 + x[0]*x[2] + x[1]*x[3],
                    x[1]**2 + x[0] + x[3] - 30,
                    -32 + x[0]**x[2] + x[1]**x[3],
                    (x[0] + x[1])**(x[2] + x[3]) + 6,
                    5*x[1]**2 + (x[2]**2)*5,
                    x[1] / 10 + (10/x[2])**2,
                    x[0]**x[1],
                    0.9**x[2],
                    (x[3] + 0.9)**(x[1] * x[0] - 0.1),
                    ]

        # Points to Evaluate at
        xinput = [0.5, 1.9, -2.3, 2]

        # Evaluating the derivatives at the points
        dx_byhand = approx_differentiation(xinput, f, epsilon=1e-9, method='bapprox')
        dx_approx = approx_fprime(xinput, f, epsilon=1e-9)

        # Checking
        npt.assert_allclose(dx_approx, dx_byhand, atol=1e-5)

    def test_compare_elementary_operators_c(self):
        # Defining the functions to check
        def f(x):
            return [10,
                    10 - 5 + 32*5 - 6**2,
                    5 + x[0] + x[1] - x[2] - x[3],
                    -32 + x[0]*x[2] + x[1]*x[3],
                    x[1]**2 + x[0] + x[3] - 30,
                    -32 + x[0]**x[2] + x[1]**x[3],
                    (x[0] + x[1])**(x[2] + x[3]) + 6,
                    5*x[1]**2 + (x[2]**2)*5,
                    x[1] / 10 + (10/x[2])**2,
                    x[0]**x[1],
                    0.9**x[2],
                    (x[3] + 0.9)**(x[1] * x[0] - 0.1),
                    ]

        # Points to Evaluate at
        xinput = [0.5, 1.9, -2.3, 2]

        # Evaluating the derivatives at the points
        dx_byhand = approx_differentiation(xinput, f, epsilon=1e-9, method='capprox')
        dx_approx = approx_fprime(xinput, f, epsilon=1e-9)

        # Checking
        npt.assert_allclose(dx_approx, dx_byhand, atol=1e-5)

    def test_compare_equality1_operators(self):
        # Defining the functions to check
        def f(x):
            return [(x[0] <= 0.1)*x[0] + x[1]**x[2],
                    (x[0] < 0.1)*x[0] + x[1]**x[2],
                    (x[0] >= 0.1) * x[0] + x[1] ** x[2],
                    (x[0] > 0.1) * x[0] + x[1] ** x[2],
                    (x[0] >= 5.0)*x[0] + x[1]**x[2],
                    (x[0] > 5.0)*x[0] + x[1]**x[2],
                    (x[0] <= 5.0)*x[0] + x[1]**x[2],
                    (x[0] < 5.0)*x[0] + x[1]**x[2],
                    (x[0] <= 5.1)*(x[0] <= 7.0)*(x[0] ** 2.5)*(x[0] + 3)**0.5 + 27*x[0]**3,
                    (x[0] < 5.1) * (x[0] + x[1] ** 2) ** 3,
                    ]

        # Points to Evaluate at
        xinput = [0.5, 1.9, -2.3, 2]

        # Evaluating the derivatives at the points
        dx_byhand = approx_differentiation(xinput, f, method='fapprox')
        dx_approx = approx_fprime(xinput, f, epsilon=1e-9)

        # Checking
        npt.assert_allclose(dx_approx, dx_byhand, atol=1e-5)

    def test_scipy_special_numapprox(self):
        def f(x):
            return [polygamma(n=1, x=x[0]),
                    polygamma(n=2, x=x[1]) + x[1]**2,
                    polygamma(n=3, x=x[2]*x[3] + x[1]),
                    polygamma(n=4, x=np.log(x[3] + x[1]) + x[0]**2) - x[3],
                    standard_normal_cdf(x=x[1]),
                    standard_normal_pdf(x=x[2]),
                    ]

        # Points to Evaluate at
        xinput = [0.5, 1.9, -2.3, 2]

        # Approximate Derivatives
        dx_approx = approx_fprime(xinput, f, epsilon=1e-9)

        # Evaluating the derivatives at the points
        dx_byhand = approx_differentiation(xinput, f, method='fapprox')

        # Checking
        npt.assert_allclose(dx_approx, dx_byhand, atol=1e-5)


class TestSandwichAutoDiff:

    # Basics

    def test_exact_bread_mean(self):
        # Data set
        y = np.array([5, 1, 2, 4, 2, 4, 5, 7, 11, 1, 6, 3, 4, 6])

        def psi(theta):
            return y - theta

        mestr = MEstimator(psi, init=[0, ])
        mestr.estimate(deriv_method='exact')

        # Checking bread estimates
        npt.assert_allclose(mestr.bread,
                            [[1]],
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mestr.asymptotic_variance,
                            np.var(y, ddof=0),
                            atol=1e-7)

    def test_exact_bread_mean_var(self):
        # Data set
        y = np.array([5, 1, 2, 4, 2, 4, 5, 7, 11, 1, 6, 3, 4, 6])

        def psi(theta):
            return ee_mean_variance(theta=theta, y=y)

        mestr = MEstimator(psi, init=[0, 1, ])
        mestr.estimate(deriv_method='exact')
        bread_exact = mestr.bread
        var_exact = mestr.variance
        mestr.estimate(deriv_method='approx')
        bread_approx = mestr.bread
        var_approx = mestr.variance

        # Checking bread estimates
        npt.assert_allclose(bread_approx, bread_exact, atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(var_approx, var_exact, atol=1e-5)

    def test_exact_bread_robust_mean(self):
        n = 500
        y = np.random.standard_cauchy(size=n)

        # Huber
        def psi(theta):
            return ee_mean_robust(theta,
                                  y=y,
                                  k=3, loss='huber')

        mestr = MEstimator(psi, init=[0.])

        # Auto-differentation
        mestr.estimate(deriv_method='exact')
        bread_exact = mestr.bread
        var_exact = mestr.variance

        # Central difference method
        mestr.estimate(deriv_method='approx')
        bread_approx = mestr.bread
        var_approx = mestr.variance

        # Checking bread estimates
        npt.assert_allclose(bread_approx, bread_exact, atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(var_approx, var_exact, atol=1e-6)

        # Andrew
        def psi(theta):
            return ee_mean_robust(theta,
                                  y=y,
                                  k=1.339, loss='andrew')

        mestr = MEstimator(psi, init=[0.])

        # Auto-differentation
        mestr.estimate(deriv_method='exact')
        bread_exact = mestr.bread
        var_exact = mestr.variance

        # Central difference method
        mestr.estimate(deriv_method='approx')
        bread_approx = mestr.bread
        var_approx = mestr.variance

        # Checking bread estimates
        npt.assert_allclose(bread_approx, bread_exact, atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(var_approx, var_exact, atol=1e-6)

        # Tukey
        def psi(theta):
            return ee_mean_robust(theta,
                                  y=y,
                                  k=4.685, loss='tukey')

        mestr = MEstimator(psi, init=[0.])

        # Auto-differentation
        mestr.estimate(deriv_method='exact')
        bread_exact = mestr.bread
        var_exact = mestr.variance

        # Central difference method
        mestr.estimate(deriv_method='approx')
        bread_approx = mestr.bread
        var_approx = mestr.variance

        # Checking bread estimates
        npt.assert_allclose(bread_approx, bread_exact, atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(var_approx, var_exact, atol=1e-6)

        # Hampel
        def psi(theta):
            return ee_mean_robust(theta,
                                  y=y,
                                  k=8, loss='hampel', lower=2, upper=4)

        mestr = MEstimator(psi, init=[0.])

        # Auto-differentation
        mestr.estimate(deriv_method='exact')
        bread_exact = mestr.bread
        var_exact = mestr.variance

        # Central difference method
        mestr.estimate(deriv_method='approx')
        bread_approx = mestr.bread
        var_approx = mestr.variance

        # Checking bread estimates
        npt.assert_allclose(bread_approx, bread_exact, atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(var_approx, var_exact, atol=1e-6)

    # Regression

    def test_exact_bread_linear_reg(self):
        n = 500
        data = pd.DataFrame()
        data['X'] = np.random.normal(size=n)
        data['Z'] = np.random.normal(size=n)
        data['Y'] = 0.5 + 2*data['X'] - 1*data['Z'] + np.random.normal(size=n)
        data['C'] = 1
        data['w'] = np.random.uniform(1, 10, size=n)

        def psi(theta):
            return ee_regression(theta,
                                 X=data[['C', 'X', 'Z']], y=data['Y'],
                                 model='linear', weights=data['w'])

        mestr = MEstimator(psi, init=[0., 2., -1.])

        # Auto-differentation
        mestr.estimate(solver='lm', deriv_method='exact')
        bread_exact = mestr.bread
        var_exact = mestr.variance

        # Central difference method
        mestr.estimate(solver='lm', deriv_method='approx')
        bread_approx = mestr.bread
        var_approx = mestr.variance

        # Checking bread estimates
        npt.assert_allclose(bread_approx, bread_exact, atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(var_approx, var_exact, atol=1e-6)

    def test_exact_bread_logit_reg(self):
        n = 500
        data = pd.DataFrame()
        data['X'] = np.random.normal(size=n)
        data['Z'] = np.random.normal(size=n)
        data['Y'] = np.random.binomial(n=1, p=logistic.cdf(0.5 + 2*data['X'] - 1*data['Z']), size=n)
        data['C'] = 1
        data['w'] = np.random.uniform(1, 10, size=n)

        def psi(theta):
            return ee_regression(theta,
                                 X=data[['C', 'X', 'Z']], y=data['Y'],
                                 model='logistic', weights=data['w'])

        mestr = MEstimator(psi, init=[0., 2., -1.])

        # Auto-differentation
        mestr.estimate(solver='lm', deriv_method='exact')
        bread_exact = mestr.bread
        var_exact = mestr.variance

        # Central difference method
        mestr.estimate(solver='lm', deriv_method='approx')
        bread_approx = mestr.bread
        var_approx = mestr.variance

        # Checking bread estimates
        npt.assert_allclose(bread_approx, bread_exact, atol=1e-7)

        # Checking variance estimates
        npt.assert_allclose(var_approx, var_exact, atol=1e-7)

    def test_exact_bread_poisson_reg(self):
        n = 500
        data = pd.DataFrame()
        data['X'] = np.random.normal(size=n)
        data['Z'] = np.random.normal(size=n)
        data['Y'] = np.random.poisson(lam=np.exp(0.5 + 2*data['X'] - 1*data['Z']), size=n)
        data['C'] = 1
        data['w'] = np.random.uniform(1, 10, size=n)

        def psi(theta):
            return ee_regression(theta,
                                 X=data[['C', 'X', 'Z']], y=data['Y'],
                                 model='poisson', weights=data['w'])

        mestr = MEstimator(psi, init=[0., 2., -1.])

        # Auto-differentation
        mestr.estimate(solver='lm', deriv_method='exact')
        bread_exact = mestr.bread
        var_exact = mestr.variance

        # Central difference method
        mestr.estimate(solver='lm', deriv_method='approx')
        bread_approx = mestr.bread
        var_approx = mestr.variance

        # Checking bread estimates
        npt.assert_allclose(bread_approx, bread_exact, atol=1e-3)

        # Checking variance estimates
        npt.assert_allclose(var_approx, var_exact, atol=1e-7)

    def test_exact_bread_glm_normal(self):
        d = pd.DataFrame()
        d['X'] = [1, -1, 0, 1, 2, 1, -2, -1, 0, 3, -3, 1, 1, -1, -1, -2, 2, 0, -1, 0]
        d['Z'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        d['Y'] = [1, 5, 1, 9, -1, 4, 3, 3, 1, -2, 4, -2, 3, 6, 6, 8, 7, 1, -2, 5]
        d['I'] = 1

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                          distribution='normal', link='identity')

        mestr = MEstimator(psi, init=[0., 0., 0.])

        # Auto-differentation
        mestr.estimate(solver='lm', deriv_method='exact')
        bread_exact = mestr.bread
        var_exact = mestr.variance

        # Central difference method
        mestr.estimate(solver='lm', deriv_method='approx')
        bread_approx = mestr.bread
        var_approx = mestr.variance

        # Checking bread estimates
        npt.assert_allclose(bread_approx, bread_exact, atol=1e-5)

        # Checking variance estimates
        npt.assert_allclose(var_approx, var_exact, atol=1e-6)

    def test_exact_bread_glm_linbin(self):
        d = pd.DataFrame()
        d['X'] = [1, -1, 0, 1, 2, 1, -2, -1, 0, 3, -3, 1, 1, -1, -1, -2, 2, 0, -1, 0]
        d['Z'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        d['Y'] = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0]
        d['I'] = 1

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                          link='identity', distribution='binomial')

        mestr = MEstimator(psi, init=[0.2, 0., 0.])

        # Auto-differentation
        mestr.estimate(solver='lm', deriv_method='exact')
        bread_exact = mestr.bread
        var_exact = mestr.variance

        # Central difference method
        mestr.estimate(solver='lm', deriv_method='approx')
        bread_approx = mestr.bread
        var_approx = mestr.variance

        # Checking bread estimates
        npt.assert_allclose(bread_approx, bread_exact, atol=1e-5)

        # Checking variance estimates
        npt.assert_allclose(var_approx, var_exact, atol=1e-7)

    def test_exact_bread_glm_loglog(self):
        d = pd.DataFrame()
        d['X'] = [1, -1, 0, 1, 2, 1, -2, -1, 0, 3, -3, 1, 1, -1, -1, -2, 2, 0, -1, 0]
        d['Z'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        d['Y'] = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0]
        d['I'] = 1

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                          link='loglog', distribution='binomial')

        mestr = MEstimator(psi, init=[0., 0., 0.])

        # Auto-differentation
        mestr.estimate(solver='lm', deriv_method='exact')
        bread_exact = mestr.bread
        var_exact = mestr.variance

        # Central difference method
        mestr.estimate(solver='lm', deriv_method='approx')
        bread_approx = mestr.bread
        var_approx = mestr.variance

        # Checking bread estimates
        npt.assert_allclose(bread_approx, bread_exact, atol=1e-5)

        # Checking variance estimates
        npt.assert_allclose(var_approx, var_exact, atol=1e-6)

    def test_exact_bread_glm_probit(self):
        d = pd.DataFrame()
        d['X'] = [1, -1, 0, 1, 2, 1, -2, -1, 0, 3, -3, 1, 1, -1, -1, -2, 2, 0, -1, 0]
        d['Z'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        d['Y'] = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0]
        d['I'] = 1

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                          link='probit', distribution='binomial')

        mestr = MEstimator(psi, init=[0., 0., 0.])

        # Auto-differentation
        mestr.estimate(solver='lm', deriv_method='exact')
        bread_exact = mestr.bread
        var_exact = mestr.variance

        # Central difference method
        mestr.estimate(solver='lm', deriv_method='approx')
        bread_approx = mestr.bread
        var_approx = mestr.variance

        # Checking bread estimates
        npt.assert_allclose(bread_approx, bread_exact, atol=1e-7)

        # Checking variance estimates
        npt.assert_allclose(var_approx, var_exact, atol=1e-7)

    def test_exact_bread_glm_cauchy(self):
        d = pd.DataFrame()
        d['X'] = [1, -1, 0, 1, 2, 1, -2, -1, 0, 3, -3, 1, 1, -1, -1, -2, 2, 0, -1, 0]
        d['Z'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        d['Y'] = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0]
        d['I'] = 1

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                          link='cauchy', distribution='binomial')

        mestr = MEstimator(psi, init=[0., 0., 0.])

        # Auto-differentation
        mestr.estimate(solver='lm', deriv_method='exact')
        bread_exact = mestr.bread
        var_exact = mestr.variance

        # Central difference method
        mestr.estimate(solver='lm', deriv_method='approx')
        bread_approx = mestr.bread
        var_approx = mestr.variance

        # Checking bread estimates
        npt.assert_allclose(bread_approx, bread_exact, atol=1e-7)

        # Checking variance estimates
        npt.assert_allclose(var_approx, var_exact, atol=1e-6)

    def test_exact_bread_glm_poisson(self):
        d = pd.DataFrame()
        d['X'] = [1, -1, 0, 1, 2, 1, -2, -1, 0, 3, -3, 1, 1, -1, -1, -2, 2, 0, -1, 0]
        d['Z'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        d['Y'] = [1, 5, 1, 9, 1, 4, 3, 3, 1, 2, 4, 2, 3, 6, 6, 8, 7, 1, 2, 5]
        d['I'] = 1

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                          distribution='poisson', link='sqrt')

        mestr = MEstimator(psi, init=[2., 0., 0.])

        # Auto-differentation
        mestr.estimate(solver='lm', deriv_method='exact')
        bread_exact = mestr.bread
        var_exact = mestr.variance

        # Central difference method
        mestr.estimate(solver='lm', deriv_method='approx')
        bread_approx = mestr.bread
        var_approx = mestr.variance

        # Checking bread estimates
        npt.assert_allclose(bread_approx, bread_exact, atol=1e-5)

        # Checking variance estimates
        npt.assert_allclose(var_approx, var_exact, atol=1e-6)

    def test_exact_bread_glm_invnormal(self):
        d = pd.DataFrame()
        d['X'] = [1, -1, 0, 1, 2, 1, -2, -1, 0, 3, -3, 1, 1, -1, -1, -2, 2, 0, -1, 0]
        d['Z'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        d['Y'] = [1, 5, 1, 9, 1, 4, 3, 3, 1, 2, 4, 2, 3, 6, 6, 8, 7, 1, 2, 5]
        d['I'] = 1

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                          distribution='inverse_normal', link='identity')

        mestr = MEstimator(psi, init=[2., 0., 0.])

        # Auto-differentation
        mestr.estimate(solver='lm', deriv_method='exact')
        bread_exact = mestr.bread
        var_exact = mestr.variance

        # Central difference method
        mestr.estimate(solver='lm', deriv_method='approx')
        bread_approx = mestr.bread
        var_approx = mestr.variance

        # Checking bread estimates
        npt.assert_allclose(bread_approx, bread_exact, atol=1e-7)

        # Checking variance estimates
        npt.assert_allclose(var_approx, var_exact, atol=1e-6)

    def test_exact_bread_glm_tweedie(self):
        d = pd.DataFrame()
        d['X'] = [1, -1, 0, 1, 2, 1, -2, -1, 0, 3, -3, 1, 1, -1, -1, -2, 2, 0, -1, 0]
        d['Z'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        d['Y'] = [1, 5, 1, 9, 1, 4, 3, 3, 1, 2, 4, 2, 3, 6, 6, 8, 7, 1, 2, 5]
        d['I'] = 1

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                          distribution='tweedie', link='log',
                          hyperparameter=1.5)

        mestr = MEstimator(psi, init=[2., 0., 0.])

        # Auto-differentation
        mestr.estimate(solver='lm', deriv_method='exact')
        bread_exact = mestr.bread
        var_exact = mestr.variance

        # Central difference method
        mestr.estimate(solver='lm', deriv_method='approx')
        bread_approx = mestr.bread
        var_approx = mestr.variance

        # Checking bread estimates
        npt.assert_allclose(bread_approx, bread_exact, atol=1e-7)

        # Checking variance estimates
        npt.assert_allclose(var_approx, var_exact, atol=1e-7)

    def test_exact_bread_glm_loggamma(self):
        d = pd.DataFrame()
        d['X'] = np.log([5, 10, 15, 20, 30, 40, 60, 80, 100])
        d['Y'] = [118, 58, 42, 35, 27, 25, 21, 19, 18]
        d['I'] = 1

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X']], y=d['Y'],
                          distribution='gamma', link='log')

        # Auto-differentation
        mestr = MEstimator(psi, init=[0., 0., 1.])
        mestr.estimate(solver='lm', deriv_method='exact')
        bread_exact = mestr.bread
        var_exact = mestr.variance

        # Central difference method
        mestr = MEstimator(psi, init=[0., 0., 1.])
        mestr.estimate(solver='lm', deriv_method='approx')
        bread_approx = mestr.bread
        var_approx = mestr.variance

        # Checking bread estimates
        npt.assert_allclose(bread_approx, bread_exact, atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(var_approx, var_exact, atol=5e-5)

    def test_exact_bread_glm_lognb(self):
        d = pd.DataFrame()
        d['X'] = [1, -1, 0, 1, 2, 1, -2, -1, 0, 3, -3, 1, 1, -1, -1, -2, 2, 0, -1, 0]
        d['Z'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        d['Y'] = [0, 0, 0, 0, 0, 15, 15, 25, 25, 45, 0, 0, 0, 0, 15, 15, 15, 25, 25, 35]
        d['I'] = 1

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X']], y=d['Y'],
                          distribution='nb', link='log')

        # Auto-differentation
        mestr = MEstimator(psi, init=[0., 0., 1.])
        mestr.estimate(solver='lm', deriv_method='exact')
        bread_exact = mestr.bread
        var_exact = mestr.variance

        # Central difference method
        mestr = MEstimator(psi, init=[0., 0., 1.])
        mestr.estimate(solver='lm', deriv_method='approx')
        bread_approx = mestr.bread
        var_approx = mestr.variance

        # Checking bread estimates
        npt.assert_allclose(bread_approx, bread_exact, atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(var_approx, var_exact, atol=1e-5)

    def test_exact_bread_robust_regression(self):
        d = pd.DataFrame()
        d['X'] = [1, -1, 0, 1, 2, 1, -2, -1, 0, 3, -3, 1, 1, -1, -1, -2, 2, 0, -1, 0]
        d['Z'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        d['Y'] = [1, 5, 1, 25, -1, 4, 3, 3, 1, -2, 4, -2, 3, 6, 6, 8, 7, 1, -20, 5]
        d['I'] = 1

        def psi(theta):
            return ee_robust_regression(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                                        model='linear', k=5)

        mestr = MEstimator(psi, init=[0., 0., 0.])

        # Auto-differentation
        mestr.estimate(solver='lm', deriv_method='exact')
        bread_exact = mestr.bread
        var_exact = mestr.variance

        # Central difference method
        mestr.estimate(solver='lm', deriv_method='approx')
        bread_approx = mestr.bread
        var_approx = mestr.variance

        # Checking bread estimates
        npt.assert_allclose(bread_approx, bread_exact, atol=1e-5)

        # Checking variance estimates
        npt.assert_allclose(var_approx, var_exact, atol=1e-6)

    def test_exact_bread_linear_ridge(self):
        n = 500
        data = pd.DataFrame()
        data['X'] = np.random.normal(size=n)
        data['Z'] = np.random.normal(size=n)
        data['Y'] = 0.5 + 2*data['X'] - 1*data['Z'] + np.random.normal(size=n)
        data['C'] = 1
        data['w'] = np.random.uniform(1, 10, size=n)

        def psi(theta):
            return ee_ridge_regression(theta,
                                       X=data[['C', 'X', 'Z']], y=data['Y'],
                                       penalty=[0, 1, 2],
                                       model='linear', weights=data['w'],
                                       center=[0, 1, -2])

        mestr = MEstimator(psi, init=[0., 2., -1.])

        # Auto-differentation
        mestr.estimate(solver='lm', deriv_method='exact')
        bread_exact = mestr.bread
        var_exact = mestr.variance

        # Central difference method
        mestr.estimate(solver='lm', deriv_method='approx')
        bread_approx = mestr.bread
        var_approx = mestr.variance

        # Checking bread estimates
        npt.assert_allclose(bread_approx,
                            bread_exact,
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(var_approx,
                            var_exact,
                            atol=1e-6)

    def test_exact_bread_logit_ridge(self):
        n = 500
        data = pd.DataFrame()
        data['X'] = np.random.normal(size=n)
        data['Z'] = np.random.normal(size=n)
        data['Y'] = np.random.binomial(n=1, p=logistic.cdf(0.5 + 2*data['X'] - 1*data['Z']), size=n)
        data['C'] = 1
        data['w'] = np.random.uniform(1, 10, size=n)

        def psi(theta):
            return ee_ridge_regression(theta,
                                       X=data[['C', 'X', 'Z']], y=data['Y'],
                                       penalty=[0, 1, 2],
                                       model='logistic', weights=data['w'])

        mestr = MEstimator(psi, init=[0., 2., -1.])

        # Auto-differentation
        mestr.estimate(solver='lm', deriv_method='exact')
        bread_exact = mestr.bread
        var_exact = mestr.variance

        # Central difference method
        mestr.estimate(solver='lm', deriv_method='approx')
        bread_approx = mestr.bread
        var_approx = mestr.variance

        # Checking bread estimates
        npt.assert_allclose(bread_approx,
                            bread_exact,
                            atol=1e-7)

        # Checking variance estimates
        npt.assert_allclose(var_approx,
                            var_exact,
                            atol=1e-7)

    def test_exact_bread_logit_gam(self):
        n = 1000
        data = pd.DataFrame()
        data['X'] = np.random.normal(size=n)
        data['Z'] = np.random.normal(size=n)
        data['Y'] = np.random.binomial(n=1, p=logistic.cdf(0.5 + 2 * data['X'] - 0.001*data['X']**2 - 1 * data['Z']),
                                       size=n)
        data['C'] = 1
        Xvals = np.asarray(data[['C', 'X', 'Z']])
        yvals = np.asarray(data['Y'])
        spec = [None, {"knots": [-1, 0, 1], "penalty": 3}, {"knots": [-2, -1, 0, 1, 2], "penalty": 5}]

        def psi_regression(theta):
            return ee_additive_regression(theta,
                                          X=Xvals, y=yvals,
                                          model='logistic',
                                          specifications=spec)

        # Auto-differentation
        mestr = MEstimator(psi_regression, init=[0., 2., 0., 0., 1., 0., 0., 0., 0.])
        mestr.estimate(solver='lm', deriv_method='exact')
        bread_exact = mestr.bread
        var_exact = mestr.variance

        # Central difference method
        mestr = MEstimator(psi_regression, init=[0., 2., 0., 0., 1., 0., 0., 0., 0.])
        mestr.estimate(solver='lm', deriv_method='approx')
        bread_approx = mestr.bread
        var_approx = mestr.variance

        # Checking bread estimates
        npt.assert_allclose(bread_approx,
                            bread_exact,
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(var_approx,
                            var_exact,
                            atol=1e-5)

    # Survival

    def test_exact_bread_weibull(self):
        times = np.array([1, 2, 3, 5, 2, 3, 4, 3, 1, 4])
        events = np.array([1, 0, 0, 0, 1, 1, 1, 0, 0, 1])

        def psi(theta):
            return ee_weibull_model(theta=theta,
                                    t=times, delta=events)

        mestr = MEstimator(psi, init=[1., 1.])

        # Auto-differentation
        mestr.estimate(solver='lm', deriv_method='exact')
        bread_exact = mestr.bread
        var_exact = mestr.variance

        # Central difference method
        mestr.estimate(solver='lm', deriv_method='approx')
        bread_approx = mestr.bread
        var_approx = mestr.variance

        # Checking bread estimates
        npt.assert_allclose(bread_approx,
                            bread_exact,
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(var_approx,
                            var_exact,
                            atol=1e-5)

    def test_exact_bread_weibull_aft(self):
        d = pd.DataFrame()
        d['t'] = [1, 2, 3, 5, 2, 3, 4, 3, 1, 4]
        d['d'] = [1, 0, 0, 0, 1, 1, 1, 0, 0, 1]
        d['X'] = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        d['I'] = 1

        def psi(theta):
            return ee_aft_weibull(theta=theta,
                                  t=d['t'], delta=d['d'], X=d[['X', ]])

        mestr = MEstimator(psi, init=[1.387, 0.107, 0.911])

        # Auto-differentation
        mestr.estimate(solver='hybr', deriv_method='exact')
        bread_exact = mestr.bread
        var_exact = mestr.variance

        # Central difference method
        mestr.estimate(solver='hybr', deriv_method='approx')
        bread_approx = mestr.bread
        var_approx = mestr.variance

        # Checking bread estimates
        npt.assert_allclose(bread_approx,
                            bread_exact,
                            atol=1e-5)

        # Checking variance estimates
        npt.assert_allclose(var_approx,
                            var_exact,
                            atol=1e-5)

    # Dose-Response

    def test_exact_doseresp_4plogit(self):
        d = load_inderjit()
        dose_data = d[:, 1] + 1e-6
        resp_data = d[:, 0]

        def psi(theta):
            pl4 = ee_loglogistic(theta=theta, dose=dose_data, response=resp_data)
            ed20 = ee_loglogistic_ed(theta[4], dose=dose_data, delta=0.20,
                                     steepness=theta[2], ed50=theta[1],
                                     lower=theta[0], upper=theta[3])

            # Returning stacked estimating equations
            return np.vstack([pl4, ed20])

        mestr = MEstimator(psi, init=[0, 10, 3, 2, 1])

        # Central difference method
        mestr.estimate(solver='lm', deriv_method='approx')
        bread_approx = mestr.bread
        var_approx = mestr.variance

        # Auto-differentation
        mestr.estimate(solver='lm', deriv_method='exact')
        bread_exact = mestr.bread
        var_exact = mestr.variance

        # Checking bread estimates
        npt.assert_allclose(bread_approx,
                            bread_exact,
                            atol=1e-5)

        # Checking variance estimates
        npt.assert_allclose(var_approx,
                            var_exact,
                            atol=1e-5)

    # Causal

    def test_exact_gcomputation(self):
        d = pd.DataFrame()
        d['W'] = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3,
                  1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3,
                  1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
        d['V'] = [1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0,
                  1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0,
                  1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0]
        d['A'] = [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1,
                  1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1,
                  1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1]
        d['Y'] = [3, 5, 1, 5, 2, 5, 2, 1, 4, 2, 3, 4, 2, 5, 5,
                  3, 5, 1, 5, 2, 5, 2, 1, 4, 2, 3, 4, 2, 5, 5,
                  3, 5, 1, 5, 2, 5, 2, 1, 4, 2, 3, 4, 2, 5, 5]
        d['I'] = 1
        d['A1'] = 1
        d['A0'] = 0

        # M-estimator
        def psi(theta):
            return ee_gformula(theta=theta,
                               y=d['Y'],
                               X=d[['I', 'A', 'V', 'W']],
                               X1=d[['I', 'A1', 'V', 'W']],
                               X0=d[['I', 'A0', 'V', 'W']])

        mestr = MEstimator(psi, init=[0., ] * 7)

        # Auto-differentation
        mestr.estimate(solver='lm', deriv_method='exact')
        bread_exact = mestr.bread
        var_exact = mestr.variance

        # Central difference method
        mestr.estimate(solver='lm', deriv_method='approx')
        bread_approx = mestr.bread
        var_approx = mestr.variance

        # Checking bread estimates
        npt.assert_allclose(bread_approx,
                            bread_exact,
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(var_approx,
                            var_exact,
                            atol=5e-5)

    def test_exact_ipw(self):
        d = pd.DataFrame()
        d['W'] = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3,
                  1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3,
                  1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
        d['V'] = [1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0,
                  1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0,
                  1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0]
        d['A'] = [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1,
                  1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1,
                  1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1]
        d['Y'] = [3, 5, 1, 5, 2, 5, 2, 1, 4, 2, 3, 4, 2, 5, 5,
                  3, 5, 1, 5, 2, 5, 2, 1, 4, 2, 3, 4, 2, 5, 5,
                  3, 5, 1, 5, 2, 5, 2, 1, 4, 2, 3, 4, 2, 5, 5]
        d['I'] = 1

        # M-estimator
        def psi(theta):
            return ee_ipw(theta=theta, y=d['Y'], A=d['A'],
                          W=d[['I', 'V', 'W']])

        mestr = MEstimator(psi, init=[0., ] * 6)

        # Auto-differentation
        mestr.estimate(solver='lm', deriv_method='exact')
        bread_exact = mestr.bread
        var_exact = mestr.variance

        # Central difference method
        mestr.estimate(solver='lm', deriv_method='approx')
        bread_approx = mestr.bread
        var_approx = mestr.variance

        # Checking bread estimates
        npt.assert_allclose(bread_approx,
                            bread_exact,
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(var_approx,
                            var_exact,
                            atol=5e-5)

    def test_exact_ipw_msm(self):
        d = pd.DataFrame()
        d['W'] = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3,
                  1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
        d['V'] = [1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0,
                  1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0]
        d['A'] = [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1,
                  1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1]
        d['Y'] = [3, 5, 1, 5, 2, 5, 2, 1, 4, 2, 3, 4, 2, 5, 5, 3, 5, 1, 5, 2, 5, 2, 1, 4, 2, 3, 4, 2, 5, 5,
                  ] + [np.nan, ]*15
        d['I'] = 1

        # Setting up data
        W = d[['I', 'V', 'W']]
        X = d[['I', 'A']]
        msm = d[['I', 'A']]
        a = d['A']
        y = d['Y'].fillna(-1)
        r = np.where(d['Y'].isna(), 0, 1)

        # M-estimation
        def psi(theta):
            # Separating parameters out
            alpha = theta[:2 + W.shape[1]]  # MSM & PS
            gamma = theta[2 + W.shape[1]:]  # Missing score

            # Estimating equation for IPMW
            ee_ms = ee_regression(theta=gamma, X=X, y=r, model='logistic')
            pi_m = inverse_logit(np.dot(X, gamma))
            ipmw = r / pi_m

            # Estimating equations for MSM and PS
            ee_msm = ee_ipw_msm(alpha, y=y, A=a, W=W, V=msm,
                                link='log', distribution='poisson', weights=ipmw)
            ee_msm = ee_msm * r
            return np.vstack([ee_msm, ee_ms])

        init_vals = [0., 0., ] + [0., 0., 0.] + [0., 0.]
        mestr = MEstimator(psi, init=init_vals)

        # Auto-differentation
        mestr.estimate(solver='lm', deriv_method='exact')
        bread_exact = mestr.bread
        var_exact = mestr.variance

        # Central difference method
        mestr.estimate(solver='lm', deriv_method='approx')
        bread_approx = mestr.bread
        var_approx = mestr.variance

        # Checking bread estimates
        npt.assert_allclose(bread_approx,
                            bread_exact,
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(var_approx,
                            var_exact,
                            atol=1e-5)

    def test_exact_aipw(self):
        d = pd.DataFrame()
        d['W'] = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3,
                  1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3,
                  1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
        d['V'] = [1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0,
                  1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0,
                  1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0]
        d['A'] = [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1,
                  1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1,
                  1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1]
        d['Y'] = [3, 5, 1, 5, 2, 5, 2, 1, 4, 2, 3, 4, 2, 5, 5,
                  3, 5, 1, 5, 2, 5, 2, 1, 4, 2, 3, 4, 2, 5, 5,
                  3, 5, 1, 5, 2, 5, 2, 1, 4, 2, 3, 4, 2, 5, 5]
        d['I'] = 1
        d['A1'] = 1
        d['A0'] = 0

        def psi(theta):
            return ee_aipw(theta, y=d['Y'], A=d['A'],
                           W=d[['I', 'W']],
                           X=d[['I', 'A', 'W']],
                           X1=d[['I', 'A1', 'W']],
                           X0=d[['I', 'A0', 'W']])

        mestr = MEstimator(psi, init=[0., ]*8)

        # Auto-differentation
        mestr.estimate(solver='lm', deriv_method='exact')
        bread_exact = mestr.bread
        var_exact = mestr.variance

        # Central difference method
        mestr.estimate(solver='lm', deriv_method='approx')
        bread_approx = mestr.bread
        var_approx = mestr.variance

        # Checking bread estimates
        npt.assert_allclose(bread_approx,
                            bread_exact,
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(var_approx,
                            var_exact,
                            atol=1e-5)

    def test_exact_gestimation_snmm(self):
        d = pd.DataFrame()
        d['W'] = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3,
                  1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3,
                  1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
        d['V'] = [1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0,
                  1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0,
                  1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0]
        d['A'] = [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1,
                  1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1,
                  1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1]
        d['Y'] = [3, 5, 1, 5, 2, 5, 2, 1, 4, 2, 3, 4, 2, 5, 5,
                  3, 5, 1, 5, 2, 5, 2, 1, 4, 2, 3, 4, 2, 5, 5,
                  3, 5, 1, 5, 2, 5, 2, 1, 4, 2, 3, 4, 2, 5, 5]
        d['I'] = 1

        # M-estimator
        def psi(theta):
            return ee_gestimation_snmm(theta=theta,
                                       y=d['Y'], A=d['A'],
                                       W=d[['I', 'V', 'W']],
                                       V=d[['I', 'V']],
                                       model='linear')

        mestr = MEstimator(psi, init=[0., ] * 5)

        # Auto-differentation
        mestr.estimate(solver='lm', deriv_method='exact')
        bread_exact = mestr.bread
        var_exact = mestr.variance

        # Central difference method
        mestr.estimate(solver='lm', deriv_method='approx')
        bread_approx = mestr.bread
        var_approx = mestr.variance

        # Checking bread estimates
        npt.assert_allclose(bread_approx,
                            bread_exact,
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(var_approx,
                            var_exact,
                            atol=1e-5)

    def test_robins_sensitivity_mean(self):
        d = pd.DataFrame()
        d['I'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        d['X'] = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
        d['Y'] = [7, 2, 5, np.nan, 1, 4, 8, np.nan, 1, np.nan]
        d['delta'] = np.where(d['Y'].isna(), 0, 1)

        def q_function(y_vals, alpha):
            y_no_miss = np.where(np.isnan(y_vals), 0, y_vals)
            return alpha * y_no_miss

        def psi(theta):
            return ee_mean_sensitivity_analysis(theta=theta,
                                                y=d['Y'], delta=d['delta'], X=d[['I', 'X']],
                                                q_eval=q_function(d['Y'], alpha=0.5),
                                                H_function=inverse_logit)

        mestr = MEstimator(psi, init=[0., 0., 0.])

        # Auto-differentation
        mestr.estimate(solver='lm', deriv_method='exact')
        bread_exact = mestr.bread
        var_exact = mestr.variance

        # Central difference method
        mestr.estimate(solver='lm', deriv_method='approx')
        bread_approx = mestr.bread
        var_approx = mestr.variance

        # Checking bread estimates
        npt.assert_allclose(bread_approx,
                            bread_exact,
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(var_approx,
                            var_exact,
                            atol=1e-6)


class TestSandwichApproxDiff:

    def test_capprox_bread_mean(self):
        # Data set
        y = np.array([5, 1, 2, 4, 2, 4, 5, 7, 11, 1, 6, 3, 4, 6])

        def psi(theta):
            return y - theta

        mestr = MEstimator(psi, init=[0, ])
        mestr.estimate(deriv_method='capprox')

        # Checking bread estimates
        npt.assert_allclose(mestr.bread,
                            [[1]],
                            rtol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mestr.asymptotic_variance,
                            np.var(y, ddof=0),
                            rtol=1e-6)

    def test_capprox_bread_mean_var(self):
        # Data set
        y = np.array([5, 1, 2, 4, 2, 4, 5, 7, 11, 1, 6, 3, 4, 6])

        def psi(theta):
            return ee_mean_variance(theta=theta, y=y)

        mestr = MEstimator(psi, init=[0, 1, ])
        mestr.estimate(deriv_method='capprox')
        bread_exact = mestr.bread
        var_exact = mestr.variance
        mestr.estimate(deriv_method='approx')
        bread_approx = mestr.bread
        var_approx = mestr.variance

        # Checking bread estimates
        npt.assert_allclose(bread_approx, bread_exact, atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(var_approx, var_exact, atol=1e-5)

    def test_capprox_bread_glm_lognb(self):
        d = pd.DataFrame()
        d['X'] = [1, -1, 0, 1, 2, 1, -2, -1, 0, 3, -3, 1, 1, -1, -1, -2, 2, 0, -1, 0]
        d['Z'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        d['Y'] = [0, 0, 0, 0, 0, 15, 15, 25, 25, 45, 0, 0, 0, 0, 15, 15, 15, 25, 25, 35]
        d['I'] = 1

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X']], y=d['Y'],
                          distribution='nb', link='log')

        # Auto-differentation
        mestr = MEstimator(psi, init=[0., 0., 1.])
        mestr.estimate(solver='lm', deriv_method='capprox')
        bread_exact = mestr.bread
        var_exact = mestr.variance

        # Central difference method
        mestr = MEstimator(psi, init=[0., 0., 1.])
        mestr.estimate(solver='lm', deriv_method='approx')
        bread_approx = mestr.bread
        var_approx = mestr.variance

        # Checking bread estimates
        npt.assert_allclose(bread_approx, bread_exact, atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(var_approx, var_exact, atol=1e-5)

    def test_capprox_aipw(self):
        d = pd.DataFrame()
        d['W'] = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3,
                  1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3,
                  1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
        d['V'] = [1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0,
                  1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0,
                  1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0]
        d['A'] = [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1,
                  1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1,
                  1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1]
        d['Y'] = [3, 5, 1, 5, 2, 5, 2, 1, 4, 2, 3, 4, 2, 5, 5,
                  3, 5, 1, 5, 2, 5, 2, 1, 4, 2, 3, 4, 2, 5, 5,
                  3, 5, 1, 5, 2, 5, 2, 1, 4, 2, 3, 4, 2, 5, 5]
        d['I'] = 1
        d['A1'] = 1
        d['A0'] = 0

        def psi(theta):
            return ee_aipw(theta, y=d['Y'], A=d['A'],
                           W=d[['I', 'W']],
                           X=d[['I', 'A', 'W']],
                           X1=d[['I', 'A1', 'W']],
                           X0=d[['I', 'A0', 'W']])

        mestr = MEstimator(psi, init=[0., ]*8)

        # Auto-differentation
        mestr.estimate(solver='lm', deriv_method='capprox')
        bread_exact = mestr.bread
        var_exact = mestr.variance

        # Central difference method
        mestr.estimate(solver='lm', deriv_method='approx')
        bread_approx = mestr.bread
        var_approx = mestr.variance

        # Checking bread estimates
        npt.assert_allclose(bread_approx,
                            bread_exact,
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(var_approx,
                            var_exact,
                            atol=1e-5)
