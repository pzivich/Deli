.. image:: images/delicatessen_header.png

Delicatessen
=====================================

``delicatessen`` is a one-stop shop for all your sandwich (variance) needs. This Python 3.8+ library supports
M-estimation and Generalized Method of Moments, which is a general statistical framework for estimating unknown
parameters. If you are an R user, I recommend looking into ``geex``
(`Saul & Hudgens (2020) <https://bsaul.github.io/geex/>`_). ``delicatessen`` supports a variety of pre-built estimating
equations as well as custom, user built estimating equations.

Here, we provide a brief overview of M-Estimation. For a more detailed, please refer to Ross et al. (2024),
Stefanski & Boos (2002), or Boos & Stefanski (2013). M-estimation was developed to study the large sample properties
of robust statistics. However, many common large-sample statistics can be expressed with estimating equations, so
M-estimation provides a unified structure and a streamlined approach to estimation. Let the parameter of interest be the
vector :math:`\theta = (\theta_1, \theta_2, ..., \theta_v)` and data is observed for :math:`n` independent units
:math:`O_1, O_2, …, O_n`. An M-estimator, :math:`\hat{\theta}`, is the solution to the estimating equation
:math:`\sum_{i=1}^{n} \psi(O_i, \hat{\theta}) = 0` where :math:`\psi` is a known :math:`v \times 1`-dimension estimating
function. M-estimators further provides a convenient and automatic method of calculating large-sample variance
estimators via the empirical sandwich variance estimator:

.. math::

    V_n(O,\hat{\theta}) = B_n(O,\hat{\theta})^{-1} F_n(O,\hat{\theta}) \left(B_n(O,\hat{\theta})^{-1}\right)^T

where the 'bread' is

.. math::

    B_n(O,\hat{\theta}) = n^{-1} \sum_{i=1}^n - \psi'(O_i, \hat{\theta})

where the :math:`\psi'` indicates the partial derivative, and the 'filling' is

.. math::

    F_n(O, \hat{\theta}) = n^{-1} \sum_{i=1}^n \psi(O_i, \hat{\theta}) \psi(O_i, \hat{\theta})^T

While M-Estimation is a general approach, widespread application is hindered by the corresponding derivative and matrix
calculations. To circumvent these barriers, ``delicatessen`` automates M-estimators using numerical approximation
methods.

The following description is a high-level overview. The user provides a :math:`v \times n` array of estimating
function(s) to the ``MEstimator`` class object. Next, the ``MEstimator`` object solves for :math:`\hat{\theta}` using a
root-finding algorithm. After successful completion of the root-finding, the bread is computed by numerically
approximating the partial derivatives and the filling is calculated. Finally, the empirical sandwich variance is
computed.

GMM is a related set of methods. The key distinction is the GMM-estimator instead solves

.. math::

    \text{argmin}_{\theta} \left[ \sum_{i=1}^n \psi(O_i, \hat{\theta}) \right]
        \text{Q}
        \left[ \sum_{i=1}^n \psi(O_i, \hat{\theta}) \right]


where :math:`\text{Q}` is a weight matrix. Unlike the M-estimator, the GMM-estimator allows for the dimension
of the estimating functions to be larger than the number of parameters (referred to as over-identification). This is
accomplished through the weight matrix. In the case where the estimating functions and parameters are the same size,
the M-estimator and GMM-estimator are expected to be equivalent.

Installation:
-------------

To install ``delicatessen``, use the following command in terminal or command prompt

``python -m pip install delicatessen``

Only two dependencies for ``delicatessen`` are: ``NumPy``, ``SciPy``.

While ``pandas`` is not a dependency, several examples are demonstrated with pandas for ease of data processing. To
replicate the tests in ``tests/`` you will need to also install ``pandas``, ``statsmodels`` and ``pytest`` (but these
are not necessary for use of the package).

Citation:
-------------
Please use the following citation for ``delicatessen``:
Zivich PN, Klose M, Cole SR, Edwards JK, & Shook-Sa BE. (2022). Delicatessen: M-Estimation in Python.
*arXiv preprint arXiv:2203.11300*. `URL <https://arxiv.org/abs/2203.11300>`_

.. code-block:: text

   @article{zivich2022,
     title={Delicatessen: M-estimation in Python},
     author={Zivich, Paul N and Klose, Mark and Cole, Stephen R and Edwards, Jessie K and Shook-Sa, Bonnie E},
     journal={arXiv preprint arXiv:2203.11300},
     year={2022}
   }

Contents:
-------------------------------------

.. toctree::
  :maxdepth: 3

  Basics.rst
  Examples/index
  Custom-EE
  Optimization Advice <Optimization Advice.rst>
  Reference/index
  Create a GitHub Issue <https://github.com/pzivich/Deli/issues>


Code and Issue Tracker
-----------------------------

Please report bugs, issues, or feature requests on GitHub
at `pzivich/Delicatessen <https://github.com/pzivich/Delicatessen/>`_.

Otherwise, you may contact me via email (gmail: zivich.5).

References
-----------------------------

Boos DD, & Stefanski LA. (2013). M-estimation (estimating equations). In *Essential Statistical Inference*
(pp. 297-337). Springer, New York, NY.

Saul BC, & Hudgens MG. (2020). The Calculus of M-Estimation in R with geex. *Journal of Statistical Software*,
92(2).

Stefanski LA, & Boos DD. (2002). The calculus of M-estimation. *The American Statistician*, 56(1), 29-38.

Ross RK, Zivich PN, Stringer JSA, & Cole SR. (2024). M-estimation for common epidemiological measures: introduction and
applied examples. *International Journal of Epidemiology*, 53(2), dyae030.

Zivich PN, Klose M, Cole SR, Edwards JK, & Shook-Sa BE. (2022). Delicatessen: M-Estimation in Python.
*arXiv preprint arXiv:2203.11300*.
