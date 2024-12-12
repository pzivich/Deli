Utilities
=====================
For manipulation of output or inputs, there are several basic utility functionalities for transformation of variables,
predicted parameters, or computations. Some are used internally by the estimating equations but are also made
available to users.

Data transformations
---------------------

.. currentmodule:: delicatessen.utilities

.. autosummary::
  :toctree: generated/

  logit
  inverse_logit
  identity
  robust_loss_functions
  spline
  regression_predictions
  polygamma
  digamma
  standard_normal_cdf
  standard_normal_pdf


Design matrices
---------------------

.. autosummary::
  :toctree: generated/

  additive_design_matrix


Differentiation
---------------------

.. currentmodule:: delicatessen.derivative

.. autosummary::
  :toctree: generated/

  approx_differentiation
  auto_differentiation


Sandwich Variance Estimator
---------------------------

.. currentmodule:: delicatessen.sandwich

.. autosummary::
   :toctree: generated/

   compute_sandwich
