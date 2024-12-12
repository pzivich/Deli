Estimating Equations
=====================
To ease use, ``delicatessen`` comes with a variety of built-in estimating equations, covering a variety of common use
cases. Below is reference documentation for currently supported estimating equations.


Basic
-------------------

.. currentmodule:: delicatessen.estimating_equations.basic

.. autosummary::
  :toctree: generated/

  ee_mean
  ee_mean_robust
  ee_mean_variance
  ee_percentile
  ee_positive_mean_deviation


Regression
-------------------

.. currentmodule:: delicatessen.estimating_equations.regression

.. autosummary::
  :toctree: generated/

  ee_regression
  ee_mlogit
  ee_glm
  ee_robust_regression
  ee_ridge_regression
  ee_dlasso_regression
  ee_lasso_regression
  ee_elasticnet_regression
  ee_bridge_regression
  ee_additive_regression


Measurement
-------------------

.. currentmodule:: delicatessen.estimating_equations.measurement

.. autosummary::
  :toctree: generated/

  ee_rogan_gladen
  ee_rogan_gladen_extended
  ee_regression_calibration


Survival
-------------------

.. currentmodule:: delicatessen.estimating_equations.survival

.. autosummary::
  :toctree: generated/

  ee_exponential_model
  ee_exponential_measure
  ee_weibull_model
  ee_weibull_measure
  ee_aft_weibull
  ee_aft_weibull_measure


Pharmacokinetic Models
----------------------

.. currentmodule:: delicatessen.estimating_equations.pharmacokinetics

.. autosummary::
  :toctree: generated/

  ee_emax
  ee_emax_ed
  ee_loglogistic
  ee_loglogistic_ed


Causal Inference
-------------------

.. currentmodule:: delicatessen.estimating_equations.causal

.. autosummary::
  :toctree: generated/

  ee_gformula
  ee_ipw
  ee_ipw_msm
  ee_aipw
  ee_gestimation_snmm
  ee_mean_sensitivity_analysis
