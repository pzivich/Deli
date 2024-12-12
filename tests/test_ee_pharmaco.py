####################################################################################################################
# Tests for built-in estimating equations -- dose-response
####################################################################################################################

import pytest
import numpy as np
import numpy.testing as npt

from delicatessen import MEstimator
from delicatessen.estimating_equations import (ee_emax, ee_emax_ed,
                                               ee_loglogistic, ee_loglogistic_ed)
from delicatessen.data import load_inderjit


class TestEstimatingEquationsDoseResponse:

    def test_emax(self):
        # R code used to check against
        # library(DoseFinding)
        # data = data.frame(r=c(7.58, 8., 8.3285714, 7.25, 7.375, 7.9625, 8.3555556,
        #                       6.9142857, 7.75, 6.8714286, 6.45, 5.9222222, 1.925,
        #                       2.8857143, 4.2333333, 1.1875, 0.8571429, 1.0571429,
        #                       0.6875, 0.525, 0.825, 0.25, 0.22, 0.44),
        #                   d=c(0., 0., 0., 0., 0., 0., 0.94, 0.94, 0.94, 1.88, 1.88,
        #                       1.88, 3.75, 3.75, 3.75, 7.5, 7.5, 7.5, 15, 15, 15, 30,
        #                       30, 30))
        # data$r = max(data$r) - data$r
        # emax0 <- fitMod(d, r, data = data,  model = "emax")
        # emax0
        d = load_inderjit()                    # Loading array of data
        resp_data = np.max(d[:, 0]) - d[:, 0]  # Response data
        dose_data = d[:, 1]                    # Dose data

        def psi(theta):
            return ee_emax(theta=theta, dose=dose_data, response=resp_data)

        # Optimization procedure
        estr = MEstimator(psi, init=[np.min(dose_data), np.max(resp_data), np.median(dose_data)])
        estr.estimate()

        # Checking mean estimate
        comparison_theta = [0.1404261, 9.8200414, 4.5745239]
        npt.assert_allclose(estr.theta, comparison_theta, atol=1e-5)

    def test_emax_ed(self):
        # NOTE: not checking against DoseFinding since ED does something different in that library
        # By-hand formula from Bonate
        ed50 = 4.5745239
        comparison_theta = np.asarray([0.5/0.5*ed50, 0.05/(1-0.05)*ed50, 0.10/(1-0.10)*ed50])

        d = load_inderjit()                    # Loading array of data
        resp_data = np.max(d[:, 0]) - d[:, 0]  # Response data
        dose_data = d[:, 1]                    # Dose data

        def psi(theta):
            emax = ee_emax(theta=theta, dose=dose_data, response=resp_data)
            ed50 = ee_emax_ed(theta[3], dose=dose_data, delta=0.50, ed50=theta[2])
            ed40 = ee_emax_ed(theta[4], dose=dose_data, delta=0.05, ed50=theta[2])
            ed60 = ee_emax_ed(theta[5], dose=dose_data, delta=0.10, ed50=theta[2])
            return np.vstack((emax, ed50, ed40, ed60))

        # Optimization procedure
        estr = MEstimator(psi, init=[0.1404261, 9.8200414, 4.5745239, 5, 1, 2])
        estr.estimate()

        # Checking mean estimate
        npt.assert_allclose(estr.theta[3:], comparison_theta, atol=1e-5)

    def test_emax_bonate(self):
        # Corrupted Bonate data (I added a zero dose to get R's DoseFinding to work
        d = [0, 5, 5, 13, 11, 14.5, 6.8, 42.5, 37.5, 25, 38, 40, 26.5, 27.7, 27.4, 45, 61.4, 52.8]
        r = [0, -12, -13, -28, -24, -45, -18, -26, -40, -45, -26, -29, -26, -22, -28, -36, -27, -48]

        # R code
        # library(DoseFinding)
        # d = c(0, 5, 5, 13, 11, 14.5, 6.8, 42.5, 37.5, 25, 38, 40, 26.5, 27.7, 27.4, 45, 61.4, 52.8)
        # r = c(0, -12, -13, -28, -24, -45, -18, -26, -40, -45, -26, -29, -26, -22, -28, -36, -27, -48)
        # data = data.frame(d=d, r=r)
        # emax0 <- fitMod(d, r, data = data,  model = "emax", bnds=c(0, 10))
        # emax0$coefs
        comparison_theta = np.asarray([1.658004, -40.138821, 5.739854])

        def psi(theta):
            return ee_emax(theta=theta, dose=d, response=r)

        estr = MEstimator(psi, init=[0, -40, 5])
        estr.estimate()

        # Checking mean estimate
        npt.assert_allclose(estr.theta, comparison_theta, atol=1e-5)

    def test_loglogistic4(self):
        # Compares against R's drc library:
        #
        # library(drc)
        # library(sandwich)
        # library(lmtest)
        # data(ryegrass)
        # rgll4 = drm(rootl ~ conc, data=ryegrass, fct=LL.4())
        # coeftest(rgll4, vcov=sandwich)
        #
        # Results from Ritz et al. and R
        comparison_theta = np.asarray([0.48141, 7.79296, 3.05795, 2.98222])
        comparison_var = np.asarray([0.12779, 0.15311, 0.26741, 0.47438])

        d = load_inderjit()
        dose_data = d[:, 1]
        resp_data = d[:, 0]

        def psi(theta):
            return ee_loglogistic(theta=theta, dose=dose_data, response=resp_data)

        # Optimization procedure
        estr = MEstimator(psi, init=[0, 10, 2, 1])
        estr.estimate(solver='lm')

        # Checking mean estimate
        npt.assert_allclose(estr.theta, comparison_theta, atol=1e-5)

        # Checking variance estimate
        npt.assert_allclose(np.diag(estr.variance)**0.5, comparison_var, atol=1e-4)

    def test_loglogistic3(self):
        # Compares against R's drc library:
        #
        # library(drc)
        # library(sandwich)
        # library(lmtest)
        # data(ryegrass)
        # rgll3 = drm(rootl ~ conc, data=ryegrass, fct=LL.3())
        # coeftest(rgll3, vcov=sandwich)
        # R optimization from Ritz et al.
        comparison_theta = np.asarray([7.85543, 3.26336, 2.47033])
        comparison_var = np.asarray([0.15397, 0.26572, 0.29238])

        d = load_inderjit()
        dose_data = d[:, 1]
        resp_data = d[:, 0]

        def psi(theta):
            return ee_loglogistic(theta=[0., ]+list(theta), dose=dose_data, response=resp_data)[1:, :]

        # Optimization procedure
        estr = MEstimator(psi, init=[10, 2, 1])
        estr.estimate(solver='lm')

        # Checking mean estimate
        npt.assert_allclose(estr.theta, comparison_theta, atol=1e-5)

        # Checking variance estimate
        npt.assert_allclose(np.diag(estr.variance)**0.5, comparison_var, atol=1e-5)

    def test_loglogistic2(self):
        # Compares against R's drc library:
        #
        # library(drc)
        # library(sandwich)
        # library(lmtest)
        #
        # data(ryegrass)
        # rgll2 = drm(rootl ~ conc, data=ryegrass, fct=LL.2(upper=8))
        # coeftest(rgll2, vcov=sandwich)
        # R optimization from Ritz et al.
        comparison_theta = np.asarray([3.19946, 2.38220])
        comparison_var = np.asarray([0.24290, 0.27937])

        d = load_inderjit()
        dose_data = d[:, 1]
        resp_data = d[:, 0]

        def psi(theta):
            return ee_loglogistic(theta=[0., 8.]+list(theta), dose=dose_data, response=resp_data)[2:, :]

        # Optimization procedure
        estr = MEstimator(psi, init=[2, 1])
        estr.estimate(solver='lm')

        # Checking mean estimate
        npt.assert_allclose(estr.theta, comparison_theta, atol=1e-5)

        # Checking variance estimate
        npt.assert_allclose(np.diag(estr.variance)**0.5, comparison_var, atol=1e-5)

    def test_loglogistic_ed(self):
        # Compares against R's drc library:
        #
        # library(drc)
        # library(sandwich)
        # data(ryegrass)
        # rgll3 = drm(rootl ~ conc, data=ryegrass, fct=LL.3())
        # ED(rgll3, c(5, 10, 50), interval='delta', vcov=sandwich)
        # R optimization from Ritz et al.
        comparison_theta = np.asarray([0.99088, 1.34086, 3.26336])
        comparison_var = np.asarray([0.12397, 0.13134, 0.26572])

        d = load_inderjit()
        dose_data = d[:, 1]
        resp_data = d[:, 0]

        def psi(theta):
            lower_limit = 0
            pl3 = ee_loglogistic(theta=[lower_limit, ]+list(theta), dose=dose_data, response=resp_data)[1:, :]
            ed05 = ee_loglogistic_ed(theta[3], dose=dose_data, delta=0.05,
                                     lower=lower_limit, upper=theta[0], ed50=theta[1], steepness=theta[2])
            ed10 = ee_loglogistic_ed(theta[4], dose=dose_data, delta=0.10,
                                     lower=lower_limit, upper=theta[0], ed50=theta[1], steepness=theta[2])
            ed50 = ee_loglogistic_ed(theta[5], dose=dose_data, delta=0.50,
                                     lower=lower_limit, upper=theta[0], ed50=theta[1], steepness=theta[2])
            return np.vstack((pl3,
                              ed05, ed10, ed50))

        # Optimization procedure
        estr = MEstimator(psi, init=[10, 3, 2, 1, 1, 2])
        estr.estimate(solver='lm')

        # Checking mean estimate
        npt.assert_allclose(estr.theta[-3:], comparison_theta, atol=1e-5)

        # Checking variance estimate
        npt.assert_allclose(np.diag(estr.variance)[-3:]**0.5, comparison_var, atol=1e-5)

    def test_loglogistic_ed2(self):
        # Compares against R's drc library:
        #
        # library(drc)
        # library(sandwich)
        # data = data.frame(r=c(7.58, 8., 8.3285714, 7.25, 7.375, 7.9625, 8.3555556,
        #                       6.9142857, 7.75, 6.8714286, 6.45, 5.9222222, 1.925,
        #                       2.8857143, 4.2333333, 1.1875, 0.8571429, 1.0571429,
        #                       0.6875, 0.525, 0.825, 0.25, 0.22, 0.44),
        #                   d=c(0., 0., 0., 0., 0., 0., 0.94, 0.94, 0.94, 1.88, 1.88,
        #                       1.88, 3.75, 3.75, 3.75, 7.5, 7.5, 7.5, 15, 15, 15, 30,
        #                       30, 30))
        # data$r = max(data$r) - data$r
        # rgll3 = drm(r ~ d, data=data, fct=LL.3())
        # ED(rgll3, c(5, 10, 50))
        # R optimization from Ritz et al.
        comparison_theta = np.asarray([0.87552, 1.18213, 2.8583])

        d = load_inderjit()
        dose_data = d[:, 1]
        resp_data = np.max(d[:, 0]) - d[:, 0]

        def psi(theta):
            lower_limit = 0
            pl3 = ee_loglogistic(theta=[lower_limit, ]+list(theta), dose=dose_data, response=resp_data)[1:, :]
            ed05 = ee_loglogistic_ed(theta[3], dose=dose_data, delta=0.05,
                                     lower=lower_limit, upper=theta[0], ed50=theta[1], steepness=theta[2])
            ed10 = ee_loglogistic_ed(theta[4], dose=dose_data, delta=0.10,
                                     lower=lower_limit, upper=theta[0], ed50=theta[1], steepness=theta[2])
            ed50 = ee_loglogistic_ed(theta[5], dose=dose_data, delta=0.50,
                                     lower=lower_limit, upper=theta[0], ed50=theta[1], steepness=theta[2])
            return np.vstack((pl3,
                              ed05, ed10, ed50))

        # Optimization procedure
        estr = MEstimator(psi, init=[10, 3, -2, 1, 1, 2])
        estr.estimate(solver='lm')

        # Checking mean estimate
        npt.assert_allclose(estr.theta[-3:], comparison_theta, atol=1e-4)
