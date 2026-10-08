# src/probability.py
import numpy as np


class MCMCProbabilityWrapper(object):
    def __init__(self, x_fit, y_fit, err_fit, time_s, bounds, **kwargs):
        self.x_fit = x_fit
        self.y_fit = y_fit
        self.err_fit = err_fit
        self.time_s = time_s
        self.bounds = bounds
        self.kwargs = kwargs

    def log_prior(self, theta):
        for val, (low, high) in zip(theta, self.bounds):
            if val < low or val > high:
                return -np.inf

        # vphot가 vmax보다 크거나 같아지는 비물리적 상황 방지
        vmax = theta[2]
        vphot = theta[3]
        if vphot >= vmax - 0.015:
            return -np.inf
        return 0.0

    def chi2_for_minimizer(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return 1e12
        from src.models import planck_with_mod_full_relativistic
        model = planck_with_mod_full_relativistic(
            self.x_fit, theta[0], theta[1], theta[2], theta[3],
            tau=theta[4], trans=theta[5], ve=theta[6], amp1=theta[7], amp2=theta[8],
            t0=self.time_s
        )
        return float(np.sum(((self.y_fit - model) / self.err_fit) ** 2))

    def __call__(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        chi2 = self.chi2_for_minimizer(theta)
        if np.isnan(chi2) or np.isinf(chi2):
            return -np.inf
        return lp - 0.5 * chi2