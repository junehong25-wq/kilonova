#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from src.models import planck_with_mod_full_relativistic

class MCMCProbabilityWrapper(object):
    def __init__(self, x_fit, y_fit, err_fit, time_s, bounds, **kwargs):
        self.x_fit = x_fit
        self.y_fit = y_fit
        self.err_fit = err_fit
        self.time_s = time_s
        self.bounds = bounds

    def log_prior(self, theta):
        if np.any(np.isnan(theta)) or np.any(np.isinf(theta)):
            return -np.inf
        T_prime, N_29, vmax, vphot, tau, trans, ve, amp1, amp2 = theta
        for val, (low, high) in zip(theta, self.bounds):
            if not (low <= val <= high):
                return -np.inf
        # 유체역학적 제약조건 (광구 속도는 분출물 최외각 속도 미만이어야 함)
        if vphot >= vmax - 0.005 or ve <= 0.001 or tau <= 0.001 or N_29 <= 0.0:
            return -np.inf
        return 0.0

    def log_likelihood(self, theta):
        T_prime, N_29, vmax, vphot, tau, trans, ve, amp1, amp2 = theta
        try:
            model = planck_with_mod_full_relativistic(
                wav=self.x_fit, T_prime=T_prime, N_29=N_29, vmax=vmax, vphot=vphot,
                tau=tau, trans=trans, ve=ve, amp1=amp1, amp2=amp2, t0=self.time_s
            )
            if np.any(np.isnan(model)) or np.any(np.isinf(model)):
                return -np.inf
            total_chi2 = np.sum(((self.y_fit - model) / self.err_fit) ** 2)
            return -0.5 * total_chi2
        except Exception:
            return -np.inf

    def __call__(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood(theta)
        return lp + ll if np.isfinite(ll) else -np.inf

    def chi2_for_minimizer(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return 1e12
        ll = self.log_likelihood(theta)
        return -2.0 * ll if np.isfinite(ll) else 1e12