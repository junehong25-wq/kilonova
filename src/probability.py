#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
src/probability.py
기하학적 trans 제약이 적용된 순수 복사전달 MCMC 확률 밀도 래퍼
"""

import numpy as np
from src.models import planck_with_mod_full_relativistic, lum_dist_arr


class MCMCProbabilityWrapper(object):
    def __init__(self, x_fit, y_fit, err_fit, time_s, bounds=None, n_days=1.427, use_ltt=False, use_nlte=False, **kwargs):
        self.x_fit = x_fit
        self.y_fit = y_fit
        self.err_fit = err_fit
        self.time_s = time_s
        self.bounds = bounds
        self.n_days = n_days
        self.use_ltt = use_ltt
        self.use_nlte = use_nlte

    def log_prior(self, theta):
        if np.any(np.isnan(theta)) or np.any(np.isinf(theta)):
            return -np.inf

        T_prime, N_29, vmax, vphot, tau, trans, ve, amp1, amp2 = theta

        if self.bounds is not None:
            for val, (low, high) in zip(theta, self.bounds):
                if not (low <= val <= high):
                    return -np.inf
        else:
            if T_prime <= 500.0 or T_prime > 40000.0: return -np.inf
            if N_29 <= 1e-4 or N_29 > 100.0: return -np.inf
            if vphot <= 0.01 or vmax >= 0.99: return -np.inf
            if tau <= 0.001 or tau > 50.0: return -np.inf
            # 기하학적 차폐 인자 물리적 범위 (0.50 ~ 1.50)
            if trans < 0.50 or trans > 1.50: return -np.inf
            if ve <= 0.001 or ve > 0.90: return -np.inf
            if amp1 < 0.0 or amp1 > 5.0 or amp2 < 0.0 or amp2 > 5.0: return -np.inf

        if vphot >= vmax - 0.005 or ve <= 0.001 or tau <= 0.001 or N_29 <= 0.0 or trans < 0.50 or trans > 1.50:
            return -np.inf

        # 모은하 광도 거리 Gaussian Prior (40.0 ± 4.0 Mpc)
        dl = lum_dist_arr(np.array([N_29]), np.array([vphot]), n_days=self.n_days)[0]
        lp_dl = -0.5 * ((dl - 40.0) / 4.0) ** 2
        return lp_dl

    def log_likelihood(self, theta):
        T_prime, N_29, vmax, vphot, tau, trans, ve, amp1, amp2 = theta
        try:
            model = planck_with_mod_full_relativistic(
                wav=self.x_fit, T_prime=T_prime, N_29=N_29, vmax=vmax, vphot=vphot,
                tau=tau, trans=trans, ve=ve, amp1=amp1, amp2=amp2, t0=self.time_s,
                use_ltt=self.use_ltt, use_nlte=self.use_nlte
            )
            if np.any(np.isnan(model)) or np.any(np.isinf(model)) or np.any(model <= 0.0):
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