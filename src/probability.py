#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
src/probability.py
MCMC 베이즈 사후확률, 우도, 물리 사전분포
"""

import numpy as np
from src.models import planck_with_mod_full_relativistic, lum_dist_arr


class MCMCProbabilityWrapper:
    def __init__(
        self,
        wave,
        flux,
        err,
        time_s,
        bounds,
        n_days=1.427,
        use_ltt=True,
        use_nlte=False,
        **kwargs
    ):
        self.wave = np.asarray(wave, dtype=np.float64)
        self.flux = np.asarray(flux, dtype=np.float64)
        self.err = np.asarray(err, dtype=np.float64)
        self.time_s = float(time_s)
        self.bounds = bounds
        self.n_days = float(n_days)
        self.use_ltt = bool(use_ltt)
        self.use_nlte = bool(use_nlte)

    def log_prior(self, theta):
        for val, (b_low, b_high) in zip(theta, self.bounds):
            if val < b_low or val > b_high:
                return -np.inf

        T_p, N_29, vmax, vphot, tau, trans, ve, amp1, amp2 = theta

        if vphot >= vmax:
            return -np.inf

        dl = lum_dist_arr(N_29, vphot, trans, n_days=self.n_days)
        if dl < 20.0 or dl > 85.0:
            return -np.inf

        # 모은하 NGC 4993 거리 사전분포 (40.7 ± 2.5 Mpc)
        lp_dl = -0.5 * ((dl - 40.7) / 2.5) ** 2
        return lp_dl

    def log_likelihood(self, theta):
        T_p, N_29, vmax, vphot, tau, trans, ve, amp1, amp2 = theta

        model = planck_with_mod_full_relativistic(
            self.wave,
            T_p,
            N_29,
            vmax,
            vphot,
            tau=tau,
            trans=trans,
            ve=ve,
            amp1=amp1,
            amp2=amp2,
            t0=self.time_s,
            use_ltt=self.use_ltt,
            use_nlte=self.use_nlte,
        )

        diff = (self.flux - model) / self.err
        return -0.5 * np.sum(diff ** 2)

    def log_probability(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood(theta)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll

    def chi2_for_minimizer(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return 1e30
        ll = self.log_likelihood(theta)
        if not np.isfinite(ll):
            return 1e30
        return -2.0 * (lp + ll)

    def __call__(self, theta):
        return self.log_probability(theta)


LogPost = MCMCProbabilityWrapper


def lum_dist_mpc(N_29, vphot, *args, n_days=1.427, **kwargs):
    return lum_dist_arr(N_29, vphot, *args, n_days=n_days, **kwargs)