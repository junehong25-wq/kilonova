#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
src/probability.py
순수 베이지안 우도 및 사전확률 모듈
- 모은하 거리 강제 구속(lp_dl) 완전 제거 (순수 EPM 독립 거리 측정)
- 파장별 인위적 우도 가중치 완전 제거 (전 파장 동등 평가)
"""

import numpy as np
from src.models import planck_with_mod_full_relativistic, lum_dist_arr


class MCMCProbabilityWrapper:
    def __init__(self, wave, flux, err, time_s, bounds, n_days=1.427, **kwargs):
        self.wave = np.asarray(wave, dtype=np.float64)
        self.flux = np.asarray(flux, dtype=np.float64)
        self.err = np.asarray(err, dtype=np.float64)
        self.time_s = float(time_s)
        self.bounds = bounds
        self.n_days = float(n_days)

    def log_prior(self, theta):
        # 1. 파라미터 경계 박스 점검
        for val, (b_low, b_high) in zip(theta, self.bounds):
            if val < b_low or val > b_high:
                return -np.inf

        T_p, N_29, vmax, vphot, tau, trans, ve, amp1, amp2 = theta

        # 2. 물리적 속도 위계 제약 (vphot < vmax)
        if vphot >= vmax - 0.010:
            return -np.inf

        # 3. 비물리적 거리 발산 방지를 위한 넓은 Flat Prior (20 ~ 80 Mpc)
        # 은하 거리를 특정 값(40.7 Mpc)으로 강제 유도하는 가우시안 제약 전면 삭제
        dl = lum_dist_arr(N_29, vphot, n_days=self.n_days)
        if dl < 15.0 or dl > 90.0:
            return -np.inf

        return 0.0

    def log_likelihood(self, theta):
        T_p, N_29, vmax, vphot, tau, trans, ve, amp1, amp2 = theta

        model = planck_with_mod_full_relativistic(
            self.wave, T_p, N_29, vmax, vphot,
            tau=tau, trans=trans, ve=ve,
            amp1=amp1, amp2=amp2, t0=self.time_s
        )

        # 파장별 인공 가중치 없이 관측 데이터와 측정 오차 그대로 평가
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