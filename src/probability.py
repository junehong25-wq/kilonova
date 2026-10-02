#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
src/probability.py
MCMC 베이즈 사후확률 (Log-Posterior), 우도 (Likelihood), 물리 사전분포 (Prior)
MCMCProbabilityWrapper 클래스 및 초기 최적화용 chi2_for_minimizer 구현
"""

import numpy as np
from src.models import planck_with_mod_full_relativistic

C_KMS = 299792.458


def lum_dist_arr(N_29, vphot, n_days):
    """
    EPM 팽창광구법 기반 광도 거리 D_L [Mpc] 계산 (배열 및 스칼라 지원)
    """
    N_29 = np.asarray(N_29, dtype=np.float64)
    vphot = np.asarray(vphot, dtype=np.float64)

    theta_rad = 2.0 * np.sqrt(5.48e6 * np.maximum(N_29, 1e-30) * 1e-29)
    r_phot_cm = vphot * (C_KMS * 1e5) * (n_days * 86400.0)
    d_cm = (2.0 * r_phot_cm) / np.maximum(theta_rad, 1e-30)
    dl_mpc = d_cm / 3.08567758149e24

    if dl_mpc.ndim == 0:
        return float(dl_mpc)
    return dl_mpc


def lum_dist_mpc(N_29, vphot, n_days):
    """lum_dist_arr 별칭 함수"""
    return lum_dist_arr(N_29, vphot, n_days)


class MCMCProbabilityWrapper:
    """
    MCMC 샘플러(emcee) 및 scipy.optimize용 확률 래퍼 클래스
    """
    def __init__(
        self,
        wave,
        flux,
        err,
        time_s=None,
        bounds=None,
        n_days=None,
        use_ltt=True,
        use_nlte=False,
        **kwargs
    ):
        self.wave = np.asarray(wave, dtype=np.float64)
        self.flux = np.asarray(flux, dtype=np.float64)
        self.err = np.asarray(err, dtype=np.float64)

        if n_days is not None:
            self.n_days = float(n_days)
            self.t0 = self.n_days * 86400.0
        elif time_s is not None:
            self.t0 = float(time_s)
            self.n_days = self.t0 / 86400.0
        else:
            self.n_days = 1.5
            self.t0 = 1.5 * 86400.0

        if bounds is not None:
            self.bounds = bounds
        else:
            self.bounds = [
                (1800.0, 7500.0),  # T_prime [K]
                (0.1, 15.0),       # N_29
                (0.25, 0.45),      # vmax
                (0.15, 0.32),      # vphot
                (0.1, 15.0),       # tau
                (0.50, 1.50),      # trans
                (0.05, 0.50),      # ve
                (0.0, 0.80),       # amp1
                (0.0, 0.80)        # amp2
            ]

        self.use_ltt = bool(use_ltt)
        self.use_nlte = bool(use_nlte)

    def log_prior(self, theta):
        """사전분포: 파라미터 경계 및 광도 거리 가우시안 사전분포"""
        for val, (b_low, b_high) in zip(theta, self.bounds):
            if val < b_low or val > b_high:
                return -np.inf

        T_p, N_29, vmax, vphot, tau, trans, ve, amp1, amp2 = theta

        if vphot >= vmax:
            return -np.inf

        # 초기 추정치가 갇히지 않도록 유효 탐색 폭(20 ~ 85 Mpc) 허용
        dl = lum_dist_arr(N_29, vphot, self.n_days)
        if dl < 20.0 or dl > 85.0:
            return -np.inf

        # NGC 4993 모은하 참값(40.7 Mpc) 중심 정규 사전분포
        lp_dl = -0.5 * ((dl - 40.7) / 2.5) ** 2
        return lp_dl

    def log_likelihood(self, theta):
        """우도 함수"""
        T_p, N_29, vmax, vphot, tau, trans, ve, amp1, amp2 = theta

        model = planck_with_mod_full_relativistic(
            self.wave, T_p, N_29, vmax, vphot,
            tau=tau, trans=trans, ve=ve,
            amp1=amp1, amp2=amp2, t0=self.t0,
            use_ltt=self.use_ltt, use_nlte=self.use_nlte
        )

        diff = (self.flux - model) / self.err
        return -0.5 * np.sum(diff ** 2)

    def log_probability(self, theta):
        """사후확률 log P = log Prior + log Likelihood"""
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood(theta)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll

    def chi2_for_minimizer(self, theta):
        """
        scipy.optimize.minimize (Nelder-Mead)용 목적함수
        MAP(최대 사후확률) 추정을 위해 -2 * log_post 반환
        """
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return 1e30
        ll = self.log_likelihood(theta)
        if not np.isfinite(ll):
            return 1e30
        return -2.0 * (lp + ll)

    def __call__(self, theta):
        return self.log_probability(theta)


# 하위 호환성 별칭
LogPost = MCMCProbabilityWrapper