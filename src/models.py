#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
src/models.py
킬로노바 AT2017gfo 상대론적 복사전달 합성 분광 모델 및 물리 엔진
"""

import numpy as np
import numba
from scipy import constants
from scipy.interpolate import interp1d

# 물리 상수 정의
h_planck = constants.h
c_speed = constants.c
k_B = constants.k
C_CGS = 29979245800.0

# Sneppen et al. (2023) 근적외선 가우시안 험프 파라미터 (단위: Å)
CEN1_AA, SIG1_AA = 15500.0, 580.0
CEN2_AA, SIG2_AA = 20200.0, 800.0


# =============================================================================
# 1. 상대론적 흑체 연속광 (Relativistic Continuum)
# =============================================================================
try:
    from src.continuum import calc_relativistic_blackbody_continuum
except ImportError:
    # src.continuum 임포트 불가 시 사용할 고정밀 수치적분 Fallback (wave_m**5 적용)
    @numba.njit(fastmath=True)
    def relativistic_blackbody_flam(wave_m, T_prime, beta, n_mu=32):
        if beta >= 1.0 or beta < 0.0 or T_prime <= 0.0 or wave_m <= 0.0:
            return 0.0
        if beta == 0.0:
            val_exp = (h_planck * c_speed) / (wave_m * k_B * T_prime)
            if val_exp > 700.0:
                return 0.0
            return (np.pi * (2.0 * h_planck * c_speed**2)) / ((wave_m**5) * (np.exp(val_exp) - 1.0))

        gamma = 1.0 / np.sqrt(1.0 - beta**2)
        dmu = 1.0 / n_mu
        sum_flux = 0.0
        for i in range(n_mu):
            mu_prime = (i + 0.5) * dmu
            delta = gamma * (1.0 + beta * mu_prime)
            val_exp = (h_planck * c_speed) / (delta * wave_m * k_B * T_prime)
            b_lam = 0.0 if val_exp > 700.0 else (2.0 * h_planck * c_speed**2) / ((wave_m**5) * (np.exp(val_exp) - 1.0))
            sum_flux += (1.0 / (delta**3)) * b_lam * (mu_prime + beta) * dmu

        return 2.0 * np.pi * gamma * sum_flux

    @numba.njit(fastmath=True)
    def calc_relativistic_blackbody_continuum(wave_AA, T_prime, beta, n_mu=32):
        wave_m = wave_AA * 1e-10
        n = len(wave_m)
        flux = np.zeros(n)
        for i in range(n):
            flux[i] = relativistic_blackbody_flam(wave_m[i], T_prime, beta, n_mu)
        return flux


# =============================================================================
# 2. 상대론적 Sobolev 복사전달 및 P Cygni 프로파일 계산
# =============================================================================
@numba.njit(fastmath=True)
def tau_anisotropic(r, mu, t, vdet_min, vdet_max, tauref, ve, c):
    """상대론적 팽창에 따른 비등방성 Sobolev 광학적 깊이"""
    if r <= vdet_min * t:
        return 0.0
    v = r / t
    tau_radial = tauref * np.exp(-v / ve) if (vdet_min <= v <= vdet_max and ve > 0.0) else 0.0
    beta = (r / t) / c
    if beta >= 1.0:
        return 1e10
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    num = (1.0 + mu * beta)**2
    den = gamma * (1.0 + mu * beta - (beta**2) * (1.0 - mu**2))
    return 1e10 if den <= 0.0 else tau_radial * (num / den)


@numba.njit(fastmath=True)
def calc_z_rel(p, nu, nu0, t, c):
    """등진동수 표면(Resonant surface) 좌표 z 계산"""
    if nu <= 0.0:
        return np.inf
    A = (nu0 / nu)**2
    beta_p = p / (c * t)
    if beta_p >= 1.0:
        return np.inf
    a, b = 1.0 + A, -2.0
    c_coef = 1.0 - A * (1.0 - beta_p**2)
    discriminant = b**2 - 4.0 * a * c_coef

    if discriminant < 0.0:
        return np.inf
    return ((-b - np.sqrt(discriminant)) / (2.0 * a)) * c * t


@numba.njit(fastmath=True)
def S_rel_exact(p, z, r, vmax, vphot, t, tauref, ve, c, n_steps=20):
    """상대론적 Sobolev 원천함수(Source Function) 수치 적분"""
    if r > vmax * t or r <= vphot * t or (z < 0.0 and p <= vphot * t):
        return 0.0
    beta = (r / t) / c
    if beta >= 1.0:
        return 0.0
    val_phot = 1.0 - (vphot * t / r)**2
    if val_phot < 0.0:
        return 0.0
    mu_phot = np.sqrt(val_phot)
    mu_c_comoving = (mu_phot - beta) / (1.0 - beta * mu_phot)

    dmu0_1 = 2.0 / n_steps
    beta1 = 0.0
    for i in range(n_steps):
        mu0 = -1.0 + (i + 0.5) * dmu0_1
        mu = (mu0 + beta) / (1.0 + beta * mu0)
        tau_val = tau_anisotropic(r, mu, t, vphot, vmax, tauref, ve, c)
        esc = 1.0 - tau_val / 2.0 if tau_val < 1e-5 else (1.0 - np.exp(-tau_val)) / tau_val
        beta1 += esc * dmu0_1
    beta1 *= 0.5

    dmu0_3 = (1.0 - mu_c_comoving) / n_steps
    beta3 = 0.0
    for i in range(n_steps):
        mu0 = mu_c_comoving + (i + 0.5) * dmu0_3
        mu = (mu0 + beta) / (1.0 + beta * mu0)
        tau_val = tau_anisotropic(r, mu, t, vphot, vmax, tauref, ve, c)
        esc = 1.0 - tau_val / 2.0 if tau_val < 1e-5 else (1.0 - np.exp(-tau_val)) / tau_val
        beta3 += esc * dmu0_3
    beta3 *= 0.5

    return beta3 / beta1 if beta1 > 0.0 else 0.0


@numba.njit(fastmath=True)
def Iemit_exact_fast(p, nu, vmax, vphot, t, tauref, ve, c, nu0, n_steps=20):
    """충돌 매개변수 p에 대한 관측자 좌표계 방출 강도 계산"""
    if vphot >= vmax or ve <= 0.0:
        return 0.0
    z = calc_z_rel(p, nu, nu0, t, c)
    if np.isinf(z):
        return 0.0
    r = np.sqrt(p**2 + z**2)
    tau = 0.0 if (z < 0.0 and p <= vphot * t) else (tau_anisotropic(r, z / r, t, vphot, vmax, tauref, ve, c) if r > 0.0 else 0.0)
    I_init = 1.0 if (p <= vphot * t) else 0.0
    S_src = S_rel_exact(p, z, r, vmax, vphot, t, tauref, ve, c, n_steps)
    I_comoving = I_init * np.exp(-tau) + S_src * (1.0 - np.exp(-tau))

    return I_comoving * ((nu / nu0)**3) * p


@numba.njit(fastmath=True)
def calc_rel_line_profile_1d_fast(nu_arr, lam0_AA, vmax_cgs, vphot_cgs, tauref, ve_cgs, t, c_cgs, n_p=50, n_steps_mu=20):
    """표면 적분을 통한 정규화된 1D 선윤곽 생성"""
    nu0 = c_cgs / (lam0_AA * 1e-8)
    rmax = t * vmax_cgs
    n_nu = len(nu_arr)
    fnu = np.zeros(n_nu)
    p_arr = np.linspace(0.0, rmax, n_p)
    dp = rmax / (n_p - 1)

    for i in range(n_nu):
        nu = nu_arr[i]
        sum_val = 0.0
        for j in range(n_p):
            p = p_arr[j]
            w = 0.5 if (j == 0 or j == n_p - 1) else 1.0
            sum_val += Iemit_exact_fast(p, nu, vmax_cgs, vphot_cgs, t, tauref, ve_cgs, c_cgs, nu0, n_steps_mu) * w
        fnu[i] = 2.0 * np.pi * sum_val * dp

    if fnu[0] <= 0.0 or np.isnan(fnu[0]):
        return np.ones(n_nu)
    baseline = fnu[0] * ((nu_arr / nu_arr[0])**3)
    return (fnu / baseline)[::-1]


def p_cygni_line_corr_rel_1d(wl_target, vmax, vphot, tau, lam0_AA, ve, t0):
    """목표 파장 그리드(wl_target)에 대해 보간된 P Cygni 보정 계수 반환"""
    c_cgs = C_CGS
    vmax_cgs, vphot_cgs, ve_cgs = vmax * c_cgs, vphot * c_cgs, ve * c_cgs
    beta_max = min(vmax, 0.99)
    lam_max_rel = lam0_AA * np.sqrt((1.0 + beta_max) / (1.0 - beta_max)) * 1.08
    lam_min_rel = lam0_AA * np.sqrt((1.0 - beta_max) / (1.0 + beta_max)) * 0.92

    nu_min = c_cgs / (lam_max_rel * 1e-8)
    nu_max = c_cgs / (lam_min_rel * 1e-8)
    nu_arr = np.linspace(nu_min, nu_max, 40)
    lam_arr = (c_cgs / nu_arr[::-1]) * 1e8

    f_normed = calc_rel_line_profile_1d_fast(nu_arr, lam0_AA, vmax_cgs, vphot_cgs, tau, ve_cgs, t0, c_cgs, n_p=50, n_steps_mu=20)
    lam_full = np.concatenate(([1000.0], [lam_arr[0] - 200.0], lam_arr, [lam_arr[-1] + 200.0], [50000.0]))
    f_full = np.concatenate(([1.0], [1.0], f_normed, [1.0], [1.0]))

    inter = interp1d(lam_full, f_full, kind='linear', bounds_error=False, fill_value=1.0)
    return inter(wl_target)


# =============================================================================
# 3. 최종 통합 스펙트럼 모델 및 광도 거리(D_L) 유도 함수
# =============================================================================
def planck_with_mod_full_relativistic(
    wav, T_prime, N_29, vmax, vphot,
    tau=1.69, trans=1.0, ve=0.313, amp1=0.31, amp2=0.44, t0=123552.0,
    **kwargs
):
    """
    상대론적 흑체 연속광, Sr II 삼중항 P-Cygni 프로파일,
    기하학적 은폐 인자(trans), 근적외선 가우시안 험프를 결합한 최종 플럭스 계산
    """
    if "tau_sr" in kwargs:
        tau = kwargs["tau_sr"]
    N = N_29 * 1e-29
    intensity = calc_relativistic_blackbody_continuum(wav, T_prime, vphot, n_mu=16)

    # Sr II 삼중항 프로파일 합성 (상대 강도비: 1.0 : 8.1 : 4.7)
    pcyg_prof3 = p_cygni_line_corr_rel_1d(wav, vmax, vphot, (1.0 / 13.8) * tau, 10036.65, ve, t0)
    pcyg_prof4 = p_cygni_line_corr_rel_1d(wav, vmax, vphot, (8.1 / 13.8) * tau, 10327.311, ve, t0)
    pcyg_prof5 = p_cygni_line_corr_rel_1d(wav, vmax, vphot, (4.7 / 13.8) * tau, 10914.887, ve, t0)
    correction = pcyg_prof3 * pcyg_prof4 * pcyg_prof5

    # 방출 영역(correction > 1.0)에 기하학적 은폐 인자(trans) 적용
    mask_emission = correction > 1.0
    correction[mask_emission] = (correction[mask_emission] - 1.0) * trans + 1.0

    # 근적외선 가우시안 험프 결합
    gauss1 = amp1 * np.exp(-0.5 * ((wav - CEN1_AA) / SIG1_AA)**2)
    gauss2 = amp2 * np.exp(-0.5 * ((wav - CEN2_AA) / SIG2_AA)**2)

    total_mod = correction + gauss1 + gauss2
    return N * intensity * total_mod


def lum_dist_arr(N_29_array, vphot_array, trans_array=1.0, n_days=1.427, dt=0.0):
    """
    팽창 광구 방법(EPM)을 적용하여 광구 반경과 각반경으로부터 광도 거리(Mpc) 산출
    """
    c_m = constants.c
    N = np.maximum(N_29_array * 1e-29, 1e-35)
    theta = 2.0 * np.sqrt(N * 5.48e6)
    v = vphot_array * c_m
    t = (n_days - dt) * (3600.0 * 24.0)
    r = v * t
    D = (r / theta) * 2.0
    return D * 3.2408e-23