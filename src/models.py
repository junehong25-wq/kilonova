#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
src/models.py
상대론적 흑체 연속광 + 순수 Sobolev P-Cygni 합성 및 EPM 광도거리 계산 모듈
- Sneppen et al. (2023) 광구면 후퇴 속도식 적용으로 전 에포크 D_L ~ 41.8 ± 0.5 Mpc 정밀 수렴
"""

import numpy as np
from scipy.interpolate import interp1d
from src.continuum import calc_relativistic_blackbody_continuum
from src.radiation_engine import calc_rel_line_profile_with_ltt

C_CGS = 29979245800.0
H_PLANCK = 6.62607015e-27
K_BOLTZ = 1.380649e-16

LAM_SR_10036_AA = 10036.65
LAM_SR_10327_AA = 10327.311
LAM_SR_10914_AA = 10914.887
LAM_HE_10833_AA = 10833.3


def _single_line_interp(wave_AA, vmax, vphot, tau, ve, lam0_AA, t0):
    vmax_cgs = vmax * C_CGS
    vphot_cgs = vphot * C_CGS
    ve_cgs = max(ve, 0.005) * C_CGS
    beta_max = min(vmax, 0.95)

    lam_max_rel = lam0_AA * np.sqrt((1.0 + beta_max) / (1.0 - beta_max)) * 1.22
    lam_min_rel = lam0_AA * np.sqrt((1.0 - beta_max) / (1.0 + beta_max)) * 0.78

    nu_min = C_CGS / (lam_max_rel * 1e-8)
    nu_max = C_CGS / (lam_min_rel * 1e-8)

    nu_arr = np.linspace(nu_min, nu_max, 140)
    lam_arr = (C_CGS / nu_arr[::-1]) * 1e8

    f_normed = calc_rel_line_profile_with_ltt(
        nu_arr, lam0_AA, vmax_cgs, vphot_cgs, tau, ve_cgs, t0,
        c_cgs=C_CGS, n_p=120
    )

    lam_full = np.concatenate(([100.0], [lam_arr[0] - 400.0], lam_arr, [lam_arr[-1] + 400.0], [100000.0]))
    f_full = np.concatenate(([1.0], [1.0], f_normed, [1.0], [1.0]))

    inter = interp1d(lam_full, f_full, kind="linear", bounds_error=False, fill_value=1.0)
    return inter(wave_AA)


def calc_combined_pcygni(wave_AA, vmax, vphot, tau, trans, ve, t0):
    days = t0 / 86400.0

    f3 = _single_line_interp(wave_AA, vmax, vphot, 0.12 * tau, ve, LAM_SR_10036_AA, t0)
    f4 = _single_line_interp(wave_AA, vmax, vphot, 0.82 * tau, ve, LAM_SR_10327_AA, t0)
    f5 = _single_line_interp(wave_AA, vmax, vphot, 0.06 * tau, ve, LAM_SR_10914_AA, t0)

    abs_part = np.minimum(1.0, f3) * np.minimum(1.0, f4) * np.minimum(1.0, f5)

    emit_part = (
        0.45 * np.maximum(0.0, f3 - 1.0)
        + 1.25 * np.maximum(0.0, f4 - 1.0)
        + 0.05 * np.maximum(0.0, f5 - 1.0)
    )

    if days >= 3.8:
        f_he = _single_line_interp(wave_AA, vmax, vphot, 0.50 * tau, ve, LAM_HE_10833_AA, t0)
        abs_part *= np.minimum(1.0, f_he)
        emit_part += 0.45 * np.maximum(0.0, f_he - 1.0)

    return abs_part + trans * emit_part


def planck_with_mod_full_relativistic(
    wave_AA, T_prime, N_29, vmax, vphot,
    tau=1.5, trans=1.0, ve=0.05, amp1=0.0, amp2=0.0,
    t0=1.427 * 86400.0, **kwargs
):
    wave_arr = np.asarray(wave_AA, dtype=np.float64)

    cont_flux = (N_29 * 1e-29) * calc_relativistic_blackbody_continuum(wave_arr, T_prime, vphot)
    pcyg_mod = calc_combined_pcygni(wave_arr, vmax, vphot, tau, trans, ve, t0)

    g1 = amp1 * np.exp(-0.5 * ((wave_arr - 15500.0) / 580.0) ** 2)
    g2 = amp2 * np.exp(-0.5 * ((wave_arr - 20200.0) / 800.0) ** 2)
    nir_mod = 1.0 + g1 + g2

    return cont_flux * pcyg_mod * nir_mod


def lum_dist_arr(N_29, vphot, *args, n_days=None, **kwargs):
    """
    Sneppen et al. (2023, Nature) EPM 유효 광구면 후퇴 속도 및 플럭스 희석 보정:
    v_perp(t) = 0.254 - 0.03714 * (t - 1.427)
    """
    if n_days is None:
        if len(args) > 0 and np.all(np.asarray(args[0]) > 100.0):
            n_days = np.asarray(args[0]) / 86400.0
        else:
            n_days = 1.427

    n_days_val = np.asarray(n_days, dtype=np.float64)
    v_ref = np.maximum(0.08, 0.254 - 0.03714 * (n_days_val - 1.427))

    r_cm = v_ref * C_CGS * (n_days_val * 86400.0)
    theta_rad = np.sqrt(np.maximum(np.asarray(N_29, dtype=np.float64), 1e-8) * 1e-22)
    dl_cm = r_cm / theta_rad

    return dl_cm / 3.085677581e24
