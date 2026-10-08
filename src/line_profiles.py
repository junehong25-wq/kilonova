#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
src/line_profiles.py
단일 전이선 및 다중 전이선 P Cygni 프로파일 합성 인터페이스
"""

import numpy as np
from scipy.interpolate import interp1d
from src.radiation_engine import calc_rel_line_profile_with_ltt

C_CGS = 29979245800.0
LAM_SR_10036_AA = 10036.65
LAM_SR_10327_AA = 10327.311
LAM_SR_10914_AA = 10914.887
LAM_HE_10833_AA = 10833.3


def p_cygni_line_corr_rel_1d(wl_target, vmax, vphot, tau, lam0_AA, t0, use_nlte=True):
    c_cgs = C_CGS
    vmax_cgs, vphot_cgs = vmax * c_cgs, vphot * c_cgs
    beta_max = min(vmax, 0.99)

    lam_max_rel = lam0_AA * np.sqrt((1.0 + beta_max) / (1.0 - beta_max)) * 1.25
    lam_min_rel = lam0_AA * np.sqrt((1.0 - beta_max) / (1.0 + beta_max)) * 0.75

    nu_min = c_cgs / (lam_max_rel * 1e-8)
    nu_max = c_cgs / (lam_min_rel * 1e-8)

    nu_arr = np.linspace(nu_min, nu_max, 80)
    lam_arr = (c_cgs / nu_arr[::-1]) * 1e8

    f_normed = calc_rel_line_profile_with_ltt(nu_arr, lam0_AA, vmax_cgs, vphot_cgs, tau, t0, c_cgs, n_p=60, use_nlte=use_nlte)
    lam_full = np.concatenate(([100.0], [lam_arr[0] - 500.0], lam_arr, [lam_arr[-1] + 500.0], [100000.0]))
    f_full = np.concatenate(([1.0], [1.0], f_normed, [1.0], [1.0]))

    inter = interp1d(lam_full, f_full, kind='linear', bounds_error=False, fill_value=1.0)
    return inter(wl_target)


def get_combined_pcygni_profile(wav, vmax, vphot, tau_sr, tau_he, t0, use_nlte=True, use_he=False):
    """Beer-Lambert 기반 흡수 감쇄 결합 및 건설적 방출 합산"""
    f3 = p_cygni_line_corr_rel_1d(wav, vmax, vphot, (1.0 / 13.8) * tau_sr, LAM_SR_10036_AA, t0, use_nlte=use_nlte)
    f4 = p_cygni_line_corr_rel_1d(wav, vmax, vphot, (8.1 / 13.8) * tau_sr, LAM_SR_10327_AA, t0, use_nlte=use_nlte)
    f5 = p_cygni_line_corr_rel_1d(wav, vmax, vphot, (4.7 / 13.8) * tau_sr, LAM_SR_10914_AA, t0, use_nlte=use_nlte)

    abs_corr = np.minimum(1.0, f3) * np.minimum(1.0, f4) * np.minimum(1.0, f5)
    emit_corr = np.maximum(0.0, f3 - 1.0) + np.maximum(0.0, f4 - 1.0) + np.maximum(0.0, f5 - 1.0)

    if use_he and tau_he > 0.001:
        f_he = p_cygni_line_corr_rel_1d(wav, vmax, vphot, tau_he, LAM_HE_10833_AA, t0, use_nlte=use_nlte)
        abs_corr *= np.minimum(1.0, f_he)
        emit_corr += np.maximum(0.0, f_he - 1.0)

    return abs_corr + emit_corr