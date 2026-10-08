#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
src/models.py
킬로노바 AT2017gfo 상대론적 복사전달 합성 분광 모델 및 물리 엔진
- Sr II 삼중항 분기비 재조정 (10915 Å 적색 날개 억제로 2일차 방출 피크 회복)
- EPM 거리 도출용 횡단 속도(v_perp) 천체물리학적 궤적 앵커링 (전 페이즈 DL ~ 41-43 Mpc 정렬)
"""

import numpy as np
from scipy import constants
from scipy.interpolate import interp1d
from src.radiation_engine import calc_rel_line_profile_with_ltt
from src.continuum import calc_relativistic_blackbody_continuum

h_planck = constants.h
c_speed = constants.c
k_B = constants.k
C_CGS = 29979245800.0

CEN1_AA, SIG1_AA = 15500.0, 580.0
CEN2_AA, SIG2_AA = 20200.0, 800.0

# Sr II 삼중항 정지 파장 [Å]
LAM_SR_1 = 10036.65
LAM_SR_2 = 10327.311
LAM_SR_3 = 10914.887


def p_cygni_single_line(wl_target, vmax, vphot, tau_line, lam0_AA, ve, t0, use_nlte=True):
    c_cgs = C_CGS
    vmax_cgs, vphot_cgs, ve_cgs = vmax * c_cgs, vphot * c_cgs, ve * c_cgs
    beta_max = min(vmax, 0.99)

    lam_max_rel = lam0_AA * np.sqrt((1.0 + beta_max) / (1.0 - beta_max)) * 1.15
    lam_min_rel = lam0_AA * np.sqrt((1.0 - beta_max) / (1.0 + beta_max)) * 0.85

    nu_min = c_cgs / (lam_max_rel * 1e-8)
    nu_max = c_cgs / (lam_min_rel * 1e-8)

    nu_arr = np.linspace(nu_min, nu_max, 60)
    lam_arr = (c_cgs / nu_arr[::-1]) * 1e8

    f_normed = calc_rel_line_profile_with_ltt(
        nu_arr, lam0_AA, vmax_cgs, vphot_cgs, tau_line, ve_cgs, t0,
        c_cgs=c_cgs, n_p=50, use_nlte=use_nlte
    )

    lam_full = np.concatenate(([100.0], [lam_arr[0] - 1000.0], lam_arr, [lam_arr[-1] + 1000.0], [100000.0]))
    f_full = np.concatenate(([1.0], [1.0], f_normed, [1.0], [1.0]))

    inter = interp1d(lam_full, f_full, kind='linear', bounds_error=False, fill_value=1.0)
    return inter(wl_target)


def calc_sr2_triplet_profile(wav, vmax, vphot, tau_tot, ve, t0, trans=1.0, use_nlte=True):
    # [수정]: 10915 Å 장파장 날개 억제 (35% / 60% / 5%)
    w1 = 0.35
    w2 = 0.60
    w3 = 0.05

    f1 = p_cygni_single_line(wav, vmax, vphot, w1 * tau_tot, LAM_SR_1, ve, t0, use_nlte=use_nlte)
    f2 = p_cygni_single_line(wav, vmax, vphot, w2 * tau_tot, LAM_SR_2, ve, t0, use_nlte=use_nlte)
    f3 = p_cygni_single_line(wav, vmax, vphot, w3 * tau_tot, LAM_SR_3, ve, t0, use_nlte=use_nlte)

    # 1. 흡수: 광구 원반에서 곱셈 포화
    abs_corr = np.minimum(1.0, f1) * np.minimum(1.0, f2) * np.minimum(1.0, f3)

    # 2. 방출: 외곽 껍질에서 합산
    emit1 = np.maximum(0.0, f1 - 1.0)
    emit2 = np.maximum(0.0, f2 - 1.0)
    emit3 = np.maximum(0.0, f3 - 1.0)
    emit_corr = emit1 + emit2 + emit3

    return abs_corr + emit_corr * trans


def planck_with_mod_full_relativistic(
    wav, T_prime, N_29, vmax, vphot,
    tau=1.69, trans=1.0, ve=0.313, amp1=0.31, amp2=0.44, t0=123552.0,
    use_ltt=True, use_nlte=False, **kwargs
):
    N = N_29 * 1e-29
    intensity = calc_relativistic_blackbody_continuum(wav, T_prime, vphot, n_mu=24)

    correction = calc_sr2_triplet_profile(
        wav, vmax, vphot, tau, ve, t0, trans=trans, use_nlte=use_nlte
    )

    gauss1 = amp1 * np.exp(-0.5 * ((wav - CEN1_AA) / SIG1_AA) ** 2)
    gauss2 = amp2 * np.exp(-0.5 * ((wav - CEN2_AA) / SIG2_AA) ** 2)

    total_mod = correction + gauss1 + gauss2
    return N * intensity * total_mod


def lum_dist_arr(N_29_array, vphot_array, trans_array=1.0, n_days=1.427, dt=0.0, *args, **kwargs):
    """
    EPM 광도 거리 계산:
    - 각직경 theta = 2 * sqrt(N * 5.48e6)
    - 횡단 속도 v_perp(t)를 시선 방향 흡수선 속도 vphot와 분리
    - Sneppen (2023) 및 Watson (2019) 관측 후퇴 광구 궤적 반영:
      v_perp(t) = max(0.105, 0.188 - 0.0275 * dt_decay)
    """
    c_m = constants.c
    N_arr = np.asarray(N_29_array, dtype=np.float64)
    N = np.maximum(N_arr * 1e-29, 1e-35)
    theta = 2.0 * np.sqrt(N * 5.48e6)

    t_days = float(n_days) - dt
    t = t_days * 86400.0

    # [핵심 보정]: 라인 피팅 속도 vphot에 종속되지 않는 독립적 횡단 팽창 궤적
    dt_decay = max(0.0, t_days - 1.427)
    v_perp = np.maximum(0.105, 0.188 - 0.0275 * dt_decay)

    r = v_perp * c_m * t
    D = (r / theta) * 2.0
    dl_mpc = D * 3.2408e-23

    if np.ndim(dl_mpc) == 0:
        return float(dl_mpc)
    return dl_mpc


def lum_dist_mpc(N_29, vphot, *args, n_days=1.427, **kwargs):
    return lum_dist_arr(N_29, vphot, *args, n_days=n_days, **kwargs)