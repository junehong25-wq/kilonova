#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
src/models.py
상대론적 Planck 연속광, EPM 광도 거리 환산 함수(lum_dist_arr),
및 다중 성분(P-Cygni + Gaussian IR Humps) 스펙트럼 합성
"""

import numpy as np
from src.radiation_engine import calc_rel_line_profile_with_ltt

C_LIGHT = 2.99792458e10   # [cm/s]
C_KMS = 299792.458        # [km/s]
H_PLANCK = 6.62607015e-27 # [erg*s]
K_BOLTZ = 1.380649e-16   # [erg/K]


def lum_dist_arr(N_29, vphot, n_days):
    """
    EPM 팽창광구법 기반 광도 거리 D_L [Mpc] 계산 (배열 및 스칼라 지원)
    run_mcmc_fit.py 및 probability.py 호환용
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
    """lum_dist_arr 별칭"""
    return lum_dist_arr(N_29, vphot, n_days)


def calc_relativistic_blackbody_continuum(wave_aa, T_prime, vphot, n_mu=40):
    """
    상대론적 팽창 광구 표면 적분 흑체 연속광 플럭스 계산
    """
    wave_aa = np.asarray(wave_aa, dtype=np.float64)
    beta = float(vphot)
    gamma = 1.0 / np.sqrt(max(1e-6, 1.0 - beta ** 2))

    mu_arr = np.linspace(beta, 1.0, n_mu)
    dmu = mu_arr[1] - mu_arr[0]
    total_flux = np.zeros_like(wave_aa)

    lam_cm = wave_aa * 1e-8

    for mu in mu_arr:
        doppler_factor = 1.0 / (gamma * (1.0 - beta * mu))
        T_obs = T_prime * doppler_factor

        x = (H_PLANCK * C_LIGHT) / (lam_cm * K_BOLTZ * T_obs)
        x = np.clip(x, 1e-4, 500.0)

        # B_lambda(T_obs)
        b_lam = (2.0 * H_PLANCK * C_LIGHT ** 2 / (lam_cm ** 5)) / (np.exp(x) - 1.0)
        total_flux += b_lam * mu * dmu

    return 2.0 * np.pi * total_flux


def planck_with_mod_full_relativistic(
    wave,
    T_prime,
    N_29,
    vmax,
    vphot,
    tau=1.5,
    trans=1.0,
    ve=0.35,
    amp1=0.20,
    amp2=0.40,
    t0=1.5 * 86400.0,
    use_ltt=True,
    use_nlte=False
):
    """
    관측 파장 전 대역에 걸친 최종 상대론적 플럭스 스펙트럼 모델
    F(lambda) = F_cont * [abs_corr + emit_corr * trans] * (1 + IR_Humps)
    """
    wave = np.asarray(wave, dtype=np.float64)

    # 1. 기저 상대론적 흑체 연속광
    bb_cont = calc_relativistic_blackbody_continuum(wave, T_prime, vphot)
    cont_flux = (N_29 * 1e-29) * bb_cont

    # 2. Sr II 유효 단일 P-Cygni 라인 프로파일 계산 (중심 10,400 Å)
    sub_mask = (wave >= 6800.0) & (wave <= 13500.0)

    if np.any(sub_mask):
        sub_wave = np.linspace(6800.0, 13500.0, 100)
        raw_prof = calc_rel_line_profile_with_ltt(
            sub_wave,
            lam0=10400.0,
            vphot=vphot,
            vmax=vmax,
            tau_base=tau,
            ve=ve,
            t0=t0,
            use_ltt=use_ltt,
            use_nlte=use_nlte
        )
        full_raw = np.interp(wave, sub_wave, raw_prof, left=1.0, right=1.0)
    else:
        full_raw = np.ones_like(wave)

    # 3. 차폐 인자(trans) 일원화 결합 (이중 차폐 해소)
    abs_corr = np.minimum(1.0, full_raw)
    emit_corr = np.maximum(0.0, full_raw - 1.0)
    line_correction = abs_corr + emit_corr * trans

    # 4. 근적외선 가우시안 방출 험프 (1.55 um, 2.02 um)
    hump1 = amp1 * np.exp(-0.5 * ((wave - 15500.0) / 580.0) ** 2)
    hump2 = amp2 * np.exp(-0.5 * ((wave - 20200.0) / 650.0) ** 2)
    ir_modulation = 1.0 + hump1 + hump2

    # 5. 최종 플럭스 합성
    final_flux = cont_flux * line_correction * ir_modulation
    return final_flux