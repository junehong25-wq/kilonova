#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
src/models.py
인위적 시그모이드 가드(emit_guard)를 완전히 제거한 순수 물리 기반 분광 합성 모듈
- 1~3일차: Sr II 삼중항 (10036, 10327, 10915 Å)
- 4~10일차: 비열적 베타붕괴 전자에 의한 NLTE He I 10833 Å 결합
"""

import numpy as np
import torch
from config.settings import (
    C_CGS, MPC_TO_CM,
    LAM_SR_10036_AA, LAM_SR_10327_AA, LAM_SR_10914_AA, LAM_HE_10833_AA,
    CEN1_AA, SIG1_AA, CEN2_AA, SIG2_AA
)
from src.continuum import DEVICE, DTYPE, calc_relativistic_blackbody_batch_gpu
from src.radiation_engine import calc_single_line_interp_batch_gpu


@torch.no_grad()
def calc_combined_pcygni_batch_gpu(wave_aa_t, vmax_t, vphot_t, tau_t, trans_t, ve_t, days, use_nlte=True):
    f3 = calc_single_line_interp_batch_gpu(wave_aa_t, vmax_t, vphot_t, 0.12 * tau_t, ve_t, LAM_SR_10036_AA, days, use_nlte)
    f4 = calc_single_line_interp_batch_gpu(wave_aa_t, vmax_t, vphot_t, 0.82 * tau_t, ve_t, LAM_SR_10327_AA, days, use_nlte)
    f5 = calc_single_line_interp_batch_gpu(wave_aa_t, vmax_t, vphot_t, 0.06 * tau_t, ve_t, LAM_SR_10914_AA, days, use_nlte)

    abs_part = torch.clamp(f3, max=1.0) * torch.clamp(f4, max=1.0) * torch.clamp(f5, max=1.0)
    emit_part = (
        0.45 * torch.clamp(f3 - 1.0, min=0.0)
        + 1.25 * torch.clamp(f4 - 1.0, min=0.0)
        + 0.05 * torch.clamp(f5 - 1.0, min=0.0)
    )

    if days >= 3.8 and use_nlte:
        f_he = calc_single_line_interp_batch_gpu(
            wave_aa_t, vmax_t, vphot_t, 0.55 * tau_t, ve_t, LAM_HE_10833_AA, days, use_nlte
        )
        abs_part = abs_part * torch.clamp(f_he, max=1.0)
        emit_part = emit_part + 0.50 * torch.clamp(f_he - 1.0, min=0.0)

    return abs_part + trans_t[:, None] * emit_part


@torch.no_grad()
def planck_with_mod_batch_gpu(wave_aa_t, theta_2d_np, days, use_nlte=True):
    if theta_2d_np.ndim == 1:
        theta_2d_np = theta_2d_np[None, :]

    th = torch.as_tensor(theta_2d_np, device=DEVICE, dtype=DTYPE)
    T_p = th[:, 0]
    N_29 = th[:, 1]
    vmax = th[:, 2]
    vphot = th[:, 3]
    tau = th[:, 4]
    trans = th[:, 5]
    ve = th[:, 6]
    amp1 = th[:, 7]
    amp2 = th[:, 8]

    cont_unit = calc_relativistic_blackbody_batch_gpu(wave_aa_t, T_p, vphot)
    cont_flux = (N_29[:, None] * 1e-29) * cont_unit

    pcyg_mod = calc_combined_pcygni_batch_gpu(wave_aa_t, vmax, vphot, tau, trans, ve, days, use_nlte=use_nlte)

    w_row = wave_aa_t[None, :]
    g1 = amp1[:, None] * torch.exp(-0.5 * ((w_row - CEN1_AA) / SIG1_AA) ** 2)
    g2 = amp2[:, None] * torch.exp(-0.5 * ((w_row - CEN2_AA) / SIG2_AA) ** 2)
    nir_mod = 1.0 + g1 + g2

    model_flux = cont_flux * pcyg_mod * nir_mod
    return model_flux, cont_flux, pcyg_mod


def planck_with_mod_full_relativistic(
    wave_AA, T_prime, N_29, vmax, vphot,
    tau=1.5, trans=1.0, ve=0.05, amp1=0.0, amp2=0.0,
    t0=1.427 * 86400.0, use_nlte=True, **kwargs
):
    days = t0 / 86400.0
    w_t = torch.as_tensor(wave_AA, device=DEVICE, dtype=DTYPE)
    theta_1d = np.array([T_prime, N_29, vmax, vphot, tau, trans, ve, amp1, amp2], dtype=np.float64)
    mod_t, _, _ = planck_with_mod_batch_gpu(w_t, theta_1d, days, use_nlte=use_nlte)
    return mod_t[0].cpu().numpy()


def calc_combined_pcygni(wave_AA, vmax, vphot, tau, trans, ve, t0, use_nlte=True):
    days = t0 / 86400.0
    w_t = torch.as_tensor(wave_AA, device=DEVICE, dtype=DTYPE)
    th = torch.tensor([[4000.0, 1.0, vmax, vphot, tau, trans, ve, 0.0, 0.0]], device=DEVICE, dtype=DTYPE)
    pcyg_t = calc_combined_pcygni_batch_gpu(w_t, th[:, 2], th[:, 3], th[:, 4], th[:, 5], th[:, 6], days, use_nlte=use_nlte)
    return pcyg_t[0].cpu().numpy()


def lum_dist_arr(N_29, vphot, *args, n_days=1.427, **kwargs):
    """
    Sneppen et al. (2023) 광구 후퇴 유효 횡단 속도 궤적 기반 EPM 광도거리 [Mpc]
    """
    n_days_val = np.asarray(n_days, dtype=np.float64)
    v_ref = np.maximum(0.06, 0.254 - 0.03714 * (n_days_val - 1.427))

    r_cm = v_ref * C_CGS * (n_days_val * 86400.0)
    theta_rad = np.sqrt(np.maximum(np.asarray(N_29, dtype=np.float64), 1e-8) * 1e-22)
    dl_cm = r_cm / theta_rad
    return dl_cm / MPC_TO_CM
