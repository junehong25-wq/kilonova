#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
src/radiation_engine.py
적분 구간 분할(Split-Domain Gauss-Legendre Quadrature) 기반 상대론적 복사전달 & 스토크스 (I, Q) 커널
- 기존 MCMC 피팅용 calc_single_line_interp_batch_gpu 100% 하위 호환 유지
- 외곽 헤일로 구적 구간 [v_phot, v_max]에서 특수 상대론적 광행차(mu'), 도플러 부스팅(delta^3),
  원소 각도 분포 개구각(theta_open_deg) 덮개 인자, 소볼레프 공명 감쇄(exp(-tau_sob))를 통합 적분
"""

import numpy as np
import torch
from config.settings import (
    LAM_SR_10036_AA, LAM_SR_10327_AA, LAM_SR_10914_AA, LAM_HE_10833_AA
)
from src.continuum import DEVICE, COMPUTE_DTYPE

_N32, _W32 = np.polynomial.legendre.leggauss(32)
GL_NODES_32 = torch.tensor(0.5 * (_N32 + 1.0), device=DEVICE, dtype=COMPUTE_DTYPE)
GL_WEIGHTS_32 = torch.tensor(0.5 * _W32, device=DEVICE, dtype=COMPUTE_DTYPE)

UNIT_GRID_128 = torch.linspace(0.0, 1.0, 128, device=DEVICE, dtype=COMPUTE_DTYPE)


def _eval_zone_contribution_compiled(
    A_3d, beta_p, w_p, vphot_3d, vmax_3d, tau_base_3d, ve_3d, days, is_core_zone, use_nlte
):
    a_coef = 1.0 + A_3d
    c_coef = 1.0 - A_3d * (1.0 - beta_p ** 2)

    disc = 4.0 - 4.0 * a_coef * c_coef
    valid_disc = (disc >= 0.0)
    beta_z = (2.0 - torch.sqrt(torch.clamp(disc, min=0.0))) / (2.0 * a_coef)
    beta_r = torch.sqrt(beta_p ** 2 + beta_z ** 2)

    if is_core_zone:
        bz_front = torch.sqrt(torch.clamp(vphot_3d ** 2 - beta_p ** 2, min=0.0))
        in_shell = valid_disc & (beta_z >= bz_front) & (beta_r >= vphot_3d) & (beta_r <= vmax_3d)
        d_delay = beta_z - bz_front
        i_init = 1.0
    else:
        in_shell = valid_disc & (beta_r >= vphot_3d) & (beta_r <= vmax_3d)
        d_delay = beta_z
        i_init = 0.0

    t_eff_ratio = torch.clamp(1.0 + d_delay, min=0.25, max=1.50)
    t_eff_days = days * t_eff_ratio

    if use_nlte:
        alpha_t = torch.where(
            (t_eff_days > 1.5) & (t_eff_days <= 3.0),
            t_eff_days / 1.5,
            torch.where(
                (t_eff_days > 3.0) & (t_eff_days <= 5.0),
                torch.clamp(2.0 - 0.5 * (t_eff_days - 3.0), min=0.6),
                1.0
            )
        )
    else:
        alpha_t = torch.ones_like(t_eff_days)

    taper_w = 0.02 * (vmax_3d - vphot_3d) + 1e-4
    phase_edge = torch.clamp((vmax_3d - beta_r) / taper_w, 0.0, 1.0)
    smooth_edge = torch.where(beta_r > vmax_3d - taper_w, 0.5 * (1.0 - torch.cos(np.pi * phase_edge)), 1.0)

    dv_shell = torch.clamp(beta_r - vphot_3d, min=0.0)
    tau_radial = tau_base_3d * alpha_t * torch.exp(-dv_shell / ve_3d) * smooth_edge

    mu = beta_z / torch.clamp(beta_r, min=1e-6)
    beta_loc = torch.clamp(beta_r, max=0.95)
    gamma_loc = 1.0 / torch.sqrt(torch.clamp(1.0 - beta_loc ** 2, min=1e-4))
    num_aniso = (1.0 + mu * beta_loc) ** 2
    den_aniso = gamma_loc * (1.0 + mu * beta_loc - (beta_loc ** 2) * (1.0 - mu ** 2))
    tau_sob = torch.where(den_aniso > 1e-5, tau_radial * (num_aniso / den_aniso), 50.0)
    tau_sob = torch.clamp(tau_sob, 0.0, 50.0)

    r_ratio = torch.clamp(vphot_3d / torch.clamp(beta_r, min=1e-5), 0.0, 1.0)
    mu_phot = torch.sqrt(torch.clamp(1.0 - r_ratio ** 2, min=0.0))
    W_geom = 0.5 * (1.0 - mu_phot)

    T_factor = (1.0 / t_eff_ratio) ** 0.5
    S_source = torch.clamp(W_geom * T_factor, 0.0, 0.90)

    exp_neg_tau = torch.exp(-tau_sob)
    i_active = i_init * exp_neg_tau + (1.0 - exp_neg_tau) * S_source
    i_final = torch.where(in_shell, i_active, i_init)

    return torch.sum(i_final * beta_p * w_p, dim=-1)


def _eval_zone_stokes_contribution(
    A_3d, beta_p, w_p, vphot_3d, vmax_3d, tau_base_3d, ve_3d, days,
    is_core_zone, use_nlte, relativistic_aberration=True
):
    a_coef = 1.0 + A_3d
    c_coef = 1.0 - A_3d * (1.0 - beta_p ** 2)

    disc = 4.0 - 4.0 * a_coef * c_coef
    valid_disc = (disc >= 0.0)
    beta_z = (2.0 - torch.sqrt(torch.clamp(disc, min=0.0))) / (2.0 * a_coef)
    beta_r = torch.sqrt(beta_p ** 2 + beta_z ** 2)

    if is_core_zone:
        bz_front = torch.sqrt(torch.clamp(vphot_3d ** 2 - beta_p ** 2, min=0.0))
        in_shell = valid_disc & (beta_z >= bz_front) & (beta_r >= vphot_3d) & (beta_r <= vmax_3d)
        d_delay = beta_z - bz_front
        i_init = 1.0
    else:
        in_shell = valid_disc & (beta_r >= vphot_3d) & (beta_r <= vmax_3d)
        d_delay = beta_z
        i_init = 0.0

    t_eff_ratio = torch.clamp(1.0 + d_delay, min=0.25, max=1.50)
    t_eff_days = days * t_eff_ratio

    if use_nlte:
        alpha_t = torch.where(
            (t_eff_days > 1.5) & (t_eff_days <= 3.0),
            t_eff_days / 1.5,
            torch.where(
                (t_eff_days > 3.0) & (t_eff_days <= 5.0),
                torch.clamp(2.0 - 0.5 * (t_eff_days - 3.0), min=0.6),
                1.0
            )
        )
    else:
        alpha_t = torch.ones_like(t_eff_days)

    taper_w = 0.02 * (vmax_3d - vphot_3d) + 1e-4
    phase_edge = torch.clamp((vmax_3d - beta_r) / taper_w, 0.0, 1.0)
    smooth_edge = torch.where(beta_r > vmax_3d - taper_w, 0.5 * (1.0 - torch.cos(np.pi * phase_edge)), 1.0)

    dv_shell = torch.clamp(beta_r - vphot_3d, min=0.0)
    exp_density = torch.exp(-dv_shell / ve_3d) * smooth_edge
    tau_radial = tau_base_3d * alpha_t * exp_density

    mu = beta_z / torch.clamp(beta_r, min=1e-6)
    beta_loc = torch.clamp(beta_r, max=0.95)
    gamma_loc = 1.0 / torch.sqrt(torch.clamp(1.0 - beta_loc ** 2, min=1e-4))
    num_aniso = (1.0 + mu * beta_loc) ** 2
    den_aniso = gamma_loc * (1.0 + mu * beta_loc - (beta_loc ** 2) * (1.0 - mu ** 2))
    tau_sob = torch.where(den_aniso > 1e-5, tau_radial * (num_aniso / den_aniso), 50.0)
    tau_sob = torch.clamp(tau_sob, 0.0, 50.0)

    r_ratio = torch.clamp(vphot_3d / torch.clamp(beta_r, min=1e-5), 0.0, 1.0)
    mu_phot = torch.sqrt(torch.clamp(1.0 - r_ratio ** 2, min=0.0))
    W_geom = 0.5 * (1.0 - mu_phot)

    T_factor = (1.0 / t_eff_ratio) ** 0.5
    S_source = torch.clamp(W_geom * T_factor, 0.0, 0.90)

    exp_neg_tau = torch.exp(-tau_sob)
    i_active = i_init * exp_neg_tau + (1.0 - exp_neg_tau) * S_source
    i_final = torch.where(in_shell, i_active, i_init)
    flux_I = torch.sum(i_final * beta_p * w_p, dim=-1)

    if is_core_zone:
        flux_Q_trans = torch.zeros_like(flux_I)
        flux_Q_cont = torch.zeros_like(flux_I)
    else:
        if relativistic_aberration:
            mu_prime = (mu - beta_loc) / torch.clamp(1.0 - beta_loc * mu, min=1e-4)
            mu_prime = torch.clamp(mu_prime, -1.0, 1.0)
            delta_loc = 1.0 / torch.clamp(gamma_loc * (1.0 - beta_loc * mu), min=0.1)
            q_source_loc = W_geom * T_factor * exp_density * (0.75 * (1.0 - mu_prime ** 2) / (1.0 + mu_prime ** 2)) * (delta_loc ** 3)
        else:
            # 비상대론적(Static/Classical) Thomson 산란 대조군
            q_source_loc = W_geom * exp_density * (0.75 * (1.0 - mu ** 2) / (1.0 + mu ** 2))

        q_cont_loc = torch.where(in_shell, q_source_loc, torch.zeros_like(q_source_loc))
        q_act_loc = torch.where(in_shell, q_source_loc * exp_neg_tau, torch.zeros_like(q_source_loc))

        flux_Q_cont = torch.sum(q_cont_loc * beta_p * w_p, dim=-1)
        flux_Q_trans = torch.sum(q_act_loc * beta_p * w_p, dim=-1)

    return flux_I, flux_Q_trans, flux_Q_cont


def _single_line_kernel(wave_aa_t, vmax_t, vphot_t, tau_t, ve_t, lam0_aa, days, use_nlte):
    beta_max = torch.clamp(vmax_t, 0.05, 0.90)
    beta_phot = torch.clamp(vphot_t, 0.02, 0.85)
    span_halo = torch.clamp(beta_max - beta_phot, min=1e-4)
    ve_clean = torch.clamp(ve_t, min=0.005)

    doppler_stretch = torch.sqrt((1.0 + beta_max) / (1.0 - beta_max))
    lam_min = (lam0_aa / doppler_stretch) * 0.85
    lam_max = (lam0_aa * doppler_stretch) * 1.15

    lam_grid = lam_min[:, None] + (lam_max - lam_min)[:, None] * UNIT_GRID_128[None, :]
    A_3d = (lam_grid[:, :, None] / lam0_aa) ** 2

    vmax_3d = beta_max[:, None, None]
    vphot_3d = beta_phot[:, None, None]
    span_3d = span_halo[:, None, None]
    ve_3d = ve_clean[:, None, None]
    tau_base_3d = torch.clamp(tau_t, min=0.0)[:, None, None]

    bp_core = vphot_3d * GL_NODES_32[None, None, :]
    wp_core = vphot_3d * GL_WEIGHTS_32[None, None, :]
    flux_core = _eval_zone_contribution_compiled(
        A_3d, bp_core, wp_core, vphot_3d, vmax_3d, tau_base_3d, ve_3d, days, True, use_nlte
    )

    bp_halo = vphot_3d + span_3d * GL_NODES_32[None, None, :]
    wp_halo = span_3d * GL_WEIGHTS_32[None, None, :]
    flux_halo = _eval_zone_contribution_compiled(
        A_3d, bp_halo, wp_halo, vphot_3d, vmax_3d, tau_base_3d, ve_3d, days, False, use_nlte
    )

    norm_factor = 0.5 * (beta_phot ** 2)[:, None]
    f_grid = (flux_core + flux_halo) / norm_factor

    w_obs = wave_aa_t[None, :]
    pos_float = ((w_obs - lam_min[:, None]) / (lam_max - lam_min)[:, None]) * 127.0
    in_window = (pos_float >= 0.0) & (pos_float <= 127.0)

    pos_clamped = torch.clamp(pos_float, 0.0, 126.999)
    idx_lo = pos_clamped.long()
    idx_hi = idx_lo + 1
    weight_hi = pos_clamped - idx_lo.to(COMPUTE_DTYPE)

    val_lo = torch.gather(f_grid, 1, idx_lo)
    val_hi = torch.gather(f_grid, 1, idx_hi)
    interp_val = (1.0 - weight_hi) * val_lo + weight_hi * val_hi

    return torch.where(in_window, interp_val, 1.0)


@torch.no_grad()
def calc_single_line_interp_batch_gpu(wave_aa_t, vmax_t, vphot_t, tau_t, ve_t, lam0_aa, days, use_nlte=True):
    return _single_line_kernel(
        wave_aa_t, vmax_t, vphot_t, tau_t, ve_t, float(lam0_aa), float(days), bool(use_nlte)
    )


@torch.no_grad()
def calc_single_line_stokes_batch_gpu(
    wave_aa_t, vmax_t, vphot_t, tau_t, ve_t, lam0_aa, days, use_nlte=True, relativistic_aberration=True
):
    beta_max = torch.clamp(vmax_t, 0.05, 0.90)
    beta_phot = torch.clamp(vphot_t, 0.02, 0.85)
    span_halo = torch.clamp(beta_max - beta_phot, min=1e-4)
    ve_clean = torch.clamp(ve_t, min=0.005)

    doppler_stretch = torch.sqrt((1.0 + beta_max) / (1.0 - beta_max))
    lam_min = (lam0_aa / doppler_stretch) * 0.85
    lam_max = (lam0_aa * doppler_stretch) * 1.15

    lam_grid = lam_min[:, None] + (lam_max - lam_min)[:, None] * UNIT_GRID_128[None, :]
    A_3d = (lam_grid[:, :, None] / lam0_aa) ** 2

    vmax_3d = beta_max[:, None, None]
    vphot_3d = beta_phot[:, None, None]
    span_3d = span_halo[:, None, None]
    ve_3d = ve_clean[:, None, None]
    tau_base_3d = torch.clamp(tau_t, min=0.0)[:, None, None]

    bp_core = vphot_3d * GL_NODES_32[None, None, :]
    wp_core = vphot_3d * GL_WEIGHTS_32[None, None, :]
    flux_core, _, _ = _eval_zone_stokes_contribution(
        A_3d, bp_core, wp_core, vphot_3d, vmax_3d, tau_base_3d, ve_3d,
        float(days), True, bool(use_nlte), bool(relativistic_aberration)
    )

    bp_halo = vphot_3d + span_3d * GL_NODES_32[None, None, :]
    wp_halo = span_3d * GL_WEIGHTS_32[None, None, :]
    flux_halo, q_halo_act, q_halo_cont = _eval_zone_stokes_contribution(
        A_3d, bp_halo, wp_halo, vphot_3d, vmax_3d, tau_base_3d, ve_3d,
        float(days), False, bool(use_nlte), bool(relativistic_aberration)
    )

    norm_factor = 0.5 * (beta_phot ** 2)[:, None]
    f_grid = (flux_core + flux_halo) / norm_factor

    t_pol_grid = torch.where(
        q_halo_cont > 1e-12,
        torch.clamp(q_halo_act / torch.clamp(q_halo_cont, min=1e-12), 0.0, 1.0),
        torch.ones_like(q_halo_act)
    )
    q_norm_grid = q_halo_cont / norm_factor

    w_obs = wave_aa_t[None, :]
    pos_float = ((w_obs - lam_min[:, None]) / (lam_max - lam_min)[:, None]) * 127.0
    in_window = (pos_float >= 0.0) & (pos_float <= 127.0)

    pos_clamped = torch.clamp(pos_float, 0.0, 126.999)
    idx_lo = pos_clamped.long()
    idx_hi = idx_lo + 1
    weight_hi = pos_clamped - idx_lo.to(COMPUTE_DTYPE)

    f_lo = torch.gather(f_grid, 1, idx_lo)
    f_hi = torch.gather(f_grid, 1, idx_hi)
    f_interp = torch.where(in_window, (1.0 - weight_hi) * f_lo + weight_hi * f_hi, 1.0)

    tp_lo = torch.gather(t_pol_grid, 1, idx_lo)
    tp_hi = torch.gather(t_pol_grid, 1, idx_hi)
    tp_interp = torch.where(in_window, (1.0 - weight_hi) * tp_lo + weight_hi * tp_hi, 1.0)

    qn_mean = torch.mean(q_norm_grid, dim=1)
    return f_interp, tp_interp, qn_mean


@torch.no_grad()
def calc_relativistic_stokes_spectrum_gpu(
    wave_AA, T_prime, vmax, vphot, tau, trans, ve, days,
    eta_axis=0.88, phi_lan_deg=30.0, theta_obs_deg=22.0,
    theta_open_deg=90.0, include_he_nlte=True, relativistic_aberration=True
):
    """
    3D GPU 분할 구적 커널 기반 상대론적 스토크스 (I, Q) 스펙트럼 합성:
    - Stokes I는 src/models.py와 100% 일치 (Plot2 정합성 보장)
    - theta_open_deg: 원소(Sr II / He I)의 극지방 원뿔 개구각 [deg] (90 deg = 전방위 구형 분포)
      헤일로 전자 산란 영역에 대한 공명선 덮개 비율 f_cov = sin^3(theta_open) 반영
    """
    w_t = torch.as_tensor(wave_AA, device=DEVICE, dtype=COMPUTE_DTYPE)
    vm_t = torch.tensor([float(vmax)], device=DEVICE, dtype=COMPUTE_DTYPE)
    vp_t = torch.tensor([float(vphot)], device=DEVICE, dtype=COMPUTE_DTYPE)
    ve_t = torch.tensor([float(ve)], device=DEVICE, dtype=COMPUTE_DTYPE)

    if include_he_nlte:
        w_he = 1.0 / (1.0 + np.exp(-(days - 3.0) / 0.45))
    else:
        w_he = 0.0
    w_sr = 1.0 - 0.65 * w_he

    tau_10036 = torch.tensor([float(tau * w_sr * 0.11)], device=DEVICE, dtype=COMPUTE_DTYPE)
    tau_10327 = torch.tensor([float(tau * w_sr * 0.50)], device=DEVICE, dtype=COMPUTE_DTYPE)
    tau_10914 = torch.tensor([float(tau * w_sr * 0.39)], device=DEVICE, dtype=COMPUTE_DTYPE)
    tau_10833 = torch.tensor([float(tau * w_he * 0.85)], device=DEVICE, dtype=COMPUTE_DTYPE)

    tau_pol_he = max(float(tau), 0.22 * np.sqrt(max(float(trans), 0.0))) * w_he * 0.85
    tau_pol_10833 = torch.tensor([tau_pol_he], device=DEVICE, dtype=COMPUTE_DTYPE)

    rel_flag = bool(relativistic_aberration)
    f_1, tp_1, _ = calc_single_line_stokes_batch_gpu(w_t, vm_t, vp_t, tau_10036, ve_t, LAM_SR_10036_AA, days, True, rel_flag)
    f_2, tp_2, qn_2 = calc_single_line_stokes_batch_gpu(w_t, vm_t, vp_t, tau_10327, ve_t, LAM_SR_10327_AA, days, True, rel_flag)
    f_3, tp_3, _ = calc_single_line_stokes_batch_gpu(w_t, vm_t, vp_t, tau_10914, ve_t, LAM_SR_10914_AA, days, True, rel_flag)
    f_he, _, _ = calc_single_line_stokes_batch_gpu(w_t, vm_t, vp_t, tau_10833, ve_t, LAM_HE_10833_AA, days, True, rel_flag)
    _, tp_he, _ = calc_single_line_stokes_batch_gpu(w_t, vm_t, vp_t, tau_pol_10833, ve_t, LAM_HE_10833_AA, days, True, rel_flag)

    fa_1, _, _ = calc_single_line_stokes_batch_gpu(w_t, vm_t, vp_t, tau_10036, ve_t, LAM_SR_10036_AA, days, False, rel_flag)
    fa_2, _, _ = calc_single_line_stokes_batch_gpu(w_t, vm_t, vp_t, tau_10327, ve_t, LAM_SR_10327_AA, days, False, rel_flag)
    fa_3, _, _ = calc_single_line_stokes_batch_gpu(w_t, vm_t, vp_t, tau_10914, ve_t, LAM_SR_10914_AA, days, False, rel_flag)
    fa_he, _, _ = calc_single_line_stokes_batch_gpu(w_t, vm_t, vp_t, tau_10833, ve_t, LAM_HE_10833_AA, days, False, rel_flag)

    f_full = f_1 * f_2 * f_3 * f_he
    f_abs = torch.clamp(fa_1 * fa_2 * fa_3 * fa_he, max=1.0)
    f_emit = torch.clamp(f_full - f_abs, min=0.0)
    i_norm_t = torch.clamp(f_abs + float(trans) * f_emit, min=1e-4)

    # 원소 공간 분포 개구각(theta_open_deg)에 따른 헤일로 편광 덮개 인자 f_cov = sin^3(theta_open)
    # 좁은 극지방 원뿔(theta_open = 30 deg)이면 적도/중위도 전자 산란 영역의 편광 파괴가 약해짐
    f_cov = float(np.sin(np.deg2rad(np.clip(float(theta_open_deg), 10.0, 90.0))) ** 3)
    t_pol_raw = tp_1 * tp_2 * tp_3 * tp_he
    t_pol_total_t = (1.0 - f_cov) + f_cov * t_pol_raw

    q_halo_integral = float(qn_2[0].cpu().item())

    dv_span = max(1e-3, float(vmax) - float(vphot))
    col_integral = float(ve) * (1.0 - np.exp(-dv_span / max(float(ve), 1e-3)))
    tau_es_t = 0.72 * ((1.427 / float(days)) ** 2) * (col_integral / 0.085)

    theta_rad = np.deg2rad(float(theta_obs_deg))
    sin_phi_lan = np.sin(np.deg2rad(float(phi_lan_deg)))
    asym_geom = np.abs(1.0 - float(eta_axis) ** 2) / (1.0 + float(eta_axis) ** 2)
    asym_lan_shadow = 0.75 * sin_phi_lan * (1.0 - sin_phi_lan ** 2)
    asymmetry_total = (asym_geom + asym_lan_shadow) * (np.sin(theta_rad) ** 2)

    lam_thermal_aa = 7400.0 * ((4500.0 / max(float(T_prime), 2000.0)) ** 0.35)
    kappa_bb_over_es = ((w_t / lam_thermal_aa) ** (-3.2)) * (0.35 + 0.65 * sin_phi_lan)
    albedo_es = 1.0 / (1.0 + kappa_bb_over_es)

    q_cont_t = 100.0 * tau_es_t * q_halo_integral * asymmetry_total * albedo_es
    q_norm_t = q_cont_t * t_pol_total_t
    p_pct_t = q_norm_t / i_norm_t

    return (
        p_pct_t[0].cpu().numpy().astype(np.float64),
        q_cont_t[0].cpu().numpy().astype(np.float64),
        i_norm_t[0].cpu().numpy().astype(np.float64),
        float(tau_es_t)
    )
