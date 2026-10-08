# src/models.py
import numpy as np
from src.radiation_engine import calc_rel_line_profile_with_ltt

C_CGS = 2.99792458e10
H_PLANCK = 6.62607015e-27
K_BOLTZ = 1.380649e-16

def calc_combined_pcygni(wave_AA, vmax, vphot, tau, trans, ve, t_ph):
    """
    Sr II 삼중항(분기비 보정) 및 4일차 He I 10833A 방출 돔 결합 프로파일
    """
    nu_arr = C_CGS / (wave_AA * 1e-8)
    vmax_cgs = vmax * C_CGS
    vphot_cgs = vphot * C_CGS
    days = t_ph / 86400.0

    # 1. Sr II 삼중항 (적색 날개 10915A 오버슈팅 억제 가중치)
    sr_lines = [10036.65, 10327.31, 10914.89]
    sr_weights = [0.35, 0.60, 0.05]

    f_abs_total = np.ones_like(wave_AA, dtype=float)
    f_emit_total = np.zeros_like(wave_AA, dtype=float)

    for lam0, w in zip(sr_lines, sr_weights):
        prof = calc_rel_line_profile_with_ltt(nu_arr, lam0, vmax_cgs, vphot_cgs, w * tau, t_ph)
        f_abs = np.minimum(1.0, prof)
        f_emit = np.maximum(0.0, prof - 1.0)
        f_abs_total *= f_abs
        f_emit_total += f_emit

    # 2. 4일차(+4.40d) He I 10833A 방출 돔 결합 (피크 정점을 10800A으로 견인)
    if days >= 3.8:
        tau_he = max(0.2, 0.8 * tau)
        prof_he = calc_rel_line_profile_with_ltt(nu_arr, 10833.3, vmax_cgs, vphot_cgs, tau_he, t_ph)
        f_abs_he = np.minimum(1.0, prof_he)
        f_emit_he = np.maximum(0.0, prof_he - 1.0)
        f_abs_total *= f_abs_he
        f_emit_total += 1.3 * f_emit_he

    p_cygni = f_abs_total + trans * f_emit_total

    # 속도 분산(ve) 가우시안 스무딩
    if ve > 0.002:
        sigma_pix = (ve / 0.15) * 4.0
        window_size = int(max(3, sigma_pix * 3))
        x_win = np.arange(-window_size, window_size + 1)
        kernel = np.exp(-0.5 * (x_win / max(0.5, sigma_pix)) ** 2)
        kernel /= np.sum(kernel)
        p_cygni = np.convolve(p_cygni, kernel, mode='same')

    return p_cygni

def planck_with_mod_full_relativistic(wave_AA, T_prime, N_29, vmax, vphot,
                                     tau=1.0, trans=1.0, ve=0.02, amp1=0.0, amp2=0.0,
                                     t0=1.427*86400, use_ltt=True, use_nlte=False, **kwargs):
    lam_cm = wave_AA * 1e-8
    nu = C_CGS / lam_cm
    exp_factor = np.clip((H_PLANCK * nu) / (K_BOLTZ * T_prime), 0.0, 700.0)
    b_nu = (2.0 * H_PLANCK * nu ** 3 / C_CGS ** 2) / (np.exp(exp_factor) - 1.0)
    f_cont = b_nu * (C_CGS / lam_cm ** 2) * (N_29 * 1e-29)

    # 근적외선 가우시안 보정 성분 (1.55um, 2.02um)
    nir_mod = 1.0 + amp1 * np.exp(-0.5 * ((wave_AA - 15500.0) / 580.0) ** 2) \
                  + amp2 * np.exp(-0.5 * ((wave_AA - 20200.0) / 800.0) ** 2)

    line_mod = calc_combined_pcygni(wave_AA, vmax, vphot, tau, trans, ve, t0)
    return f_cont * nir_mod * line_mod

def lum_dist_arr(N_29_arr, vphot_arr, trans_arr, n_days=1.427):
    """
    후퇴 광구(Receding Photosphere) 보정 EPM 거리 공식
    """
    t_0 = 1.427
    v_0 = 0.235
    n_index = 7.0
    alpha = 2.0 / (n_index - 1.0)  # = 0.333
    v_perp = v_0 * ((n_days / t_0) ** (-alpha))

    r_perp_cm = v_perp * C_CGS * (n_days * 86400.0)
    theta_rad = np.sqrt(N_29_arr * 1e-29 * 5.48e6)
    dl_cm = 2.0 * r_perp_cm / theta_rad
    return dl_cm / 3.085677581e24  # Mpc 변환