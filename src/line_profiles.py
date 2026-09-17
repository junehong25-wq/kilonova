import numpy as np
from scipy.interpolate import interp1d
from src.radiation_engine import calc_rel_line_profile_with_ltt
from config.settings import C_CGS, LAM_SR_10036_AA, LAM_SR_10327_AA, LAM_SR_10914_AA, LAM_HE_10833_AA

def p_cygni_line_corr_rel_1d(wl_target, vmax, vphot, tau, lam0_AA, t0, use_nlte=True):
    c_cgs = C_CGS
    vmax_cgs, vphot_cgs = vmax * c_cgs, vphot * c_cgs
    beta_max = min(vmax, 0.99)
    lam_max_rel = lam0_AA * np.sqrt((1.0 + beta_max) / (1.0 - beta_max)) * 1.08
    lam_min_rel = lam0_AA * np.sqrt((1.0 - beta_max) / (1.0 + beta_max)) * 0.92

    nu_min = c_cgs / (lam_max_rel * 1e-8)
    nu_max = c_cgs / (lam_min_rel * 1e-8)
    nu_arr = np.linspace(nu_min, nu_max, 40)
    lam_arr = (c_cgs / nu_arr[::-1]) * 1e8

    f_normed = calc_rel_line_profile_with_ltt(nu_arr, lam0_AA, vmax_cgs, vphot_cgs, tau, t0, c_cgs, n_p=40, use_nlte=use_nlte)
    lam_full = np.concatenate(([1000.0], [lam_arr[0] - 200.0], lam_arr, [lam_arr[-1] + 200.0], [50000.0]))
    f_full = np.concatenate(([1.0], [1.0], f_normed, [1.0], [1.0]))

    inter = interp1d(lam_full, f_full, kind='linear', bounds_error=False, fill_value=1.0)
    return inter(wl_target)

def get_combined_pcygni_profile(wav, vmax, vphot, tau_sr, tau_he, t0, use_nlte, use_he, combine_optical_depths_beer_lambert):
    f3 = p_cygni_line_corr_rel_1d(wav, vmax, vphot, 0.12 * tau_sr, LAM_SR_10036_AA, t0, use_nlte=use_nlte)
    f4 = p_cygni_line_corr_rel_1d(wav, vmax, vphot, 1.00 * tau_sr, LAM_SR_10327_AA, t0, use_nlte=use_nlte)
    f5 = p_cygni_line_corr_rel_1d(wav, vmax, vphot, 0.58 * tau_sr, LAM_SR_10914_AA, t0, use_nlte=use_nlte)

    if use_he and tau_he > 0.001:
        f_he = p_cygni_line_corr_rel_1d(wav, vmax, vphot, tau_he, LAM_HE_10833_AA, t0, use_nlte=use_nlte)
    else:
        f_he = np.ones_like(wav)

    return combine_optical_depths_beer_lambert(f3, f4, f5, f_he)
