import numpy as np
import numba
from scipy import constants
from src.continuum import calc_relativistic_blackbody_continuum
from src.line_profiles import p_cygni_line_corr_rel_1d, get_combined_pcygni_profile
from config.settings import CEN1_AA, SIG1_AA, CEN2_AA, SIG2_AA, LAM_SR_10036_AA, LAM_SR_10327_AA, LAM_SR_10914_AA, LAM_HE_10833_AA

@numba.njit(fastmath=True)
def combine_optical_depths_beer_lambert(f3, f4, f5, f_he, trans):
    n = len(f3)
    result = np.zeros(n)
    for i in range(n):
        tau3 = -np.log(np.maximum(1e-10, min(1.0, f3[i])))
        tau4 = -np.log(np.maximum(1e-10, min(1.0, f4[i])))
        tau5 = -np.log(np.maximum(1e-10, min(1.0, f5[i])))
        tau_he = -np.log(np.maximum(1e-10, min(1.0, f_he[i])))
        total_tau = tau3 + tau4 + tau5 + tau_he

        em3 = np.maximum(0.0, f3[i] - 1.0)
        em4 = np.maximum(0.0, f4[i] - 1.0)
        em5 = np.maximum(0.0, f5[i] - 1.0)
        em_he = np.maximum(0.0, f_he[i] - 1.0)
        total_em = (em3 + em4 + em5 + em_he) * trans

        result[i] = np.exp(-total_tau) + total_em
    return result

def planck_with_mod_full_relativistic(
        wav, T_prime, N_29, vmax, vphot, tau_sr=3.80, tau_he=0.0, trans=1.0, t0=123552.0,
        use_nlte=True, use_he=True
):
    N = N_29 * 1e-29
    intensity = calc_relativistic_blackbody_continuum(wav, T_prime, vphot, n_mu=16)

    # Use exact ratios 0.12, 1.00, 0.58 from Physics Rules memory
    f3 = p_cygni_line_corr_rel_1d(wav, vmax, vphot, 0.12 * tau_sr, LAM_SR_10036_AA, t0, use_nlte=use_nlte)
    f4 = p_cygni_line_corr_rel_1d(wav, vmax, vphot, 1.00 * tau_sr, LAM_SR_10327_AA, t0, use_nlte=use_nlte)
    f5 = p_cygni_line_corr_rel_1d(wav, vmax, vphot, 0.58 * tau_sr, LAM_SR_10914_AA, t0, use_nlte=use_nlte)

    if use_he and tau_he > 0.001:
        pcyg_he = p_cygni_line_corr_rel_1d(wav, vmax, vphot, tau_he, LAM_HE_10833_AA, t0, use_nlte=use_nlte)
    else:
        pcyg_he = np.ones_like(wav)

    # Use Beer-Lambert exponential combination per Physics Rules memory
    total_line_corr = combine_optical_depths_beer_lambert(f3, f4, f5, pcyg_he, trans)

    total_mod = total_line_corr
    return N * intensity * total_mod


def lum_dist_arr(N_29_array, vphot_array, trans_array=1.0, n_days=1.427, dt=0.0):
    c_m = constants.c
    N = np.maximum(N_29_array * 1e-29, 1e-35)
    theta = 2.0 * np.sqrt(N * 5.48e6)
    v = vphot_array * c_m
    t = (n_days - dt) * (3600.0 * 24.0)
    r = v * t

    D = (r / theta) * 2.0
    D_mpc = D * (3.2408e-23)
    return D_mpc
