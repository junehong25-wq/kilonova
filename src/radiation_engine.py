# src/radiation_engine.py
import numpy as np
import numba


@numba.njit(fastmath=True)
def calc_z_rel(p, nu, nu0, t_ph, c_cgs=2.99792458e10):
    nu_ratio = nu0 / nu
    A = nu_ratio ** 2
    a = 1.0 + A
    b = -2.0
    beta_p = p / (c_cgs * t_ph)
    c_coef = 1.0 - A * (1.0 - beta_p ** 2)
    disc = b ** 2 - 4.0 * a * c_coef
    if disc < 0.0:
        return np.inf
    beta_z = (-b - np.sqrt(disc)) / (2.0 * a)
    return beta_z * c_cgs * t_ph


@numba.njit(fastmath=True)
def calc_rel_line_profile_with_ltt(nu_arr, lam0_AA, vmax_cgs, vphot_cgs,
                                   tau_base, t_ph, c_cgs=2.99792458e10, n_p=80):
    """
    n_p=80 격자 상향으로 우도 표면의 인공 요철 및 코너 플롯 줄무늬 제거
    """
    nu0 = c_cgs / (lam0_AA * 1e-8)
    R_phot = t_ph * vphot_cgs
    rmax = t_ph * vmax_cgs
    n_nu = len(nu_arr)
    fnu = np.zeros(n_nu)

    p_arr = np.linspace(0.0, rmax, n_p)
    dp = rmax / (n_p - 1)

    norm_sum = 0.0
    for j in range(n_p):
        p = p_arr[j]
        w = 0.5 if (j == 0 or j == n_p - 1) else 1.0
        I_init = 1.0 if p <= R_phot else 0.0
        norm_sum += I_init * p * w

    if norm_sum <= 0.0:
        return np.ones(n_nu)

    for i in range(n_nu):
        nu = nu_arr[i]
        sum_val = 0.0
        for j in range(n_p):
            p = p_arr[j]
            w = 0.5 if (j == 0 or j == n_p - 1) else 1.0
            z = calc_z_rel(p, nu, nu0, t_ph, c_cgs)
            if not np.isinf(z):
                r = np.sqrt(p ** 2 + z ** 2)
                mu = z / r if r > 0.0 else 0.0

                # LTT 효과
                z_front = np.sqrt(max(0.0, R_phot ** 2 - p ** 2)) if p <= R_phot else 0.0
                d_delay = max(0.0, z_front - z)
                t_eff = max(0.35 * t_ph, t_ph - d_delay / c_cgs)

                # Sobolev 광학두께 프로파일 (Power-law n=7)
                v_local = r / t_eff
                v_phot_local = R_phot / t_eff
                if v_local >= v_phot_local:
                    tau_val = tau_base * (v_local / v_phot_local) ** (-7.0)
                else:
                    tau_val = tau_base

                I_init = 1.0 if p <= R_phot else 0.0
                I_comoving = I_init * np.exp(-tau_val) + 0.5 * (1.0 - np.exp(-tau_val))
                sum_val += I_comoving * ((nu / nu0) ** 3) * p * w

        fnu[i] = sum_val / norm_sum

    return fnu[::-1]