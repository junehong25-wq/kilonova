import numpy as np
import numba
from config.settings import C_CGS

@numba.njit(fastmath=True)
def alpha_tau_evolution(t_days, use_nlte=True):
    if not use_nlte:
        return 1.0
    if 1.5 < t_days <= 3.0:
        return t_days / 1.5
    elif 3.0 < t_days <= 5.0:
        return 2.0 - 0.5 * (t_days - 3.0)
    else:
        return 1.0

@numba.njit(fastmath=True)
def tau_powerlaw_anisotropic(r, mu, t_ph, R_phot, tau_base, beta_power=3.0, c=C_CGS, use_nlte=True):
    if r < R_phot:
        return 0.0

    t_days = t_ph / 86400.0
    alpha_t = alpha_tau_evolution(t_days, use_nlte=use_nlte)
    tau_radial = tau_base * alpha_t * ((r / R_phot) ** (-beta_power))

    beta = (r / t_ph) / c
    if beta >= 1.0:
        return 1e10

    gamma = 1.0 / np.sqrt(1.0 - beta ** 2)
    num = (1.0 - mu * beta) ** 2
    den = gamma * (1.0 - mu * beta - (beta ** 2) * (1.0 - mu ** 2))
    return 1e10 if den <= 0.0 else tau_radial * (num / den)

@numba.njit(fastmath=True)
def calc_z_rel(p, nu, nu0, t_ph, c):
    if nu <= 0.0:
        return np.inf
    A = (nu0 / nu) ** 2
    beta_p = p / (c * t_ph)
    if beta_p >= 1.0:
        return np.inf
    a, b = 1.0 + A, -2.0
    c_coef = 1.0 - A * (1.0 - beta_p ** 2)
    discriminant = b ** 2 - 4.0 * a * c_coef
    if discriminant < 0.0:
        return np.inf
    return ((-b - np.sqrt(discriminant)) / (2.0 * a)) * c * t_ph

@numba.njit(fastmath=True)
def calc_rel_line_profile_with_ltt(nu_arr, lam0_AA, vmax_cgs, vphot_cgs, tau_base, t_ph, c_cgs=C_CGS, n_p=40, use_nlte=True):
    nu0 = c_cgs / (lam0_AA * 1e-8)
    R_phot = t_ph * vphot_cgs
    rmax = t_ph * vmax_cgs
    n_nu = len(nu_arr)
    fnu = np.zeros(n_nu)
    p_arr = np.linspace(0.0, rmax, n_p)
    dp = rmax / (n_p - 1)

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

                d_delay = z if z > 0 else -np.sqrt(np.maximum(0.0, R_phot ** 2 - p ** 2))
                t_det_eff = t_ph + (d_delay / c_cgs)

                tau_val = tau_powerlaw_anisotropic(r, mu, t_det_eff, R_phot, tau_base, beta_power=3.0, c=c_cgs, use_nlte=use_nlte)

                # Smooth handling of I_init across the boundary r = R_phot (or p = R_phot for projection)
                # We use a logistic function to handle the kink smoothly instead of a hard step.
                k = 50.0
                I_init = 1.0 - 1.0 / (1.0 + np.exp(-k * (p / R_phot - 1.0)))

                if r <= R_phot:
                    W = 0.5
                else:
                    mu_phot = np.sqrt(1.0 - (R_phot / r)**2)
                    beta_loc = (r / t_ph) / c_cgs
                    W = 0.5 * (1.0 - (mu_phot - beta_loc) / (1.0 - beta_loc * mu_phot))
                    W = np.maximum(0.0, np.minimum(0.5, W))

                if z < 0.0 and p <= R_phot:
                    I_comoving = 0.0
                else:
                    I_comoving = I_init * np.exp(-tau_val) + (1.0 - np.exp(-tau_val)) * W
                sum_val += I_comoving * (nu / nu0) ** 3 * p * w

        fnu[i] = 2.0 * np.pi * sum_val * dp

    if fnu[0] <= 0.0 or np.isnan(fnu[0]):
        return np.ones(n_nu)
    baseline = fnu[0] * (nu_arr / nu_arr[0]) ** 3
    return (fnu / baseline)[::-1]
