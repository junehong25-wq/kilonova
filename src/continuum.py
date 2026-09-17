import numpy as np
import numba
from config.settings import h_planck, c_speed, k_B

@numba.njit(fastmath=True)
def relativistic_blackbody_flam(wave_m, T_prime, beta, n_mu=16):
    if beta >= 1.0 or beta <= 0.0 or T_prime <= 0.0 or wave_m <= 0.0:
        return 0.0
    gamma = 1.0 / np.sqrt(1.0 - beta ** 2)
    dmu = 1.0 / n_mu
    sum_flux = 0.0
    for i in range(n_mu):
        mu_prime = (i + 0.5) * dmu
        delta = gamma * (1.0 + beta * mu_prime)
        lam_prime = delta * wave_m
        val_exp = (h_planck * c_speed) / (lam_prime * k_B * T_prime)
        b_lam = 0.0 if val_exp > 700.0 else (2.0 * h_planck * c_speed ** 2) / (
                    (lam_prime ** 5) * (np.exp(val_exp) - 1.0))
        sum_flux += (1.0 / (delta ** 3)) * b_lam * (mu_prime + beta) * dmu
    return 2.0 * np.pi * gamma * sum_flux

@numba.njit(fastmath=True)
def calc_relativistic_blackbody_continuum(wave_AA, T_prime, beta, n_mu=16):
    wave_m = wave_AA * 1e-10
    n = len(wave_m)
    flux = np.zeros(n)
    for i in range(n):
        flux[i] = relativistic_blackbody_flam(wave_m[i], T_prime, beta, n_mu)
    return flux
