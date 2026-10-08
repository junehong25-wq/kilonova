import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy import constants
h = constants.h
c = constants.c
k_B = constants.k
C1_N = 2.0 * h / c ** 2
C2_N = h / k_B
C1_L = 2.0 * h * c ** 2
C2_L = (h * c) / k_B
def planck_nu(T, nu):
    exponent = np.clip(C2_N * nu / T, None, 700)
    return (C1_N * nu ** 3) / (np.exp(exponent) - 1.0)
def flux_numerical(lambd_m, T_prime, beta):
    gamma = 1.0 / np.sqrt(1.0 - beta ** 2)
    front = 2.0 * np.pi * gamma
    lambd_array = np.atleast_1d(lambd_m)
    res = np.zeros_like(lambd_array, dtype=float)
    for i, wav in enumerate(lambd_array):
        def integrand(mu_prime):
            delta = gamma * (1.0 + beta * mu_prime)
            exp_val = np.clip(C2_L / (delta * wav * T_prime), None, 700)
            B_lam = (C1_L / wav ** 5) / (np.exp(exp_val) - 1.0)
            return (1.0 / delta ** 3) * B_lam * (mu_prime + beta)
        val, _ = quad(integrand, 0.0, 1.0, epsabs=1e-12, epsrel=1e-4)
        res[i] = front * val
    return res
def flux_analytic_modified_jacobian(lambd_m, T_prime, beta, mu_prime_fixed=0.6):
    gamma = 1.0 / np.sqrt(1.0 - beta ** 2)
    delta_fixed = gamma * (1.0 + beta * mu_prime_fixed)
    front = np.pi * (1.0 + 2.0 * beta) * gamma
    nu_prime = c / (delta_fixed * lambd_m)
    B_nu_val = planck_nu(T_prime, nu_prime)
    jacobian = c / (delta_fixed * lambd_m) ** 2
    return front * jacobian * B_nu_val
def flux_analytic_standard_jacobian(lambd_m, T_prime, beta, mu_prime_fixed=0.6):
    gamma = 1.0 / np.sqrt(1.0 - beta ** 2)
    delta_fixed = gamma * (1.0 + beta * mu_prime_fixed)
    front = np.pi * (1.0 + 2.0 * beta) * gamma
    nu_prime = c / (delta_fixed * lambd_m)
    B_nu_val = planck_nu(T_prime, nu_prime)
    jacobian = c / lambd_m ** 2
    return front * jacobian * B_nu_val
wavelengths_A = np.linspace(2000, 20000, 200)
wavelengths_m = wavelengths_A * 1e-10
T_prime = 5000.0
beta = 0.40
F_num = flux_numerical(wavelengths_m, T_prime, beta)
F_ana_mod = flux_analytic_modified_jacobian(wavelengths_m, T_prime, beta)
F_ana_std = flux_analytic_standard_jacobian(wavelengths_m, T_prime, beta)
err_mod = (F_ana_mod - F_num) / F_num * 100
err_std = (F_ana_std - F_num) / F_num * 100
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), gridspec_kw={'height_ratios': [2.5, 1]}, sharex=True)
ax1.plot(wavelengths_A, F_num, 'b-', linewidth=3, label='1. Numerical Integration')
ax1.plot(wavelengths_A, F_ana_mod, 'orange', linestyle='--', linewidth=2.5,
         label=r'2. Analytic (Jacobian: $c/(\delta\lambda)^2$)')
ax1.plot(wavelengths_A, F_ana_std, 'r-.', linewidth=2.5, label=r'3. Analytic (Jacobian: $c/\lambda^2$)')
ax1.set_title(rf"Relativistic Blackbody ($T' = {T_prime}$ K, $\beta = {beta}$)", fontsize=14)
ax1.set_ylabel(r"Flux $F_\lambda$ (W/m$^3$)", fontsize=12)
ax1.grid(True, alpha=0.4)
ax1.legend(fontsize=11)
ax2.axhline(0, color='gray', linestyle='-', linewidth=1.5)
ax2.plot(wavelengths_A, err_mod, 'orange', linestyle='--', linewidth=2.5, label=r'Error: Jacobian: $c/(\delta\lambda)^2$(%)')
ax2.plot(wavelengths_A, err_std, 'r-.', linewidth=2.5, label=r'Error: Jacobian: $c/\lambda^2$ (%)')
ax2.set_xlabel(r"Wavelength ($\AA$)", fontsize=12)
ax2.set_ylabel("Error vs Correct (%)", fontsize=12)
ax2.grid(True, alpha=0.4)
ax2.legend(fontsize=10, loc='lower right')
ax2.set_ylim(-60, 20)
plt.tight_layout()
plt.savefig('Relativistic_Blackbody.png', dpi=300)
plt.show()
