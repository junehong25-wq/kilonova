import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy import constants
h = constants.h
c = constants.c
k_B = constants.k
C1 = 2.0 * h * c ** 2
C2 = (h * c) / k_B
def flux_numerical(lambd, T_prime, beta):
    gamma = 1.0 / np.sqrt(1.0 - beta ** 2)
    front_term = 2.0 * np.pi * gamma
    lambd_array = np.atleast_1d(lambd)
    flux_result = np.zeros_like(lambd_array, dtype=float)
    for i, wav in enumerate(lambd_array):
        def integrand(mu_prime):
            delta = gamma * (1.0 + beta * mu_prime)
            exponent = np.clip(C2 / (delta * wav * T_prime), None, 700)
            planck_raw = (C1 / wav ** 5) / (np.exp(exponent) - 1.0)
            return (1.0 / delta ** 3) * planck_raw * (mu_prime + beta)
        val, _ = quad(integrand, 0.0, 1.0, epsabs=1e-12, epsrel=1e-4)
        flux_result[i] = front_term * val
    return flux_result
def flux_analytic(lambd, T_prime, beta, mu_fixed=0.6):
    gamma = 1.0 / np.sqrt(1.0 - beta ** 2)
    delta_fixed = gamma * (1.0 + beta * mu_fixed)
    front_term = np.pi * (1.0 + 2.0 * beta) * gamma
    exponent = np.clip(C2 / (delta_fixed * lambd * T_prime), None, 700)
    planck_raw = (C1 / lambd ** 5) / (np.exp(exponent) - 1.0)
    return front_term * (1.0 / delta_fixed ** 3) * planck_raw
def flux_rest(lambd, T_prime):
    return flux_numerical(lambd, T_prime, 0.0)
wavelengths_A = np.linspace(2000, 20000, 200)
wavelengths_m = wavelengths_A * 1e-10
T_prime = 5000.0
beta = 0.40
F_num = flux_numerical(wavelengths_m, T_prime, beta)
F_ana = flux_analytic(wavelengths_m, T_prime, beta)
F_rest = flux_rest(wavelengths_m, T_prime)
plt.figure(figsize=(10, 6), dpi=150)
plt.plot(wavelengths_A, F_num, 'b-', linewidth=3, label='Numerical Integration')
plt.plot(wavelengths_A, F_ana, 'orange', linestyle='--', linewidth=3, label=r'Analytic Approx. ($\mu^\prime = 0.6$ fixed)')
plt.plot(wavelengths_A, F_rest, 'g:', linewidth=3, label=r'Rest Frame Blackbody ($\beta = 0$)')
plt.title(rf"Wavelength Integration ($T' = {T_prime:.0f}$K, $\beta = {beta:.2f}$)", fontsize=16)
plt.xlabel(r"Wavelength ($\AA$)", fontsize=14)
plt.ylabel(r"Flux $F_\lambda$ (Arbitrary Units)", fontsize=14)
plt.grid(True, alpha=0.4)
plt.legend(fontsize=12, loc='upper right')
plt.tight_layout()
plt.savefig('Wavelength Integration.png', dpi=300)
plt.show()
