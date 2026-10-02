#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
src/continuum.py
상대론적 팽창 흑체 단색 표면 복사 플럭스 수치 적분 엔진
"""

import numpy as np
import numba
from scipy import constants

h_planck = constants.h
c_speed = constants.c
k_B = constants.k


@numba.njit(fastmath=True)
def relativistic_blackbody_flam(wave_m, T_prime, beta, n_mu=32):
    """
    고유계 온도 T'와 팽창 속도 beta = v/c에 대한 상대론적 단색 표면 플럭스 F_lambda 수치 적분 (W/m^3)
    논문 식 (43) 기반: F_lambda = 2*pi*gamma * int_0^1 (1/delta^3) * B_lambda(delta*lambda, T') * (mu' + beta) dmu'
    """
    if beta >= 1.0 or beta < 0.0 or T_prime <= 0.0 or wave_m <= 0.0:
        return 0.0

    if beta == 0.0:
        val_exp = (h_planck * c_speed) / (wave_m * k_B * T_prime)
        if val_exp > 700.0:
            return 0.0
        return (np.pi * (2.0 * h_planck * c_speed**2)) / ((wave_m**5) * (np.exp(val_exp) - 1.0))

    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    dmu = 1.0 / n_mu
    sum_flux = 0.0

    for i in range(n_mu):
        mu_prime = (i + 0.5) * dmu
        delta = gamma * (1.0 + beta * mu_prime)
        val_exp = (h_planck * c_speed) / (delta * wave_m * k_B * T_prime)
        b_lam = 0.0 if val_exp > 700.0 else (2.0 * h_planck * c_speed**2) / ((wave_m**5) * (np.exp(val_exp) - 1.0))
        sum_flux += (1.0 / (delta**3)) * b_lam * (mu_prime + beta) * dmu

    return 2.0 * np.pi * gamma * sum_flux


@numba.njit(fastmath=True)
def calc_relativistic_blackbody_continuum(wave_AA, T_prime, beta, n_mu=32):
    """파장 배열(Angstrom)을 입력받아 단색 복사 플럭스 배열(W/m^3)을 반환"""
    wave_m = wave_AA * 1e-10
    n = len(wave_m)
    flux = np.zeros(n)
    for i in range(n):
        flux[i] = relativistic_blackbody_flam(wave_m[i], T_prime, beta, n_mu)
    return flux