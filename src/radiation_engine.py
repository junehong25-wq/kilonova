#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
src/radiation_engine.py
LTT(빛 이동 시간) 및 NLTE 결합 복사전달 엔진 (수치 안정성 및 지연 시간 교정)
"""

import numpy as np
import numba

C_CGS = 29979245800.0


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
    num = (1.0 + mu * beta) ** 2
    den = gamma * (1.0 + mu * beta - (beta ** 2) * (1.0 - mu ** 2))
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
def calc_rel_line_profile_with_ltt(nu_arr, lam0_AA, vmax_cgs, vphot_cgs, tau_base, t_ph, c_cgs=C_CGS, n_p=50,
                                   use_nlte=True):
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

            I_init = 1.0 if p <= R_phot else 0.0
            z = calc_z_rel(p, nu, nu0, t_ph, c_cgs)

            if not np.isinf(z):
                r = np.sqrt(p ** 2 + z ** 2)
                if r <= rmax and r >= R_phot:
                    z_phot_front = np.sqrt(max(0.0, R_phot ** 2 - p ** 2)) if p <= R_phot else 0.0

                    # 광구 후면 완전 차폐 영역 검사 (z < 0 and p <= R_phot)
                    if not (z < 0.0 and p <= R_phot):
                        mu = z / r if r > 0.0 else 0.0
                        # [교정]: 지연 거리를 유효 z 좌표로 정상 계산 (1e30 폭주 원천 차단)
                        d_delay = z - z_phot_front if p <= R_phot else z
                        t_det_eff = max(t_ph * 0.1, t_ph + (d_delay / c_cgs))

                        tau_val = tau_powerlaw_anisotropic(r, mu, t_det_eff, R_phot, tau_base, beta_power=3.0, c=c_cgs,
                                                           use_nlte=use_nlte)

                        mu_phot = np.sqrt(max(0.0, 1.0 - (R_phot / r) ** 2))
                        beta_loc = (r / t_ph) / c_cgs
                        W = 0.5 * (1.0 - (mu_phot - beta_loc) / (1.0 - beta_loc * mu_phot))
                        W = max(0.0, min(1.0, W))

                        # 도플러 인자 및 산란 방출 결합 (인위적 0.5 캡 해제)
                        I_comoving = I_init * np.exp(-tau_val) + (1.0 - np.exp(-tau_val)) * W * 1.5
                    else:
                        I_comoving = 0.0
                else:
                    I_comoving = I_init
            else:
                I_comoving = I_init

            sum_val += I_comoving * p * w

        norm_factor = 0.5 * (R_phot ** 2)
        fnu[i] = (sum_val * dp) / norm_factor if norm_factor > 0.0 else 1.0

    return fnu[::-1]