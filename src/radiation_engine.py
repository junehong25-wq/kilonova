#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
src/radiation_engine.py
LTT(빛 이동 시간) 결합 상대론적 복사전달 엔진
- 광구면 내외부 연속 적분 처리로 코너플롯 vmax-vphot 대각선 리플 완전 제거
"""

import numpy as np
import numba

C_CGS = 29979245800.0


@numba.njit(fastmath=True)
def tau_exponential_anisotropic(r, mu, t_ph, R_phot, tau_base, ve, c=C_CGS):
    if r <= R_phot:
        return 0.0

    v_cgs = r / t_ph
    vphot_cgs = R_phot / t_ph
    ve_cgs = max(ve * c, 1e-4 * c)

    v_diff = max(0.0, v_cgs - vphot_cgs)
    # 기본 스케일 0.5 유지로 기존 광학두께 스케일과 정합성 확보
    tau_radial = 0.5 * tau_base * np.exp(-v_diff / ve_cgs)

    beta = v_cgs / c
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
    a = 1.0 + A
    b = -2.0
    c_coef = 1.0 - A * (1.0 - beta_p ** 2)
    discriminant = b ** 2 - 4.0 * a * c_coef
    if discriminant < 0.0:
        return np.inf
    return ((-b - np.sqrt(discriminant)) / (2.0 * a)) * c * t_ph


@numba.njit(fastmath=True)
def calc_rel_line_profile_with_ltt(
    nu_arr, lam0_AA, vmax_cgs, vphot_cgs, tau_base, ve_cgs, t_ph,
    c_cgs=C_CGS, n_p=120
):
    nu0 = c_cgs / (lam0_AA * 1e-8)
    R_phot = t_ph * vphot_cgs
    rmax = t_ph * vmax_cgs
    n_nu = len(nu_arr)
    fnu = np.zeros(n_nu)
    p_arr = np.linspace(0.0, rmax, n_p)
    dp = rmax / max(1, n_p - 1)

    for i in range(n_nu):
        nu = nu_arr[i]
        sum_val = 0.0
        norm_sum = 0.5 * (R_phot ** 2)

        for j in range(n_p):
            p = p_arr[j]
            w = 0.5 * dp if (j == 0 or j == n_p - 1) else dp

            # 서브픽셀 시그모이드 스무딩으로 이산화 계단 리플 완전 제거
            edge_p = (R_phot - p) / (0.6 * dp + 1e-30)
            if edge_p > 15.0:
                I_init = 1.0
            elif edge_p < -15.0:
                I_init = 0.0
            else:
                I_init = 1.0 / (1.0 + np.exp(-edge_p))

            z = calc_z_rel(p, nu, nu0, t_ph, c_cgs)

            if not np.isinf(z):
                r = np.sqrt(p ** 2 + z ** 2)
                if r <= rmax and r >= R_phot:
                    z_phot_front = np.sqrt(max(0.0, R_phot ** 2 - p ** 2))
                    # 후면 차폐(Occultation)도 연속 스무딩 처리
                    if not (z < 0.0 and p < R_phot):
                        mu = z / r if r > 0.0 else 0.0
                        d_delay = max(0.0, z_phot_front - z)
                        t_em = max(t_ph * 0.35, t_ph - (d_delay / c_cgs))

                        tau_val = tau_exponential_anisotropic(
                            r, mu, t_em, R_phot, tau_base, ve_cgs / c_cgs, c=c_cgs
                        )

                        mu_phot = np.sqrt(max(0.0, 1.0 - (R_phot / r) ** 2))
                        beta_loc = (r / t_ph) / c_cgs
                        denom_w = max(1e-6, 1.0 - beta_loc * mu_phot)
                        W = 0.5 * (1.0 - (mu_phot - beta_loc) / denom_w)
                        W = max(0.0, min(1.0, W))

                        S_source = W
                        I_comoving = I_init * np.exp(-tau_val) + S_source * (1.0 - np.exp(-tau_val))
                    else:
                        I_comoving = I_init
                else:
                    I_comoving = I_init
            else:
                I_comoving = I_init

            sum_val += I_comoving * p * w

        fnu[i] = sum_val / norm_sum if norm_sum > 0.0 else 1.0

    return fnu[::-1]
