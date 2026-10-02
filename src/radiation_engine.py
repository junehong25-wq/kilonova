#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
src/radiation_engine.py
상대론적 Sobolev 복사전달 엔진 (EATS 시공간 LTT 및 Comoving Frame 해석해)
- 적색편이 영역(z < 0) 광구 앞면 연속광(I=1.0) 보존 로직 적용
- 장파장 끝단에서 연속광(1.0)으로의 매끄러운 수렴 보장
"""

import numpy as np

# 물리 상수
C_KMS = 299792.458  # 광속 [km/s]


def solve_resonant_z_vectorized(p_arr, lam_ratio, c_kms, t0_s):
    """
    상대론적 등진동수 공명면 z 좌표 계산 (벡터 연산)
    lam_ratio = lambda_obs / lambda_0
    """
    A = lam_ratio ** 2
    r_scale = c_kms * t0_s
    beta_p = p_arr / r_scale

    # 2차 방정식 판별식: disc = 1 - (1+A)*[1 - A*(1 - beta_p^2)]
    term = 1.0 - A * (1.0 - beta_p ** 2)
    disc = 1.0 - (1.0 + A) * term

    valid_mask = disc >= 0.0
    z_arr = np.full_like(p_arr, np.nan)
    beta_z_arr = np.full_like(p_arr, np.nan)
    beta_arr = np.full_like(p_arr, np.nan)

    if np.any(valid_mask):
        p_val = p_arr[valid_mask]
        bp_val = beta_p[valid_mask]
        disc_val = disc[valid_mask]

        # 물리적 해 (부호: -)
        bz = (1.0 - np.sqrt(disc_val)) / (1.0 + A)
        b_sq = bp_val ** 2 + bz ** 2
        phys_mask = b_sq < 1.0

        if np.any(phys_mask):
            idx = np.where(valid_mask)[0][phys_mask]
            bz_phys = bz[phys_mask]
            beta_z_arr[idx] = bz_phys
            beta_arr[idx] = np.sqrt(b_sq[phys_mask])
            z_arr[idx] = bz_phys * r_scale

    return z_arr, beta_z_arr, beta_arr


def calc_rel_line_profile_with_ltt(
    wave_obs,
    lam0=10400.0,
    vphot=0.22,
    vmax=0.35,
    tau_base=1.5,
    ve=0.35,
    t0=1.5 * 86400.0,
    use_ltt=True,
    use_nlte=False,
    n_p=90
):
    """
    상대론적 P-Cygni 1D 정규화 선윤곽 f(lambda) 계산
    반환값: f(lambda) >= 1.0 (적색편이 영역), f(lambda) -> 1.0 (최대 속도 밖)
    """
    wave_obs = np.asarray(wave_obs, dtype=np.float64)
    f_prof = np.ones_like(wave_obs)

    r_phot = vphot * C_KMS * t0
    r_max = vmax * C_KMS * t0

    # 상대론적 도플러 최소/최대 파장 범위 계산
    doppler_min = np.sqrt((1.0 - vmax) / (1.0 + vmax))
    doppler_max = np.sqrt((1.0 + vmax) / (1.0 - vmax))
    lam_min = lam0 * doppler_min * 0.98
    lam_max = lam0 * doppler_max * 1.02

    # 시선 충돌 매개변수 p 그리드
    p_grid = np.linspace(0.0, r_max, n_p)
    dp = p_grid[1] - p_grid[0]
    norm_factor = 0.5 * (r_phot ** 2)

    for i, lam in enumerate(wave_obs):
        if lam < lam_min or lam > lam_max:
            f_prof[i] = 1.0
            continue

        lam_ratio = lam / lam0
        z_arr, beta_z_arr, beta_arr = solve_resonant_z_vectorized(p_grid, lam_ratio, C_KMS, t0)

        # 각 p 그리드별 복사 강도 I_comoving 계산
        I_comoving = np.zeros(n_p, dtype=np.float64)

        for j in range(n_p):
            p = p_grid[j]
            z = z_arr[j]
            beta = beta_arr[j]

            # 1. 공명점이 존재하지 않거나 최대 분출 반경 바깥인 경우
            if np.isnan(z):
                if p <= r_phot:
                    I_comoving[j] = 1.0  # 광구 흑체 연속광 방출
                else:
                    I_comoving[j] = 0.0
                continue

            r = np.sqrt(p ** 2 + z ** 2)
            if r > r_max:
                if p <= r_phot:
                    I_comoving[j] = 1.0
                else:
                    I_comoving[j] = 0.0
                continue

            # 2. 광학적 깊이 tau 계산 (속도 구배 지수함수형 감쇄)
            v = r / (C_KMS * t0)
            tau_radial = tau_base * np.exp(-max(0.0, v - vphot) / max(ve, 1e-4))

            mu = z / r if r > 0 else 0.0
            gamma = 1.0 / np.sqrt(max(1e-6, 1.0 - beta ** 2))
            denom = gamma * (1.0 + mu * beta - (beta ** 2) * (1.0 - mu ** 2))
            denom = max(denom, 1e-6)
            geom_factor = ((1.0 + mu * beta) ** 2) / denom
            tau_val = tau_radial * geom_factor

            # 3. 원천함수 S 계산 (EATS 시간 지연 및 냉각 보정)
            if r <= r_phot:
                W = 1.0
            else:
                mu_phot = np.sqrt(max(0.0, 1.0 - (r_phot / r) ** 2))
                denom_w = max(1e-6, 1.0 - beta * mu_phot)
                W = 0.5 * (1.0 - (mu_phot - beta) / denom_w)
                W = np.clip(W, 0.0, 1.0)

            if use_ltt:
                z_front = np.sqrt(max(0.0, r_phot ** 2 - p ** 2)) if p <= r_phot else 0.0
                t_em = max(0.1 * t0, t0 - (z_front - z) / C_KMS)
                t_ratio = (t0 / t_em) ** 0.5
            else:
                t_ratio = 1.0

            S_source = W * t_ratio

            # NLTE 형광/비열적 펌핑 방출 보정
            if use_nlte:
                S_source *= (1.0 + 0.6 * np.exp(-max(0.0, v - vphot) / max(ve, 1e-4)))

            # 4. 복사전달 방정식 형식해 적용 (핵심 수정 구역)
            if p <= r_phot:
                z_front = np.sqrt(r_phot ** 2 - p ** 2)
                if z >= z_front:
                    # [구역 3: 광구 앞면 대기] -> 흡수 및 재방출
                    I_comoving[j] = 1.0 * np.exp(-tau_val) + S_source * (1.0 - np.exp(-tau_val))
                else:
                    # [구역 1 & 2: 광구 뒤편 대기 및 앞면 표면]
                    # 뒤쪽 대기 방출은 광구에 차폐(0.0), 앞면 연속광은 흡수선 없이 100% 통과(1.0)
                    I_comoving[j] = 1.0
            else:
                # [구역 B: 광구 바깥쪽 껍질] -> 순수 방출
                I_comoving[j] = S_source * (1.0 - np.exp(-tau_val))

        # 충돌 매개변수 평면 면적분
        flux_sum = np.sum(I_comoving * p_grid) * dp
        f_prof[i] = flux_sum / norm_factor

    return f_prof