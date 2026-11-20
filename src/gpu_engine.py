#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
src/gpu_engine.py
NVIDIA RTX 5000 Pro (48GB VRAM) 가속용 PyTorch 3D 텐서 상대론적 복사전달 엔진
- (N_walkers, N_wave, N_quad) 3차원 브로드캐스팅으로 전 워커 동시 계산
- 정확한 EFS 공명 해 (A = (lam / lam_0)^2) 및 Hutsemekers & Surdej (1990) 비등방 광학두께 구현
"""

import torch
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

# 물리 상수 (CGS)
C_CGS = 2.99792458e10
H_CGS = 6.62607015e-27
K_CGS = 1.380649e-16
C1_CGS = 2.0 * H_CGS * (C_CGS ** 2)
C2_CGS = (H_CGS * C_CGS) / K_CGS

# 가우스-르장드르 구적 노드 (GPU 상주 텐서)
_MU_N, _MU_W = np.polynomial.legendre.leggauss(32)
GL_MU_NODES = torch.tensor(0.5 * (_MU_N + 1.0), device=DEVICE, dtype=DTYPE)
GL_MU_WEIGHTS = torch.tensor(0.5 * _MU_W, device=DEVICE, dtype=DTYPE)

_P_N, _P_W = np.polynomial.legendre.leggauss(48)
GL_P_NODES = torch.tensor(0.5 * (_P_N + 1.0), device=DEVICE, dtype=DTYPE)
GL_P_WEIGHTS = torch.tensor(0.5 * _P_W, device=DEVICE, dtype=DTYPE)

# 전이선 파장 및 가중치 (Sr II 삼중항 + He I 10833)
LINE_WAVES = torch.tensor([10036.65, 10327.31, 10914.89, 10833.31], device=DEVICE, dtype=DTYPE)
SR_WEIGHTS = torch.tensor([0.03, 0.94, 0.03], device=DEVICE, dtype=DTYPE)

# 소스 함수 탈출 확률 적분용 고정 코사인 격자
MU_PRIME_GRID = torch.linspace(-1.0, 1.0, 32, device=DEVICE, dtype=DTYPE)


@torch.no_grad()
def calc_relativistic_continuum_gpu(wave_aa_t, T_prime_t, N29_t, vphot_t):
    """
    상대론적 흑체 연속광 계산
    wave_aa_t: (W,)
    T_prime_t, N29_t, vphot_t: (K,) - K개 워커
    반환: (K, W) 플럭스 텐서
    """
    beta = torch.clamp(vphot_t, 1e-4, 0.90)                  # (K,)
    gamma = 1.0 / torch.sqrt(1.0 - beta * beta)              # (K,)

    # 브로드캐스팅 차원 정렬: (K, 1, 1), (1, W, 1), (1, 1, M)
    beta_3d = beta[:, None, None]
    gamma_3d = gamma[:, None, None]
    T_3d = T_prime_t[:, None, None]

    lam_cm = (wave_aa_t * 1e-8)[None, :, None]               # (1, W, 1)
    lam_cm5 = lam_cm ** 5

    mu_p = GL_MU_NODES[None, None, :]                        # (1, 1, M)
    w_p = GL_MU_WEIGHTS[None, None, :]                       # (1, 1, M)

    delta = gamma_3d * (1.0 + beta_3d * mu_p)                # (K, 1, M)
    exponent = torch.clamp(C2_CGS / (delta * lam_cm * T_3d), max=500.0)

    b_lam = (C1_CGS / lam_cm5) / torch.expm1(exponent)       # (K, W, M)
    integrand = (b_lam / (delta ** 3)) * (mu_p + beta_3d)

    integral = torch.sum(w_p * integrand, dim=-1)            # (K, W)

    # N29 스케일링 (단위: 1e-29 sr -> erg/s/cm^2/A)
    front = 2.0 * np.pi * gamma * (N29_t * 1e-29)            # (K,)
    flux_cont = front[:, None] * integral * 1e-8             # (K, W)
    return flux_cont


@torch.no_grad()
def calc_sobolev_pcygni_gpu(wave_aa_t, vmax_t, vphot_t, tau_t, trans_t, ve_t, he_ratio=0.0):
    """
    상대론적 곡면 공명면(Curved Resonance Surface) 및 비등방 광학두께 기반 P-Cygni 프로파일
    반환: (K, W) 정규화 투과/방출 프로파일
    """
    K = vmax_t.shape[0]
    W = wave_aa_t.shape[0]
    P = GL_P_NODES.shape[0]

    # 충격 매개변수 p를 [0, vmax] 전체 구간에서 적분
    p_nodes = GL_P_NODES[None, None, :]                      # (1, 1, P)
    p_weights = GL_P_WEIGHTS[None, None, :]                  # (1, 1, P)

    vmax_3d = vmax_t[:, None, None]                          # (K, 1, 1)
    vphot_3d = vphot_t[:, None, None]                        # (K, 1, 1)
    ve_3d = torch.clamp(ve_t[:, None, None], min=1e-3)

    beta_p = vmax_3d * p_nodes                               # (K, 1, P)
    dp_weight = vmax_3d * p_weights                          # (K, 1, P)
    jac_p = 2.0 * beta_p                                     # (K, 1, P)

    # 광구 원반 내부 여부 (beta_p <= vphot)
    is_core = (beta_p <= vphot_3d)                           # (K, 1, P)
    bz_phot = torch.sqrt(torch.clamp(vphot_3d**2 - beta_p**2, min=0.0))

    # 초기 광선 강도: 원반 내부는 1.0, 외곽 림은 0.0
    i_ray = torch.where(is_core, 1.0, 0.0).expand(K, W, P).clone()

    # 전이선 설정 (Sr II 삼중항 + 후기 He I 혼합)
    if he_ratio > 0.0:
        active_waves = LINE_WAVES
        active_weights = torch.cat([SR_WEIGHTS * (1.0 - he_ratio), torch.tensor([he_ratio], device=DEVICE, dtype=DTYPE)])
    else:
        active_waves = LINE_WAVES[:3]
        active_weights = SR_WEIGHTS

    lam_3d = wave_aa_t[None, :, None]                        # (1, W, 1)

    for il in range(len(active_waves)):
        lam0 = active_waves[il]
        w_line = active_weights[il]
        tau0_k = (tau_t * w_line)[:, None, None]             # (K, 1, 1)

        # 정확한 상대론적 도플러 공명 방정식: A = (nu_0 / nu)^2 = (lam / lam_0)^2
        A = (lam_3d / lam0) ** 2                             # (1, W, 1)
        a_coef = 1.0 + A
        c_coef = 1.0 - A * (1.0 - beta_p ** 2)               # (K, W, P)

        disc = 4.0 - 4.0 * a_coef * c_coef
        valid_disc = (disc >= 0.0)
        sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))

        # 아상대론적 물리 해: 청색편이(A < 1) -> beta_z > 0 (관측자 방향 전면)
        beta_z = (2.0 - sqrt_disc) / (2.0 * a_coef)          # (K, W, P)
        beta_r = torch.sqrt(beta_p ** 2 + beta_z ** 2)       # (K, W, P)

        # 광구 배면 차폐 조건: 원반 내부(beta_p <= vphot)에서 광구 뒤쪽(beta_z < bz_phot)은 가려짐
        not_occulted = (~is_core) | (beta_z >= bz_phot)
        in_shell = valid_disc & not_occulted & (beta_r >= vphot_3d) & (beta_r <= vmax_3d)

        # 부드러운 외곽 경계 테이퍼
        taper_w = 0.01 * (vmax_3d - vphot_3d) + 1e-5
        phase_t = torch.clamp((vmax_3d - beta_r) / taper_w, 0.0, 1.0)
        smooth_edge = torch.where(beta_r > vmax_3d - taper_w, 0.5 * (1.0 - torch.cos(np.pi * phase_t)), 1.0)

        # 지수 감쇠 방사 방향 광학두께
        tau_rad = tau0_k * torch.exp(-(beta_r - vphot_3d) / ve_3d) * smooth_edge

        # Hutsemekers & Surdej (1990) 상대론적 시선각 비등방 인자
        mu = beta_z / torch.clamp(beta_r, min=1e-6)
        denom = torch.clamp((1.0 - beta_r) * (1.0 - mu * beta_r - (beta_r ** 2) * (1.0 - mu ** 2)), min=1e-5)
        aniso_factor = ((1.0 - mu * beta_r) ** 2) / denom
        tau_sob = torch.where(in_shell, torch.clamp(tau_rad * aniso_factor, 0.0, 50.0), 0.0)

        # 기하학적 희석 인자 (Source function S = beta_3 / beta_1 근사 및 광구 입체각 비)
        r_ratio = torch.clamp(vphot_3d / torch.clamp(beta_r, min=1e-5), 0.0, 1.0)
        mu_crit = torch.sqrt(torch.clamp(1.0 - r_ratio ** 2, min=0.0))
        w_geom = 0.5 * (1.0 - mu_crit)                       # (K, W, P)

        # 도플러 부스팅 반영 소스 함수
        gamma_r = 1.0 / torch.sqrt(torch.clamp(1.0 - beta_r ** 2, min=1e-4))
        delta_r = gamma_r * (1.0 + beta_r * mu)
        source_fn = trans_t[:, None, None] * w_geom * (delta_r ** 3)

        exp_neg_tau = torch.exp(-tau_sob)
        i_ray = torch.where(in_shell, i_ray * exp_neg_tau + source_fn * (1.0 - exp_neg_tau), i_ray)

    flux_int = torch.sum(dp_weight * jac_p * i_ray, dim=-1)  # (K, W)
    norm_disk = torch.clamp(vphot_t ** 2, min=1e-10)[:, None]
    profile = flux_int / norm_disk
    return profile


@torch.no_grad()
def synthesize_batch_spectra_gpu(wave_aa_t, theta_batch_np, he_ratio=0.0):
    """
    emcee vectorize=True 호환 배치 스펙트럼 합성기
    theta_batch_np: (K, 9) -> [T', N29, vmax, vphot, tau, trans, ve, amp1, amp2]
    """
    if theta_batch_np.ndim == 1:
        theta_batch_np = theta_batch_np[None, :]

    th = torch.as_tensor(theta_batch_np, device=DEVICE, dtype=DTYPE)
    T_p = th[:, 0]
    N29 = th[:, 1]
    vmax = th[:, 2]
    vphot = th[:, 3]
    tau = th[:, 4]
    trans = th[:, 5]
    ve = th[:, 6]
    amp1 = th[:, 7]
    amp2 = th[:, 8]

    f_cont = calc_relativistic_continuum_gpu(wave_aa_t, T_p, N29, vphot)
    pcyg = calc_sobolev_pcygni_gpu(wave_aa_t, vmax, vphot, tau, trans, ve, he_ratio=he_ratio)

    # 근적외선 가우시안 범프 (15500 A, 20200 A)
    w_row = wave_aa_t[None, :]
    g1 = amp1[:, None] * torch.exp(-0.5 * ((w_row - 15500.0) / 580.0) ** 2)
    g2 = amp2[:, None] * torch.exp(-0.5 * ((w_row - 20200.0) / 800.0) ** 2)

    f_total = f_cont * (pcyg + g1 + g2)
    return f_total, f_cont, pcyg
