#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
src/continuum.py
KPS 원본 SI 단위계 플랑크 연속광 수식 + RTX 5000 Pro FP32/FP64 혼합 정밀도 & torch.compile 가속
"""

import numpy as np
import torch
from config.settings import H_PLANCK_SI, C_SPEED_SI, K_BOLTZ_SI

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COMPUTE_DTYPE = torch.float32
ACCUM_DTYPE = torch.float64
DTYPE = COMPUTE_DTYPE  # 하위 모듈 호환용 기본 연산 타입

if hasattr(torch, "_dynamo"):
    torch._dynamo.config.suppress_errors = True


def maybe_compile(fn):
    if not torch.cuda.is_available():
        return fn
    try:
        return torch.compile(fn, mode="default", dynamic=True)
    except Exception:
        return fn


_MU_N, _MU_W = np.polynomial.legendre.leggauss(32)
GL_MU_NODES = torch.tensor(0.5 * (_MU_N + 1.0), device=DEVICE, dtype=COMPUTE_DTYPE)
GL_MU_WEIGHTS = torch.tensor(0.5 * _MU_W, device=DEVICE, dtype=COMPUTE_DTYPE)

C1_SI = float(2.0 * H_PLANCK_SI * (C_SPEED_SI ** 2))
C2_SI = float((H_PLANCK_SI * C_SPEED_SI) / K_BOLTZ_SI)


@maybe_compile
def _blackbody_kernel(wave_aa_t, T_prime_t, beta_t):
    beta = torch.clamp(beta_t, 1e-5, 0.92)
    gamma = 1.0 / torch.sqrt(1.0 - beta * beta)

    beta_3d = beta[:, None, None]
    gamma_3d = gamma[:, None, None]
    T_3d = torch.clamp(T_prime_t[:, None, None], min=500.0)

    lam_m = (wave_aa_t * 1e-10)[None, :, None]
    lam_m5 = lam_m ** 5

    mu_p = GL_MU_NODES[None, None, :]
    w_p = GL_MU_WEIGHTS[None, None, :]

    delta = gamma_3d * (1.0 + beta_3d * mu_p)
    # float32 지수 오버플로(exp(88.7)) 방지를 위해 상한 80.0 클램핑
    exponent = torch.clamp(C2_SI / (delta * lam_m * T_3d), max=80.0)

    b_lam = (C1_SI / lam_m5) / torch.expm1(exponent)
    integrand = (b_lam / (delta ** 3)) * (mu_p + beta_3d)
    integral = torch.sum(w_p * integrand, dim=-1)

    front = (2.0 * np.pi) * gamma
    return front[:, None] * integral


@torch.no_grad()
def calc_relativistic_blackbody_batch_gpu(wave_aa_t, T_prime_t, beta_t):
    return _blackbody_kernel(wave_aa_t, T_prime_t, beta_t)


def calc_relativistic_blackbody_continuum(wave_AA, T_prime, beta):
    w_t = torch.as_tensor(wave_AA, device=DEVICE, dtype=COMPUTE_DTYPE)
    t_t = torch.tensor([float(T_prime)], device=DEVICE, dtype=COMPUTE_DTYPE)
    b_t = torch.tensor([float(beta)], device=DEVICE, dtype=COMPUTE_DTYPE)
    res = calc_relativistic_blackbody_batch_gpu(w_t, t_t, b_t)
    return res[0].cpu().numpy().astype(np.float64)
