#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
src/probability.py
고정 배치 크기(64) 전달로 torch.compile 재컴파일을 차단하고,
FP32 복사전달 연산 후 FP64 정밀도로 chi^2를 합산하는 순수 데이터 우도 모듈
"""

import numpy as np
import torch
from src.continuum import DEVICE, COMPUTE_DTYPE, ACCUM_DTYPE
from src.models import planck_with_mod_batch_gpu


class MCMCProbabilityWrapper(object):
    def __init__(self, wave, flux, err, days, bounds, is_photospheric=True):
        self.wave_np = np.asarray(wave, dtype=np.float64)
        self.flux_np = np.asarray(flux, dtype=np.float64)
        self.err_np = np.asarray(err, dtype=np.float64)

        self.wave_t = torch.tensor(self.wave_np, device=DEVICE, dtype=COMPUTE_DTYPE)
        self.flux_t = torch.tensor(self.flux_np, device=DEVICE, dtype=COMPUTE_DTYPE)
        self.err_t = torch.tensor(self.err_np, device=DEVICE, dtype=COMPUTE_DTYPE)

        self.days = float(days)
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.low_b = self.bounds[:, 0]
        self.high_b = self.bounds[:, 1]
        self.spans = self.high_b - self.low_b
        self.is_photospheric = bool(is_photospheric)

    def evaluate_batch(self, theta_2d):
        # 1. Flat Prior & 유체역학적 조건(v_phot < v_max - 0.02)
        in_bounds = np.all((theta_2d >= self.low_b[None, :]) & (theta_2d <= self.high_b[None, :]), axis=1)
        valid_vel = theta_2d[:, 3] < (theta_2d[:, 2] - 0.020)
        valid_mask = in_bounds & valid_vel

        if not np.any(valid_mask):
            return np.full(theta_2d.shape[0], -np.inf, dtype=np.float64)

        # 2. 고정 배치 크기 유지를 위해 유효하지 않은 행은 안전 중심값으로 대체 후 전체 배치를 GPU에 전달
        safe_theta = np.where(valid_mask[:, None], theta_2d, 0.5 * (self.low_b + self.high_b)[None, :])
        mod_t, _, _ = planck_with_mod_batch_gpu(self.wave_t, safe_theta, self.days, use_nlte=True)

        # 3. FP64 정밀도 업캐스팅 후 카이제곱 합산
        diff = (self.flux_t[None, :] - mod_t) / self.err_t[None, :]
        chi2 = torch.sum(diff.to(ACCUM_DTYPE) ** 2, dim=1).cpu().numpy()

        ll = -0.5 * chi2
        return np.where(valid_mask & np.isfinite(ll), ll, -np.inf)

    def __call__(self, theta):
        theta_arr = np.asarray(theta, dtype=np.float64)
        if theta_arr.ndim == 1:
            return float(self.evaluate_batch(theta_arr[None, :])[0])
        return self.evaluate_batch(theta_arr)

    def chi2_for_minimizer(self, theta_1d):
        val = self.__call__(theta_1d)
        if not np.isfinite(val):
            return 1e30
        return -2.0 * val
