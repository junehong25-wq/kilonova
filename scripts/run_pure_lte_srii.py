#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import src.models as models
import src.probability as probability

# 1. GPU 커널 및 우도 평가 함수 전체에 use_nlte=False 강제 바인딩
_orig_planck_batch = models.planck_with_mod_batch_gpu
_orig_pcygni_batch = models.calc_combined_pcygni_batch_gpu

def _lte_planck_batch(wave_aa_t, theta_2d_np, days, use_nlte=False):
    return _orig_planck_batch(wave_aa_t, theta_2d_np, days, use_nlte=False)

def _lte_pcygni_batch(wave_aa_t, vmax_t, vphot_t, tau_t, trans_t, ve_t, days, use_nlte=False):
    return _orig_pcygni_batch(wave_aa_t, vmax_t, vphot_t, tau_t, trans_t, ve_t, days, use_nlte=False)

models.planck_with_mod_batch_gpu = _lte_planck_batch
probability.planck_with_mod_batch_gpu = _lte_planck_batch
models.calc_combined_pcygni_batch_gpu = _lte_pcygni_batch

# 2. 기존 run_mcmc_fit.py의 저장 경로만 분리하여 동일 MCMC 파이프라인 실행
target_script = os.path.join(PROJECT_ROOT, "scripts", "run_mcmc_fit.py")
with open(target_script, "r", encoding="utf-8") as f:
    code = f.read()

code = code.replace("GPU_10Epoch_Relativistic_NLTE", "GPU_10Epoch_Relativistic_PureLTE_SrII")
code = code.replace("use_nlte=True", "use_nlte=False")

print("=== [대조군 실행] Pure LTE + Sr II 단독 모델 (use_nlte=False, He I 비활성화) ===")
exec_globals = {"__name__": "__main__", "__file__": target_script}
exec(compile(code, target_script, "exec"), exec_globals)
