#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import shutil
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

OUT_DIR = os.path.join(PROJECT_ROOT, "output_results", "GPU_10Epoch_Relativistic_PureLTE_SrII")
SUMMARY_PATH = os.path.join(OUT_DIR, "fit_summary_10epochs.json")
BACKUP_PATH = SUMMARY_PATH + ".bak"

# 1. 기존 10-에포크 요약 JSON 백업
if os.path.exists(SUMMARY_PATH):
    shutil.copy2(SUMMARY_PATH, BACKUP_PATH)

# 2. Pure LTE + Sr II 바인딩 (use_nlte=False 강제)
import config.settings as settings
import src.models as models
import src.probability as probability

_orig_planck_batch = models.planck_with_mod_batch_gpu
_orig_pcygni_batch = models.calc_combined_pcygni_batch_gpu

def _lte_planck_batch(wave_aa_t, theta_2d_np, days, use_nlte=False):
    return _orig_planck_batch(wave_aa_t, theta_2d_np, days, use_nlte=False)

def _lte_pcygni_batch(wave_aa_t, vmax_t, vphot_t, tau_t, trans_t, ve_t, days, use_nlte=False):
    return _orig_pcygni_batch(wave_aa_t, vmax_t, vphot_t, tau_t, trans_t, ve_t, days, use_nlte=False)

models.planck_with_mod_batch_gpu = _lte_planck_batch
probability.planck_with_mod_batch_gpu = _lte_planck_batch
models.calc_combined_pcygni_batch_gpu = _lte_pcygni_batch

# 3. config.settings 내 4.403일차 에포크만 선택하고 ve 상한을 0.22 -> 0.35로 개방
for attr_name in dir(settings):
    val = getattr(settings, attr_name)
    if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict) and "days" in val[0]:
        target_ep = [dict(ep) for ep in val if round(float(ep["days"]), 2) == 4.40]
        if target_ep:
            new_bounds = list(target_ep[0]["bounds"])
            new_bounds[6] = (0.015, 0.220)  # ve 상한 개방
            target_ep[0]["bounds"] = new_bounds
            setattr(settings, attr_name, target_ep)
            print(f"[설정 완료] {attr_name}에서 +4.40d 단독 실행 및 ve bounds={new_bounds[6]} 적용")

# 4. run_mcmc_fit.py 실행 (4.40d 1개 에포크만 GPU MCMC 가동)
target_script = os.path.join(PROJECT_ROOT, "scripts", "run_mcmc_fit.py")
with open(target_script, "r", encoding="utf-8") as f:
    code = f.read()

code = code.replace("GPU_10Epoch_Relativistic_NLTE", "GPU_10Epoch_Relativistic_PureLTE_SrII")
code = code.replace("use_nlte=True", "use_nlte=False")

exec_globals = {"__name__": "__main__", "__file__": target_script}
exec(compile(code, target_script, "exec"), exec_globals)

# 5. 새로 계산된 4.40d 결과를 기존 10-에포크 fit_summary_10epochs.json에 안전하게 병합
if os.path.exists(BACKUP_PATH) and os.path.exists(SUMMARY_PATH):
    with open(BACKUP_PATH, "r", encoding="utf-8") as f1, open(SUMMARY_PATH, "r", encoding="utf-8") as f2:
        full_data = json.load(f1)
        new_data = json.load(f2)

    new_eps = new_data.get("epochs", new_data if isinstance(new_data, list) else [])
    ep_440 = next((e for e in new_eps if round(float(e.get("days", 0)), 2) == 4.40), None)

    if ep_440 is not None:
        if isinstance(full_data, dict) and "epochs" in full_data:
            for idx, e in enumerate(full_data["epochs"]):
                if round(float(e.get("days", 0)), 2) == 4.40:
                    full_data["epochs"][idx] = ep_440
        elif isinstance(full_data, list):
            for idx, e in enumerate(full_data):
                if round(float(e.get("days", 0)), 2) == 4.40:
                    full_data[idx] = ep_440

        with open(SUMMARY_PATH, "w", encoding="utf-8") as f_out:
            json.dump(full_data, f_out, indent=2)
        print("[병합 완료] 기존 9개 에포크 보존 + 4.40d 최신 결과 병합 완료")
