#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import src.models as models

# 1. 시각화 시 합성 스펙트럼 및 정규화 라인 프로파일도 Pure LTE + Sr II 단독으로 계산되도록 오버라이드
_orig_planck_full = models.planck_with_mod_full_relativistic
_orig_pcygni_single = models.calc_combined_pcygni
_orig_planck_batch = models.planck_with_mod_batch_gpu
_orig_pcygni_batch = models.calc_combined_pcygni_batch_gpu

def _lte_planck_full(*args, **kwargs):
    kwargs["use_nlte"] = False
    return _orig_planck_full(*args, **kwargs)

def _lte_pcygni_single(*args, **kwargs):
    kwargs["use_nlte"] = False
    return _orig_pcygni_single(*args, **kwargs)

def _lte_planck_batch(wave_aa_t, theta_2d_np, days, use_nlte=False):
    return _orig_planck_batch(wave_aa_t, theta_2d_np, days, use_nlte=False)

def _lte_pcygni_batch(wave_aa_t, vmax_t, vphot_t, tau_t, trans_t, ve_t, days, use_nlte=False):
    return _orig_pcygni_batch(wave_aa_t, vmax_t, vphot_t, tau_t, trans_t, ve_t, days, use_nlte=False)

models.planck_with_mod_full_relativistic = _lte_planck_full
models.calc_combined_pcygni = _lte_pcygni_single
models.planck_with_mod_batch_gpu = _lte_planck_batch
models.calc_combined_pcygni_batch_gpu = _lte_pcygni_batch

# 2. 기존 scripts/plot_results.py를 Pure LTE + Sr II 출력 경로로 실행
plot_script = os.path.join(PROJECT_ROOT, "scripts", "plot_results.py")
with open(plot_script, "r", encoding="utf-8") as f:
    code = f.read()

code = code.replace("GPU_10Epoch_Relativistic_NLTE", "GPU_10Epoch_Relativistic_PureLTE_SrII")
code = code.replace("use_nlte=True", "use_nlte=False")

print("=== [1/2] Pure LTE + Sr II 단독 모델 10-에포크 도판 생성 시작 ===")
exec_globals = {"__name__": "__main__", "__file__": plot_script}
exec(compile(code, plot_script, "exec"), exec_globals)

# 3. NLTE (Sr II + He I) vs Pure LTE (Sr II Only) 정량 비교표 및 Ablation 비교 플롯 생성
print("\n=== [2/2] NLTE(Sr II+He I) vs Pure LTE(Sr II Only) 비교 분석 ===")
nlte_dir = os.path.join(PROJECT_ROOT, "output_results", "GPU_10Epoch_Relativistic_NLTE")
lte_dir = os.path.join(PROJECT_ROOT, "output_results", "GPU_10Epoch_Relativistic_PureLTE_SrII")

with open(os.path.join(nlte_dir, "fit_summary_10epochs.json"), "r", encoding="utf-8") as f1, \
     open(os.path.join(lte_dir, "fit_summary_10epochs.json"), "r", encoding="utf-8") as f2:
    d_nlte = json.load(f1)
    d_lte = json.load(f2)

ep_nlte = d_nlte.get("epochs", d_nlte if isinstance(d_nlte, list) else [])
ep_lte = d_lte.get("epochs", d_lte if isinstance(d_lte, list) else [])

days_list, chi2_l, chi2_n, dl_l, dl_n, trans_l, trans_n = [], [], [], [], [], [], []

print(f"{'Phase':<18} | {'Chi2(LTE)':>9} | {'Chi2(NLTE)':>10} | {'trans(LTE)':>10} | {'trans(NLTE)':>11} | {'DL(LTE)':>8} | {'DL(NLTE)':>9}")
print("-" * 92)
for rl, rn in zip(ep_lte, ep_nlte):
    d_val = float(rl.get("days", 0.0))
    lbl = rl.get("label", f"+{d_val:.2f}d")
    cl = float(rl.get("reduced_chi2", 0.0))
    cn = float(rn.get("reduced_chi2", 0.0))
    pl = rl.get("best_fit", rl.get("params", {}))
    pn = rn.get("best_fit", rn.get("params", {}))
    tl = float(pl.get("trans", 0.0))
    tn = float(pn.get("trans", 0.0))
    dll = float(rl.get("luminosity_distance_mpc", {}).get("median", rl.get("D_L", 0.0)))
    dln = float(rn.get("luminosity_distance_mpc", {}).get("median", rn.get("D_L", 0.0)))

    days_list.append(d_val)
    chi2_l.append(cl); chi2_n.append(cn)
    trans_l.append(tl); trans_n.append(tn)
    dl_l.append(dll); dl_n.append(dln)

    print(f"{lbl:<18} | {cl:9.3f} | {cn:10.3f} | {tl:10.2f} | {tn:11.2f} | {dll:8.2f} | {dln:9.2f}")

# 비교 플롯 저장 (Plot_Ablation_NLTE_vs_PureLTE.png)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=300)

axes[0].plot(days_list, chi2_l, "o--", color="#d95f02", lw=2, label="Pure LTE (Sr II Only)")
axes[0].plot(days_list, chi2_n, "s-", color="#1b9e77", lw=2.2, label="Relativistic NLTE (Sr II + He I)")
axes[0].axvline(3.8, color="gray", ls=":", label="He I NLTE Onset (+3.8d)")
axes[0].set_title(r"Fit Goodness ($\mathrm{Red.}\ \chi^2$)", fontweight="bold")
axes[0].set_xlabel("Time post-merger [days]")
axes[0].set_ylabel(r"Reduced $\chi^2$")
axes[0].grid(True, alpha=0.3); axes[0].legend(fontsize=8.5)

axes[1].plot(days_list, trans_l, "o--", color="#d95f02", lw=2, label="Pure LTE (Sr II Only)")
axes[1].plot(days_list, trans_n, "s-", color="#1b9e77", lw=2.2, label="Relativistic NLTE (Sr II + He I)")
axes[1].axvline(3.8, color="gray", ls=":")
axes[1].set_title(r"Emission Modulation ($\mathrm{trans}$)", fontweight="bold")
axes[1].set_xlabel("Time post-merger [days]")
axes[1].set_ylabel(r"$\mathrm{trans}$")
axes[1].grid(True, alpha=0.3); axes[1].legend(fontsize=8.5)

axes[2].axhspan(40.7 - 2.5, 40.7 + 2.5, color="lightgray", alpha=0.5, label="NGC 4993 (40.7 ± 2.5 Mpc)")
axes[2].plot(days_list, dl_l, "o--", color="#d95f02", lw=2, label="Pure LTE (Sr II Only)")
axes[2].plot(days_list, dl_n, "s-", color="#1b9e77", lw=2.2, label="Relativistic NLTE (Sr II + He I)")
axes[2].set_title(r"EPM Distance Stability ($D_L$)", fontweight="bold")
axes[2].set_xlabel("Time post-merger [days]")
axes[2].set_ylabel(r"$D_L$ [Mpc]")
axes[2].grid(True, alpha=0.3); axes[2].legend(fontsize=8.5)

plt.tight_layout()
comp_png = os.path.join(lte_dir, "Plot_Ablation_NLTE_vs_PureLTE.png")
plt.savefig(comp_png, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"\n--> 비교 도판 저장 완료: '{comp_png}'")
