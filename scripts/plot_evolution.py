#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
scripts/plot_evolution.py
킬로노바 AT2017gfo 고유계 광구면 온도(T') 및 방출 계수(trans) 시계열 진화 플롯 모듈
- 영역 음영 구분 없이 순수 데이터 포인트 및 에러바 중심 시각화
"""

import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

TARGET_CASE = "Case1_withLTT_PureLTE"

PHASES = [
    {"label": "+1.43d", "days": 1.427},
    {"label": "+2.42d", "days": 2.417},
    {"label": "+3.41d", "days": 3.413},
    {"label": "+4.40d", "days": 4.403},
]

IDX_T = 0      # T_prime
IDX_TRANS = 5  # trans


def calc_quantiles(arr):
    q16, q50, q84 = np.percentile(arr, [16.0, 50.0, 84.0])
    return q50, q84 - q50, q50 - q16


def power_law(t, t0, alpha):
    return t0 * (t / 1.427) ** (-alpha)


def main():
    base_dir = os.path.join(PROJECT_ROOT, "output_results", TARGET_CASE)
    if not os.path.exists(base_dir):
        print(f"[오류] 결과 디렉토리를 찾을 수 없습니다: {base_dir}")
        return

    days_list = []
    T_med, T_err_p, T_err_m = [], [], []
    trans_med, trans_err_p, trans_err_m = [], [], []

    for p in PHASES:
        d = p["days"]
        lbl = p["label"]
        npy_path = os.path.join(base_dir, f"samples_{d:.3f}d.npy")

        if not os.path.exists(npy_path):
            continue

        chain = np.load(npy_path)
        days_list.append(d)

        # 1. 온도 사후분포
        t_val, t_p, t_m = calc_quantiles(chain[:, IDX_T])
        T_med.append(t_val)
        T_err_p.append(t_p)
        T_err_m.append(t_m)

        # 2. 방출 계수 사후분포
        tr_val, tr_p, tr_m = calc_quantiles(chain[:, IDX_TRANS])
        trans_med.append(tr_val)
        trans_err_p.append(tr_p)
        trans_err_m.append(tr_m)

    days_arr = np.array(days_list)
    T_med_arr = np.array(T_med)
    trans_med_arr = np.array(trans_med)

    # 단열 냉각 멱법칙 피팅
    popt_T, _ = curve_fit(power_law, days_arr, T_med_arr, p0=[4500.0, 0.5])
    t_smooth = np.linspace(1.2, 4.8, 200)
    T_smooth = power_law(t_smooth, *popt_T)

    # -------------------------------------------------------------
    # 1. 발표용 2단 복합 도판 (Time_Evolution_T_and_Trans.png)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8.5), sharex=True)
    plt.subplots_adjust(hspace=0.15)

    # 상단: 온도 (T')
    ax1.plot(t_smooth, T_smooth, color="navy", linestyle="--", lw=1.8,
             label=rf"Adiabatic Cooling Fit: $T^\prime(t) \propto t^{{-{popt_T[1]:.2f}}}$")
    ax1.errorbar(
        days_arr, T_med_arr, yerr=[T_err_m, T_err_p],
        fmt='o-', color="#b30000", ecolor="#b30000", elinewidth=2.2, capsize=5, capthick=1.8,
        markersize=8, lw=1.5, zorder=5, label=r"Rest-frame Photosphere Temperature $T^\prime$"
    )
    for d, t_val in zip(days_arr, T_med_arr):
        ax1.text(d, t_val + 110, f"{t_val:.0f} K", ha="center", fontsize=10.5, fontweight="bold", color="#b30000")

    ax1.set_ylabel(r"Temperature $T^\prime$ [K]", fontsize=12, fontweight="bold")
    ax1.set_title("Photospheric Temperature & Line Transition Factor Evolution", fontsize=13, fontweight="bold", pad=12)
    ax1.set_ylim(2200, 5000)
    ax1.grid(True, linestyle=":", alpha=0.55)
    ax1.legend(loc="upper right", fontsize=10, framealpha=0.92)

    # 하단: trans (음영 영역 없이 클린 플롯)
    ax2.errorbar(
        days_arr, trans_med_arr, yerr=[trans_err_m, trans_err_p],
        fmt='s-', color="#1f77b4", ecolor="#1f77b4", elinewidth=2.2, capsize=5, capthick=1.8,
        markersize=8, lw=2.0, zorder=5, label=r"Line Emission Factor ($\mathrm{trans}$)"
    )
    for d, tr_val in zip(days_arr, trans_med_arr):
        ax2.text(d, tr_val + 0.35, f"{tr_val:.2f}", ha="center", fontsize=10.5, fontweight="bold", color="#1f77b4")

    ax2.set_xlabel("Time post-merger [days]", fontsize=12, fontweight="bold")
    ax2.set_ylabel(r"Emission Factor $\mathrm{trans}$", fontsize=12, fontweight="bold")
    ax2.set_xlim(1.2, 4.8)
    ax2.set_ylim(-0.5, 9.8)
    ax2.grid(True, linestyle=":", alpha=0.55)
    ax2.legend(loc="upper left", fontsize=10, framealpha=0.92)

    out_combined = os.path.join(base_dir, "Time_Evolution_T_and_Trans.png")
    plt.savefig(out_combined, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------
    # 2. 단독 trans 플롯 (Time_Evolution_Trans.png)
    # -------------------------------------------------------------
    fig_tr, ax_tr = plt.subplots(figsize=(8, 5))
    ax_tr.errorbar(
        days_arr, trans_med_arr, yerr=[trans_err_m, trans_err_p],
        fmt='s-', color="#1f77b4", ecolor="#1f77b4", elinewidth=2.2, capsize=5, capthick=1.8,
        markersize=8, lw=2.0, zorder=5, label=r"Line Emission Factor ($\mathrm{trans}$)"
    )
    for d, tr_val in zip(days_arr, trans_med_arr):
        ax_tr.text(d, tr_val + 0.35, f"{tr_val:.2f}", ha="center", fontsize=10.5, fontweight="bold", color="#1f77b4")

    ax_tr.set_xlabel("Time post-merger [days]", fontsize=12, fontweight="bold")
    ax_tr.set_ylabel(r"Line Emission Factor ($\mathrm{trans}$)", fontsize=12, fontweight="bold")
    ax_tr.set_title("AT2017gfo Line Emission Factor ($\mathrm{trans}$) Evolution", fontsize=13, fontweight="bold")
    ax_tr.set_xlim(1.2, 4.8)
    ax_tr.set_ylim(-0.5, 9.8)
    ax_tr.grid(True, linestyle=":", alpha=0.55)
    ax_tr.legend(loc="upper left", fontsize=10.5, framealpha=0.92)

    plt.tight_layout()
    out_tr = os.path.join(base_dir, "Time_Evolution_Trans.png")
    plt.savefig(out_tr, dpi=300, bbox_inches="tight")
    plt.close(fig_tr)

    print("\n[완료] trans 영역 음영이 제거된 클린 도판이 정상적으로 생성되었습니다.")


if __name__ == "__main__":
    main()
