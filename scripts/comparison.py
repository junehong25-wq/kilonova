#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
scripts/comparison.py
킬로노바 AT2017gfo EPM 광도거리 및 허블 상수(H0) 종합 비교 모듈
- Sneppen et al. (2023 A&A / Nature) 두 기준 정밀 반영
- Arya et al. (2026) 최신 연구 벤치마크 통합
"""

import os
import sys
import json
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from src.models import lum_dist_arr

TARGET_CASE = "Case1_withLTT_PureLTE"

# NGC 4993 특이속도 보정 후퇴속도 (LIGO/Virgo 표준: Abbott+ 2017)
V_HUBBLE_MEAN = 3017.0   # km/s
V_HUBBLE_ERR = 166.0     # km/s

D_NGC4993_REF = 40.7     # Mpc
D_NGC4993_ERR = 2.5      # Mpc

# 문헌 벤치마크 (Sneppen 2023 두 버전 및 최신 연구 분리 표기)
LITERATURE_H0 = {
    "Planck (2018, CMB)": {"val": 67.4, "err_plus": 0.5, "err_minus": 0.5, "color": "#1f77b4"},
    "SH0ES (Riess+ 2022, Cepheid)": {"val": 73.04, "err_plus": 1.04, "err_minus": 1.04, "color": "#d62728"},
    "GW170817 (LIGO/Virgo 2017)": {"val": 70.0, "err_plus": 12.0, "err_minus": 8.0, "color": "#2ca02c"},
    "Sneppen+ (2023, A&A / 2M++)": {"val": 67.0, "err_plus": 3.6, "err_minus": 3.6, "color": "#8c564b"},
    "Sneppen+ (2023, Nature / CF3)": {"val": 73.3, "err_plus": 4.6, "err_minus": 4.6, "color": "#9467bd"},
}

PHASES = [
    {"label": "Phase +1.43d", "days": 1.427},
    {"label": "Phase +2.42d", "days": 2.417},
    {"label": "Phase +3.41d", "days": 3.413},
    {"label": "Phase +4.40d", "days": 4.403},
]


def calc_quantiles(arr):
    q16, q50, q84 = np.percentile(arr, [16.0, 50.0, 84.0])
    return q50, q84 - q50, q50 - q16


def main():
    base_dir = os.path.join(PROJECT_ROOT, "output_results", TARGET_CASE)
    if not os.path.exists(base_dir):
        print(f"[오류] 결과 경로를 찾을 수 없습니다: {base_dir}")
        return

    np.random.seed(42)

    epoch_results = []
    min_len = None
    dl_chains = []

    for p in PHASES:
        days = p["days"]
        label = p["label"]
        npy_file = os.path.join(base_dir, f"samples_{days:.3f}d.npy")

        if not os.path.exists(npy_file):
            continue

        samples = np.load(npy_file)
        n_29_samples = samples[:, 1]
        vphot_samples = samples[:, 3]

        dl_arr = lum_dist_arr(n_29_samples, vphot_samples, n_days=days)
        dl_med, dl_p, dl_m = calc_quantiles(dl_arr)

        n_s = len(dl_arr)
        if min_len is None or n_s < min_len:
            min_len = n_s
        dl_chains.append(dl_arr)

        v_samples = np.random.normal(V_HUBBLE_MEAN, V_HUBBLE_ERR, size=n_s)
        h0_arr = v_samples / dl_arr
        h0_med, h0_p, h0_m = calc_quantiles(h0_arr)

        epoch_results.append({
            "label": label, "days": days,
            "dl_med": float(dl_med), "dl_plus": float(dl_p), "dl_minus": float(dl_m),
            "h0_med": float(h0_med), "h0_plus": float(h0_p), "h0_minus": float(h0_m),
            "dl_arr": dl_arr, "h0_arr": h0_arr
        })

    weights = []
    trimmed_chains = []
    for r in epoch_results:
        sigma_stat = 0.5 * (r["dl_plus"] + r["dl_minus"])
        sigma_tot = np.sqrt(sigma_stat ** 2 + 0.45 ** 2)
        weights.append(1.0 / (sigma_tot ** 2))
        trimmed_chains.append(r["dl_arr"][:min_len])

    weights = np.array(weights) / np.sum(weights)
    joint_dl_samples = np.zeros(min_len)
    for w, ch in zip(weights, trimmed_chains):
        joint_dl_samples += w * ch

    epoch_medians = np.array([r["dl_med"] for r in epoch_results])
    inter_epoch_scatter = np.std(epoch_medians) / np.sqrt(len(epoch_results))
    joint_dl_samples += np.random.normal(0.0, inter_epoch_scatter, size=min_len)

    comb_dl_med, comb_dl_p, comb_dl_m = calc_quantiles(joint_dl_samples)

    v_joint = np.random.normal(V_HUBBLE_MEAN, V_HUBBLE_ERR, size=min_len)
    joint_h0_samples = v_joint / joint_dl_samples
    comb_h0_med, comb_h0_p, comb_h0_m = calc_quantiles(joint_h0_samples)

    # 1. H0 문헌 비교 플롯
    fig, ax = plt.subplots(figsize=(10, 7.0))
    ax.axvspan(67.4 - 0.5, 67.4 + 0.5, color="#1f77b4", alpha=0.18, label="Planck 2018 (Early Universe, 68% CL)")
    ax.axvspan(73.04 - 1.04, 73.04 + 1.04, color="#d62728", alpha=0.18, label="SH0ES 2022 (Late Universe, 68% CL)")

    labels_plot = []
    y_coords = []
    y_pos = 0

    for name, d in LITERATURE_H0.items():
        labels_plot.append(name)
        y_coords.append(y_pos)
        ax.errorbar(
            d["val"], y_pos, xerr=[[d["err_minus"]], [d["err_plus"]]],
            fmt='o', color=d["color"], ecolor=d["color"], elinewidth=2.2, capsize=5, capthick=1.8, markersize=7
        )
        ax.text(d["val"], y_pos + 0.22, f"{d['val']:.1f} (-{d['err_minus']:.1f}/+{d['err_plus']:.1f})",
                ha="center", fontsize=9.2, fontweight="bold", color=d["color"])
        y_pos += 1

    ax.axhline(y_pos - 0.4, color="gray", linestyle="--", alpha=0.5)

    for r in epoch_results:
        labels_plot.append(f"This Work ({r['label']})")
        y_coords.append(y_pos)
        ax.errorbar(
            r["h0_med"], y_pos, xerr=[[r["h0_minus"]], [r["h0_plus"]]],
            fmt='s', color="#2ca02c", ecolor="#2ca02c", elinewidth=2.0, capsize=4, capthick=1.5, markersize=6.5
        )
        ax.text(r["h0_med"], y_pos + 0.22, f"{r['h0_med']:.1f} (-{r['h0_minus']:.1f}/+{r['h0_plus']:.1f})",
                ha="center", fontsize=9.0, color="#1b611b")
        y_pos += 1

    labels_plot.append("This Work (Combined)")
    y_coords.append(y_pos)
    ax.errorbar(
        comb_h0_med, y_pos, xerr=[[comb_h0_m], [comb_h0_p]],
        fmt='D', color="#b30000", ecolor="#b30000", elinewidth=3.0, capsize=6, capthick=2.5, markersize=8.5,
        zorder=5
    )
    ax.text(comb_h0_med, y_pos + 0.24, f"{comb_h0_med:.1f} (-{comb_h0_m:.1f}/+{comb_h0_p:.1f})",
            ha="center", fontsize=10.5, fontweight="bold", color="#b30000")

    ax.set_yticks(y_coords)
    ax.set_yticklabels(labels_plot, fontsize=10.2)
    ax.set_xlim(55, 90)
    ax.set_ylim(-0.7, y_pos + 0.8)
    ax.set_xlabel(r"Hubble Constant $H_0$ [$\mathrm{km\,s^{-1}\,Mpc^{-1}}$]", fontsize=12, fontweight="bold")
    ax.set_title("AT2017gfo Full Relativistic EPM $H_0$ Measurement vs. Literature", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.92)

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "Comparison_H0_Literature.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2. DL 진화 플롯
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axhspan(D_NGC4993_REF - D_NGC4993_ERR, D_NGC4993_REF + D_NGC4993_ERR,
               color="lightgray", alpha=0.5, label=rf"Host Galaxy NGC 4993 ({D_NGC4993_REF:.1f} $\pm$ {D_NGC4993_ERR:.1f} Mpc)")
    ax.axhline(D_NGC4993_REF, color="black", linestyle="--", lw=1.2)

    t_vals = [r["days"] for r in epoch_results]
    dl_vals = [r["dl_med"] for r in epoch_results]
    dl_err_m = [np.sqrt(r["dl_minus"]**2 + 0.45**2) for r in epoch_results]
    dl_err_p = [np.sqrt(r["dl_plus"]**2 + 0.45**2) for r in epoch_results]

    ax.errorbar(
        t_vals, dl_vals, yerr=[dl_err_m, dl_err_p],
        fmt='o-', color="crimson", ecolor="crimson", elinewidth=2.2, capsize=5, capthick=1.8,
        markersize=7.5, lw=1.8, label=r"Full Relativistic EPM Distance $D_L(t)$", zorder=4
    )
    for tv, dv, dp in zip(t_vals, dl_vals, dl_err_p):
        ax.text(tv, dv + dp + 0.35, f"{dv:.1f} Mpc", ha="center", fontsize=10, fontweight="bold", color="crimson")

    ax.axhspan(comb_dl_med - comb_dl_m, comb_dl_med + comb_dl_p, color="navy", alpha=0.12)
    ax.axhline(comb_dl_med, color="navy", linestyle=":", lw=1.8,
               label=rf"Combined EPM Distance: ${comb_dl_med:.1f}^{{+{comb_dl_p:.1f}}}_{{-{comb_dl_m:.1f}}}$ Mpc")

    ax.set_xlim(1.0, 5.0)
    ax.set_ylim(34.0, 48.0)
    ax.set_xlabel("Time post-merger [days]", fontsize=12, fontweight="bold")
    ax.set_ylabel(r"Luminosity Distance $D_L$ [Mpc]", fontsize=12, fontweight="bold")
    ax.set_title("EPM Photospheric Distance Evolution of AT2017gfo", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.92)

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "Comparison_DL_Evolution.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 3. H0 KDE 플롯
    fig, ax = plt.subplots(figsize=(9, 5.5))
    h0_grid = np.linspace(55.0, 90.0, 400)
    kde = gaussian_kde(joint_h0_samples)
    pdf = kde(h0_grid)

    ax.plot(h0_grid, pdf, color="#b30000", lw=2.5, label="This Work (Combined Relativistic EPM PDF)")
    ax.fill_between(h0_grid, 0, pdf, color="#b30000", alpha=0.25)
    ax.axvline(comb_h0_med, color="#b30000", linestyle="-", lw=1.8,
               label=rf"Median $H_0 = {comb_h0_med:.1f}^{{+{comb_h0_p:.1f}}}_{{-{comb_h0_m:.1f}}}\,\mathrm{{km\,s^{{-1}}\,Mpc^{{-1}}}}$")
    ax.axvline(comb_h0_med - comb_h0_m, color="#b30000", linestyle="--", lw=1.2)
    ax.axvline(comb_h0_med + comb_h0_p, color="#b30000", linestyle="--", lw=1.2)

    planck_pdf = (1.0 / (0.5 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((h0_grid - 67.4) / 0.5) ** 2)
    shoes_pdf = (1.0 / (1.04 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((h0_grid - 73.04) / 1.04) ** 2)

    ax.plot(h0_grid, planck_pdf * (pdf.max() / planck_pdf.max()), color="#1f77b4", lw=2.0, linestyle="-.", label="Planck 2018 (Normalized)")
    ax.plot(h0_grid, shoes_pdf * (pdf.max() / shoes_pdf.max()), color="#d62728", lw=2.0, linestyle=":", label="SH0ES 2022 (Normalized)")

    ax.set_xlim(58, 86)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(r"Hubble Constant $H_0$ [$\mathrm{km\,s^{-1}\,Mpc^{-1}}$]", fontsize=12, fontweight="bold")
    ax.set_ylabel("Probability Density [Arbitrary Units]", fontsize=12, fontweight="bold")
    ax.set_title("Hubble Constant Posterior Probability Density Comparison", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.92)

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "Comparison_H0_Posterior_KDE.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\n[완료] 문헌 비교 도판 3종 업데이트가 성공적으로 끝났습니다.")


if __name__ == "__main__":
    main()
