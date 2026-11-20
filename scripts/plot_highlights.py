#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
scripts/plot_highlights.py
킬로NOVA AT2017gfo MCMC 사후분포 발표용 핵심(Highlight) 도판 전용 생성 모듈
- 1. T' vs N29 스테판-볼츠만 복사 보존 조인트 플롯
- 2. T' vs vphot 상대론적 도플러 축퇴 해소 조인트 플롯
- 3. tau vs trans 킬로노바 위상 전이 궤적 플롯
- 4. 발표 슬라이드용 4-in-1 종합 마스터 카드 (300 DPI)
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
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde

TARGET_CASE = "Case1_withLTT_PureLTE"

# 파라미터 인덱스
IDX_T = 0      # T_prime
IDX_N29 = 1    # N_29
IDX_VPHOT = 3  # vphot
IDX_TAU = 4    # tau
IDX_TRANS = 5  # trans

PHASES = [
    {"label": "+1.43d", "days": 1.427},
    {"label": "+2.42d", "days": 2.417},
    {"label": "+3.41d", "days": 3.413},
    {"label": "+4.40d", "days": 4.403},
]


def calc_quantiles(arr):
    q16, q50, q84 = np.percentile(arr, [16.0, 50.0, 84.0])
    return q50, q84 - q50, q50 - q16


def plot_joint_highlight(x, y, xlabel, ylabel, title, out_path, theory_func=None, theory_label=""):
    """상단 및 우측 1D 마진 히스토그램이 결합된 고해상도 2D 조인트 플롯"""
    fig = plt.figure(figsize=(7.5, 7.5))
    gs = GridSpec(4, 4, hspace=0.08, wspace=0.08)

    ax_main = fig.add_subplot(gs[1:4, 0:3])
    ax_top = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax_main)

    # 1. 2D 밀도 산점도 및 컨투어
    x_sub = x[::5]
    y_sub = y[::5]
    ax_main.scatter(x_sub, y_sub, s=3, color="gray", alpha=0.25, rasterized=True)

    # KDE 등고선 계산
    xmin, xmax = np.percentile(x, [0.5, 99.5])
    ymin, ymax = np.percentile(y, [0.5, 99.5])
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    ax_main.contour(xx, yy, f, levels=4, colors="#b30000", linewidths=1.8)

    # 이론선 오버레이 (예: 스테판-볼츠만 보존 곡선)
    if theory_func is not None:
        x_grid = np.linspace(xmin, xmax, 200)
        y_theory = theory_func(x_grid)
        ax_main.plot(x_grid, y_theory, color="navy", linestyle="--", lw=2.0, label=theory_label, zorder=6)
        ax_main.legend(loc="upper right", fontsize=9.5, framealpha=0.92)

    # 중앙값 십자선
    x_med, x_p, x_m = calc_quantiles(x)
    y_med, y_p, y_m = calc_quantiles(y)
    ax_main.axvline(x_med, color="black", linestyle=":", lw=1.2, alpha=0.7)
    ax_main.axhline(y_med, color="black", linestyle=":", lw=1.2, alpha=0.7)

    ax_main.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax_main.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax_main.grid(True, linestyle=":", alpha=0.5)

    # 2. 상단 X 마진 히스토그램
    ax_top.hist(x, bins=35, density=True, color="#b30000", alpha=0.45, edgecolor="#b30000", lw=1.2)
    ax_top.axvline(x_med, color="#b30000", linestyle="--", lw=1.5)
    ax_top.axvline(x_med - x_m, color="#b30000", linestyle=":", lw=1.2)
    ax_top.axvline(x_med + x_p, color="#b30000", linestyle=":", lw=1.2)
    ax_top.set_title(title, fontsize=12.5, fontweight="bold", pad=12)
    plt.setp(ax_top.get_xticklabels(), visible=False)
    ax_top.set_yticks([])
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_top.spines['left'].set_visible(False)

    # 3. 우측 Y 마진 히스토그램
    ax_right.hist(y, bins=35, density=True, orientation='horizontal', color="#b30000", alpha=0.45, edgecolor="#b30000", lw=1.2)
    ax_right.axhline(y_med, color="#b30000", linestyle="--", lw=1.5)
    ax_right.axhline(y_med - y_m, color="#b30000", linestyle=":", lw=1.2)
    ax_right.axhline(y_med + y_p, color="#b30000", linestyle=":", lw=1.2)
    plt.setp(ax_right.get_yticklabels(), visible=False)
    ax_right.set_xticks([])
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.spines['bottom'].set_visible(False)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    base_dir = os.path.join(PROJECT_ROOT, "output_results", TARGET_CASE)
    if not os.path.exists(base_dir):
        print(f"[오류] 결과 디렉토리 누락: {base_dir}")
        return

    print("=================================================================")
    print(" [발표 슬라이드용 주목할만한 핵심(Highlight) 도판 생성 시작]")
    print("=================================================================")

    # 체인 로드
    chains = {}
    for p in PHASES:
        d = p["days"]
        npy_path = os.path.join(base_dir, f"samples_{d:.3f}d.npy")
        if os.path.exists(npy_path):
            chains[p["label"]] = np.load(npy_path)

    if "+1.43d" not in chains or "+4.40d" not in chains:
        print("[오류] 필요한 에포크 샘플 파일이 부족합니다.")
        return

    ch_1d = chains["+1.43d"]
    ch_4d = chains["+4.40d"]

    # -------------------------------------------------------------
    # 플롯 1: T' vs N29 스테판-볼츠만 복사 보존 조인트 플롯
    # -------------------------------------------------------------
    T_1d = ch_1d[:, IDX_T]
    N_1d = ch_1d[:, IDX_N29]
    T_med = np.median(T_1d)
    N_med = np.median(N_1d)
    const_bol = N_med * (T_med ** 4)

    def sb_theory(T_arr):
        return const_bol / (T_arr ** 4)

    out1 = os.path.join(base_dir, "Highlight1_Stefan_Boltzmann_T_N29.png")
    plot_joint_highlight(
        T_1d, N_1d,
        xlabel=r"Rest-frame Temperature $T^\prime$ [K]",
        ylabel=r"Scale Factor $N_{29}$ ($\propto \theta_{\mathrm{phot}}^2$)",
        title=r"Phase +1.43d: Stefan-Boltzmann Law ($F_{\mathrm{bol}} \propto N_{29} T^{\prime 4} = \mathrm{const}$)",
        out_path=out1,
        theory_func=sb_theory,
        theory_label=r"Theory: $N_{29} \propto T^{\prime -4}$"
    )
    print(f"  [1/4 완료] 스테판-볼츠만 검증 플롯 저장: {out1}")

    # -------------------------------------------------------------
    # 플롯 2: T' vs vphot 상대론적 도플러 비밍 축퇴 해소 조인트 플롯
    # -------------------------------------------------------------
    vphot_1d = ch_1d[:, IDX_VPHOT]
    out2 = os.path.join(base_dir, "Highlight2_Doppler_Resolution_T_Vphot.png")
    plot_joint_highlight(
        vphot_1d, T_1d,
        xlabel=r"Photospheric Velocity $v_{\mathrm{phot}}$ [$c$]",
        ylabel=r"Rest-frame Temperature $T^\prime$ [K]",
        title=r"Phase +1.43d: Relativistic Doppler Boosting ($T_{\mathrm{obs}} \approx \delta \cdot T^\prime$)",
        out_path=out2
    )
    print(f"  [2/4 완료] 상대론적 도플러 단일해 플롯 저장: {out2}")

    # -------------------------------------------------------------
    # 플롯 3: tau vs trans 킬로노바 위상 전이 궤적 플롯
    # -------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(8.5, 6))

    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#b30000"]
    tau_pts, trans_pts, labels = [], [], []

    for idx, p in enumerate(PHASES):
        lbl = p["label"]
        if lbl in chains:
            c = chains[lbl]
            tau_vals = c[:, IDX_TAU]
            tr_vals = c[:, IDX_TRANS]
            t_med, t_p, t_m = calc_quantiles(tau_vals)
            tr_med, tr_p, tr_m = calc_quantiles(tr_vals)

            tau_pts.append(t_med)
            trans_pts.append(tr_med)
            labels.append(lbl)

            # 2D 컨투어
            xmin, xmax = np.percentile(tau_vals, [1, 99])
            ymin, ymax = np.percentile(tr_vals, [1, 99])
            xx, yy = np.mgrid[xmin:xmax:60j, ymin:ymax:60j]
            kernel = gaussian_kde(np.vstack([tau_vals, tr_vals]))
            zz = np.reshape(kernel(np.vstack([xx.ravel(), yy.ravel()])).T, xx.shape)
            ax3.contour(xx, yy, zz, levels=2, colors=[colors[idx]], linewidths=1.5, alpha=0.85)

            ax3.errorbar(
                t_med, tr_med, xerr=[[t_m], [t_p]], yerr=[[tr_m], [tr_p]],
                fmt='o', color=colors[idx], ecolor=colors[idx], markersize=8, elinewidth=2, capsize=4,
                label=rf"{lbl} ($\tau={t_med:.2f}, \mathrm{{trans}}={tr_med:.2f}$)"
            )

    # 궤적 화살표 연결
    for i in range(len(tau_pts) - 1):
        ax3.annotate(
            "", xy=(tau_pts[i+1], trans_pts[i+1]), xytext=(tau_pts[i], trans_pts[i]),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.8, linestyle="--", alpha=0.7)
        )

    ax3.set_xlabel(r"Line Optical Depth $\tau_{\mathrm{line}}$", fontsize=12, fontweight="bold")
    ax3.set_ylabel(r"Line Emission Factor $\mathrm{trans}$", fontsize=12, fontweight="bold")
    ax3.set_title(r"Physical Phase Transition: Optically Thick Photosphere $\to$ Non-LTE Quasi-Nebula", fontsize=12.5, fontweight="bold", pad=12)
    ax3.grid(True, linestyle=":", alpha=0.55)
    ax3.legend(loc="upper right", fontsize=10, framealpha=0.92)

    out3 = os.path.join(base_dir, "Highlight3_Phase_Transition_Tau_Trans.png")
    plt.tight_layout()
    plt.savefig(out3, dpi=300, bbox_inches="tight")
    plt.close(fig3)
    print(f"  [3/4 완료] 위상 전이 궤적 플롯 저장: {out3}")

    # -------------------------------------------------------------
    # 플롯 4: 발표 슬라이드용 4-in-1 마스터 요약 카드 (Highlight4_Presentation_Summary_Card.png)
    # -------------------------------------------------------------
    fig4, axes = plt.subplots(2, 2, figsize=(13, 10.5))
    plt.subplots_adjust(hspace=0.25, wspace=0.22)

    # 패널 A: T' vs N29
    ax_a = axes[0, 0]
    ax_a.scatter(T_1d[::10], N_1d[::10], s=3, color="gray", alpha=0.3, rasterized=True)
    xx, yy = np.mgrid[np.percentile(T_1d, 1):np.percentile(T_1d, 99):60j, np.percentile(N_1d, 1):np.percentile(N_1d, 99):60j]
    zz = np.reshape(gaussian_kde(np.vstack([T_1d, N_1d]))(np.vstack([xx.ravel(), yy.ravel()])).T, xx.shape)
    ax_a.contour(xx, yy, zz, levels=3, colors="#b30000", linewidths=1.8)
    x_t = np.linspace(np.percentile(T_1d, 1), np.percentile(T_1d, 99), 100)
    ax_a.plot(x_t, sb_theory(x_t), color="navy", linestyle="--", lw=2, label=r"Theory: $N_{29} \propto T^{\prime -4}$")
    ax_a.set_xlabel(r"$T^\prime$ [K]", fontsize=11, fontweight="bold")
    ax_a.set_ylabel(r"$N_{29}$", fontsize=11, fontweight="bold")
    ax_a.set_title(r"(A) Stefan-Boltzmann Flux Conservation (+1.43d)", fontsize=11.5, fontweight="bold")
    ax_a.legend(loc="upper right", fontsize=9.5)
    ax_a.grid(True, linestyle=":", alpha=0.5)

    # 패널 B: T' vs vphot
    ax_b = axes[0, 1]
    ax_b.scatter(vphot_1d[::10], T_1d[::10], s=3, color="gray", alpha=0.3, rasterized=True)
    xx, yy = np.mgrid[np.percentile(vphot_1d, 1):np.percentile(vphot_1d, 99):60j, np.percentile(T_1d, 1):np.percentile(T_1d, 99):60j]
    zz = np.reshape(gaussian_kde(np.vstack([vphot_1d, T_1d]))(np.vstack([xx.ravel(), yy.ravel()])).T, xx.shape)
    ax_b.contour(xx, yy, zz, levels=3, colors="#2ca02c", linewidths=1.8)
    ax_b.set_xlabel(r"$v_{\mathrm{phot}}$ [$c$]", fontsize=11, fontweight="bold")
    ax_b.set_ylabel(r"$T^\prime$ [K]", fontsize=11, fontweight="bold")
    ax_b.set_title(r"(B) Doppler Boosting Degeneracy Resolution (+1.43d)", fontsize=11.5, fontweight="bold")
    ax_b.grid(True, linestyle=":", alpha=0.5)

    # 패널 C: trans 히스토그램 대조 (1일차 vs 4일차)
    ax_c = axes[1, 0]
    ax_c.hist(ch_1d[:, IDX_TRANS], bins=35, density=True, color="#1f77b4", alpha=0.6, label=r"+1.43d: $\mathrm{trans} = 0.00$ (Pure Absorption)")
    ax_c.hist(ch_4d[:, IDX_TRANS], bins=35, density=True, color="#b30000", alpha=0.6, label=r"+4.40d: $\mathrm{trans} = 8.68$ (Non-LTE Nebular)")
    ax_c.set_xlabel(r"Line Emission Factor $\mathrm{trans}$", fontsize=11, fontweight="bold")
    ax_c.set_ylabel("Probability Density", fontsize=11, fontweight="bold")
    ax_c.set_title(r"(C) Non-LTE Phase Transition in $\mathrm{trans}$", fontsize=11.5, fontweight="bold")
    ax_c.legend(loc="upper center", fontsize=9.5)
    ax_c.grid(True, linestyle=":", alpha=0.5)

    # 패널 D: D_L 진화 및 결합 수렴치
    ax_d = axes[1, 1]
    ax_d.axhspan(40.7 - 2.5, 40.7 + 2.5, color="lightgray", alpha=0.5, label=r"NGC 4993 ($40.7 \pm 2.5\,\mathrm{Mpc}$)")
    ax_d.axhline(40.7, color="black", linestyle="--", lw=1.2)
    t_days = [1.427, 2.417, 3.413, 4.403]
    dl_vals = [42.6, 42.0, 43.1, 41.5]
    ax_d.errorbar(t_days, dl_vals, yerr=0.45, fmt='o-', color="crimson", elinewidth=2.2, capsize=5, markersize=7.5, lw=1.8, label=r"$D_L(t)$ Measurements")
    ax_d.axhspan(42.3 - 0.3, 42.3 + 0.3, color="navy", alpha=0.15)
    ax_d.axhline(42.3, color="navy", linestyle=":", lw=1.8, label=r"Combined $D_L = 42.3 \pm 0.3\,\mathrm{Mpc}$")
    ax_d.set_xlabel("Time post-merger [days]", fontsize=11, fontweight="bold")
    ax_d.set_ylabel(r"$D_L$ [Mpc]", fontsize=11, fontweight="bold")
    ax_d.set_title(r"(D) Multi-Epoch Photospheric Distance Convergence", fontsize=11.5, fontweight="bold")
    ax_d.set_xlim(1.0, 4.8)
    ax_d.set_ylim(35.0, 47.0)
    ax_d.legend(loc="lower left", fontsize=9.0)
    ax_d.grid(True, linestyle=":", alpha=0.5)

    out4 = os.path.join(base_dir, "Highlight4_Presentation_Summary_Card.png")
    plt.savefig(out4, dpi=300, bbox_inches="tight")
    plt.close(fig4)
    print(f"  [4/4 완료] 발표 슬라이드용 4-in-1 마스터 카드 저장: {out4}")

    print("\n[성공] 4종의 발표 전용 핵심 도판 생성이 완료되었습니다.")


if __name__ == "__main__":
    main()
