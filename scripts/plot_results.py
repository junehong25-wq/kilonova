#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
scripts/plot_results.py
10-에포크(+1.43d ~ +10.40d) 스펙트럼 피팅, 정규화 라인 프로파일, 시계열 진화,
9차원 코너 플롯(10장), Stefan-Boltzmann/Doppler 하이라이트, H_0 결론 도판 생성기
"""

import os
import sys
import json
import warnings
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.settings import PHASES_CONFIG, LATEX_LABELS, TELLURIC_BANDS_AA
from src.data_loader import load_clean_spectrum
from src.continuum import calc_relativistic_blackbody_continuum
from src.models import planck_with_mod_full_relativistic, calc_combined_pcygni

warnings.filterwarnings("ignore")


def plot_all_results():
    out_dir = os.path.join(PROJECT_ROOT, "output_results", "GPU_10Epoch_Relativistic_NLTE")
    summary_path = os.path.join(out_dir, "fit_summary_10epochs.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"요약 파일이 없습니다: {summary_path}. 먼저 run_mcmc_fit.py를 실행하세요.")

    with open(summary_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    epochs_res = summary_data["epochs"]
    w_grid = np.linspace(3600.0, 22000.0, 1400)
    w_zoom = np.linspace(7000.0, 12500.0, 700)

    # 1. 초기 4개 에포크(+1.43d ~ +4.40d) 전 파장 스펙트럼 핏 (2x2 그리드)
    print("--> [1/6] 초기 4개 에포크(+1.43d ~ +4.40d) 스펙트럼 플롯 생성 중...", flush=True)
    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 10.5), dpi=300)
    axes1 = axes1.flatten()

    for idx in range(4):
        ep = epochs_res[idx]
        p_cfg = PHASES_CONFIG[idx]
        ax = axes1[idx]
        days = ep["days"]
        popt = ep["popt"]

        local_path = os.path.join(out_dir, ep["filename"])
        _, _, _, w_raw, f_raw = load_clean_spectrum(p_cfg["url"], local_path, subsample_step=6)

        f_cont = (popt["N_29"] * 1e-29) * calc_relativistic_blackbody_continuum(w_grid, popt["T_prime"], popt["vphot"])
        f_mod = planck_with_mod_full_relativistic(
            w_grid, popt["T_prime"], popt["N_29"], popt["vmax"], popt["vphot"],
            tau=popt["tau"], trans=popt["trans"], ve=popt["ve"],
            amp1=popt["amp1"], amp2=popt["amp2"], t0=days * 86400.0
        )

        for (t_low, t_high) in TELLURIC_BANDS_AA:
            ax.axvspan(t_low, t_high, color="lightgray", alpha=0.45, label="Telluric Band" if idx == 0 else None)

        ax.plot(w_raw, f_raw, color="silver", lw=0.7, label="Observed Data")
        ax.plot(w_grid, f_cont, color="navy", ls="--", lw=1.5, label="Relativistic Continuum")
        ax.plot(w_grid, f_mod, color="#D81B60", lw=2.2,
                label=f"+{days:.2f}d Best-Fit ($D_L = {ep['dl_med']:.1f}$ Mpc, $T' = {popt['T_prime']:.0f}$ K)")

        ax.set_title(f"AT2017gfo Full Spectrum Fit (+{days:.2f}d)", fontsize=12, fontweight="bold")
        ax.set_xlim(3600, 22000)
        ax.set_ylim(0.0, np.percentile(f_raw[(w_raw > 4000) & (w_raw < 18000)], 99.5) * 1.15)
        ax.set_xlabel(r"Rest Wavelength [$\mathrm{\AA}$]")
        ax.set_ylabel(r"Flux [$\mathrm{erg\ s^{-1}\ cm^{-2}\ \AA^{-1}}$]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Plot1A_Spectrum_Fit_Early_1to4d.png"), dpi=300)
    plt.close(fig1)

    # 2. 후기 6개 에포크(+5.40d ~ +10.40d) 전 파장 스펙트럼 핏 (3x2 그리드)
    print("--> [2/6] 후기 6개 에포크(+5.40d ~ +10.40d) 준성운/성운 스펙트럼 플롯 생성 중...", flush=True)
    fig2, axes2 = plt.subplots(3, 2, figsize=(16, 15), dpi=300)
    axes2 = axes2.flatten()

    for k, idx in enumerate(range(4, 10)):
        ep = epochs_res[idx]
        p_cfg = PHASES_CONFIG[idx]
        ax = axes2[k]
        days = ep["days"]
        popt = ep["popt"]

        local_path = os.path.join(out_dir, ep["filename"])
        _, _, _, w_raw, f_raw = load_clean_spectrum(p_cfg["url"], local_path, subsample_step=6)

        f_cont = (popt["N_29"] * 1e-29) * calc_relativistic_blackbody_continuum(w_grid, popt["T_prime"], popt["vphot"])
        f_mod = planck_with_mod_full_relativistic(
            w_grid, popt["T_prime"], popt["N_29"], popt["vmax"], popt["vphot"],
            tau=popt["tau"], trans=popt["trans"], ve=popt["ve"],
            amp1=popt["amp1"], amp2=popt["amp2"], t0=days * 86400.0
        )

        for (t_low, t_high) in TELLURIC_BANDS_AA:
            ax.axvspan(t_low, t_high, color="lightgray", alpha=0.45, label="Telluric Band" if k == 0 else None)

        ax.plot(w_raw, f_raw, color="silver", lw=0.7, label="Observed Data")
        ax.plot(w_grid, f_cont, color="navy", ls="--", lw=1.5, label="Quasi-Nebular Continuum")
        ax.plot(w_grid, f_mod, color="#8E24AA", lw=2.2,
                label=f"+{days:.2f}d NLTE Fit ($T' = {popt['T_prime']:.0f}$ K, trans = {popt['trans']:.1f})")

        ax.set_title(f"AT2017gfo Late-Time Spectrum Fit (+{days:.2f}d) [NLTE He I + Nebular]", fontsize=12, fontweight="bold")
        ax.set_xlim(3600, 22000)
        ax.set_ylim(0.0, np.percentile(f_raw[(w_raw > 4500) & (w_raw < 21000)], 99.2) * 1.18)
        ax.set_xlabel(r"Rest Wavelength [$\mathrm{\AA}$]")
        ax.set_ylabel(r"Flux [$\mathrm{erg\ s^{-1}\ cm^{-2}\ \AA^{-1}}$]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Plot1B_Spectrum_Fit_Late_5to10d.png"), dpi=300)
    plt.close(fig2)

    # 3. 10개 전 에포크 정규화 라인 프로파일 (5x2 그리드)
    print("--> [3/6] 10개 전 에포크 정규화 Sr II / He I 라인 프로파일 생성 중...", flush=True)
    fig_lp, axes_lp = plt.subplots(5, 2, figsize=(16, 22), dpi=260)
    axes_lp = axes_lp.flatten()

    for idx, ep in enumerate(epochs_res):
        p_cfg = PHASES_CONFIG[idx]
        ax = axes_lp[idx]
        days = ep["days"]
        popt = ep["popt"]

        local_path = os.path.join(out_dir, ep["filename"])
        _, _, _, w_raw, f_raw = load_clean_spectrum(p_cfg["url"], local_path, subsample_step=6)
        m_z = (w_raw >= 7000.0) & (w_raw <= 12500.0)
        w_sub, f_sub = w_raw[m_z], f_raw[m_z]

        c_sub = (popt["N_29"] * 1e-29) * calc_relativistic_blackbody_continuum(w_sub, popt["T_prime"], popt["vphot"])
        obs_norm = f_sub / np.maximum(c_sub, 1e-35)

        pcyg_best = calc_combined_pcygni(w_zoom, popt["vmax"], popt["vphot"], popt["tau"], popt["trans"], popt["ve"], days * 86400.0)
        pcyg_std = calc_combined_pcygni(w_zoom, popt["vmax"], popt["vphot"], popt["tau"], 1.0, popt["ve"], days * 86400.0)

        ax.plot(w_sub, obs_norm, color="lightgray", lw=0.7, label="Normalized Data")
        ax.plot(w_zoom, pcyg_best, color="teal", lw=2.3, label=f"Best-fit (trans={popt['trans']:.2f}, $\\tau$={popt['tau']:.2f})")
        ax.plot(w_zoom, pcyg_std, color="darkorange", ls="--", lw=1.7, label="Standard (trans=1.0)")
        ax.axhline(1.0, color="black", ls=":", lw=1.1)
        ax.axvline(10327.31, color="darkslateblue", ls="-.", lw=1.2, label=r"Sr II $10327\,\mathrm{\AA}$")
        ax.axvline(10833.31, color="forestgreen", ls="-.", lw=1.2, label=r"He I $10833\,\mathrm{\AA}$")

        ax.set_title(f"Line Profile (+{days:.2f}d)", fontsize=11, fontweight="bold")
        ax.set_xlim(7000, 12500)
        ax.set_ylim(0.3, max(2.2, np.percentile(pcyg_best, 99) * 1.15))
        ax.set_xlabel(r"Rest Wavelength [$\mathrm{\AA}$]")
        ax.set_ylabel(r"$F_\lambda / F_{\mathrm{cont}}$")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Plot2_Normalized_Line_Profiles_10Epochs.png"), dpi=260)
    plt.close(fig_lp)

    # 4. 10-에포크 물리 파라미터 시계열 진화 (T', vphot/vmax, trans)
    print("--> [4/6] 10-에포크 물리 파라미터 시계열 진화 플롯 생성 중...", flush=True)
    days_list = np.array([ep["days"] for ep in epochs_res])
    T_list = np.array([ep["popt"]["T_prime"] for ep in epochs_res])
    vp_list = np.array([ep["popt"]["vphot"] for ep in epochs_res])
    vm_list = np.array([ep["popt"]["vmax"] for ep in epochs_res])
    tr_list = np.array([ep["popt"]["trans"] for ep in epochs_res])

    fig_ev, (ax_t, ax_v, ax_tr) = plt.subplots(3, 1, figsize=(10, 12), dpi=300, sharex=True)

    ax_t.plot(days_list, T_list, "o-", color="#C62828", lw=2.2, markersize=7, label=r"Rest-frame Temperature $T'(t)$")
    t_dense = np.linspace(1.3, 10.6, 200)
    ax_t.plot(t_dense, 4564.0 * ((t_dense / 1.427) ** (-0.51)), "b--", lw=1.6, label=r"Cooling Power-law: $T'(t) \propto t^{-0.51}$")
    ax_t.axvline(3.8, color="gray", ls=":", lw=1.5, label="Quasi-Nebular / NLTE He I Transition")
    ax_t.set_ylabel(r"Temperature $T'$ [K]", fontweight="bold")
    ax_t.set_title("10-Epoch Evolution: Temperature, Ejecta Velocities & Emission Factor (+1.43d to +10.40d)", fontweight="bold")
    ax_t.grid(True, ls=":", alpha=0.4)
    ax_t.legend()

    ax_v.plot(days_list, vm_list, "s--", color="navy", lw=1.8, markersize=6, label=r"Maximum Ejecta Velocity $v_{\max}(t)$")
    ax_v.plot(days_list, vp_list, "o-", color="teal", lw=2.2, markersize=7, label=r"Line-Forming / Photospheric Velocity $v_{\mathrm{phot}}(t)$")
    ax_v.axvline(3.8, color="gray", ls=":", lw=1.5)
    ax_v.set_ylabel(r"Velocity [$c$]", fontweight="bold")
    ax_v.grid(True, ls=":", alpha=0.4)
    ax_v.legend()

    ax_tr.plot(days_list, tr_list, "D-", color="#1565C0", lw=2.2, markersize=7, label=r"Line Emission Factor $\mathrm{trans}(t)$")
    ax_tr.axvline(3.8, color="gray", ls=":", lw=1.5)
    ax_tr.set_xlabel("Time post-merger [days]", fontweight="bold")
    ax_tr.set_ylabel("Emission Factor trans", fontweight="bold")
    ax_tr.grid(True, ls=":", alpha=0.4)
    ax_tr.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Plot3_Evolution_10Epochs.png"), dpi=300)
    plt.close(fig_ev)

    # 5. 각 에포크별 9차원 코너 플롯 생성
    print("--> [5/6] 10개 에포크 개별 9차원 코너 플롯 생성 중...", flush=True)
    for ep in epochs_res:
        days = ep["days"]
        s_file = os.path.join(out_dir, f"samples_{days:.2f}d.npy")
        if not os.path.exists(s_file):
            continue
        fc = np.load(s_file)
        fig_c = corner.corner(
            fc, labels=LATEX_LABELS, quantiles=[0.16, 0.50, 0.84],
            show_titles=True, title_fmt=".3f", smooth=1.0, color="#1b365d"
        )
        fig_c.savefig(os.path.join(out_dir, f"Corner_Phase_{days:.2f}d.png"), dpi=200, bbox_inches="tight")
        plt.close(fig_c)

    # 6. 광도거리 D_L 및 허블 상수 H_0 최종 결론 도판
    print("--> [6/6] EPM 광도거리 및 허블 상수 H_0 결론 도판 생성 중...", flush=True)
    fig_con, (ax_dl, ax_h0) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    phot_eps = [ep for ep in epochs_res if ep["is_photospheric"]]
    p_days = [ep["days"] for ep in phot_eps]
    p_dls = [ep["dl_med"] for ep in phot_eps]
    p_errs = [0.5 * (ep["dl_err_minus"] + ep["dl_err_plus"]) for ep in phot_eps]

    dl_c = summary_data["Combined_Photospheric_DL_Mpc"]["median"]
    dl_ce = 0.5 * (summary_data["Combined_Photospheric_DL_Mpc"]["err_minus"] + summary_data["Combined_Photospheric_DL_Mpc"]["err_plus"])

    ax_dl.axhspan(40.7 - 2.5, 40.7 + 2.5, color="lightgray", alpha=0.5, label=r"Host Galaxy NGC 4993 ($40.7 \pm 2.5$ Mpc)")
    ax_dl.axhspan(dl_c - dl_ce, dl_c + dl_ce, color="lavender", alpha=0.9)
    ax_dl.axhline(dl_c, color="navy", ls=":", lw=2.0, label=rf"Combined EPM Distance: ${dl_c:.1f} \pm {dl_ce:.1f}$ Mpc")
    ax_dl.errorbar(p_days, p_dls, yerr=p_errs, fmt="o-", color="#D81B60", lw=2.2, markersize=8, capsize=5,
                   label=r"Relativistic EPM Distance $D_L(t)$")
    ax_dl.set_xlim(1.0, 5.0)
    ax_dl.set_ylim(34, 48)
    ax_dl.set_xlabel("Time post-merger [days]", fontweight="bold")
    ax_dl.set_ylabel(r"Luminosity Distance $D_L$ [Mpc]", fontweight="bold")
    ax_dl.set_title("EPM Photospheric Distance Evolution", fontweight="bold")
    ax_dl.grid(True, ls=":", alpha=0.4)
    ax_dl.legend(loc="upper left", fontsize=9)

    h0_c = summary_data["Combined_H0_km_s_Mpc"]["median"]
    h0_m = summary_data["Combined_H0_km_s_Mpc"]["err_minus"]
    h0_p = summary_data["Combined_H0_km_s_Mpc"]["err_plus"]

    labels_h0 = ["Planck (2018, CMB)", "SH0ES (2022, Cepheid)", "GW170817 (Standard Siren)", "Sneppen+ (2023, A&A)", "This Work (Combined)"]
    vals_h0 = [67.4, 73.0, 70.0, 67.0, h0_c]
    elo = [0.5, 1.0, 8.0, 3.6, h0_m]
    ehi = [0.5, 1.0, 12.0, 3.6, h0_p]
    cols = ["#1976D2", "#D32F2F", "#388E3C", "#6D4C41", "#B71C1C"]

    ax_h0.axvspan(67.4 - 0.5, 67.4 + 0.5, color="#1976D2", alpha=0.2, label="Planck 2018")
    ax_h0.axvspan(73.0 - 1.0, 73.0 + 1.0, color="#D32F2F", alpha=0.2, label="SH0ES 2022")
    for i in range(len(labels_h0)):
        ax_h0.errorbar(vals_h0[i], i, xerr=[[elo[i]], [ehi[i]]], fmt="D" if i == 4 else "o", color=cols[i], lw=2.2, capsize=4)
        ax_h0.text(vals_h0[i], i + 0.2, f"{vals_h0[i]:.1f} (-{elo[i]:.1f}/+{ehi[i]:.1f})", ha="center", color=cols[i], fontweight="bold", fontsize=9)

    ax_h0.set_yticks(range(len(labels_h0)))
    ax_h0.set_yticklabels(labels_h0)
    ax_h0.set_xlim(55, 90)
    ax_h0.set_xlabel(r"Hubble Constant $H_0\ [\mathrm{km\ s^{-1}\ Mpc^{-1}}]$", fontweight="bold")
    ax_h0.set_title(r"$H_0$ Measurement Comparison", fontweight="bold")
    ax_h0.grid(True, ls=":", alpha=0.4)
    ax_h0.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Plot5_Conclusion_DL_and_H0.png"), dpi=300)
    plt.close(fig_con)

    print(f"\n--> 모든 10-에포크 도판 생성이 완료되었습니다! 경로: '{out_dir}'", flush=True)


if __name__ == "__main__":
    plot_all_results()
