#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
scripts/plot_results.py
킬로노바 AT2017gfo MCMC 피팅 결과 시각화 파이프라인
1. Corner Plot (스무딩 및 1D/2D 정규분포 개선, 경계벽 충돌 완화)
2. Plot1: 전 파장 대역 스펙트럼 피팅 (3,500 ~ 22,000 Å, Telluric Band 음영)
3. Plot2: 흡수/방출선 영역 정규화 라인 프로파일 (7,000 ~ 12,500 Å, Sr II / He I 기준선)
"""

import os
import sys
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
import corner

# 프로젝트 루트 경로 등록
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models import (
    planck_with_mod_full_relativistic,
    calc_combined_pcygni,
    lum_dist_arr,
    C_CGS, H_PLANCK, K_BOLTZ
)

warnings.filterwarnings("ignore")

# 고해상도 출판 품질 플롯 스타일 설정
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10.5,
    "figure.titlesize": 15,
    "mathtext.fontset": "cm",
    "font.family": "serif"
})

CASE_LIST = [
    "Case1_withLTT_PureLTE"
]

TELLURIC_BANDS = [
    (13100.0, 14450.0),
    (17500.0, 19250.0)
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_continuum(wave_AA, popt):
    """흑체 연속광 및 근적외선 가우시안 보정 결합 기저선 산출"""
    lam_cm = wave_AA * 1e-8
    nu = C_CGS / lam_cm
    exp_factor = np.clip((H_PLANCK * nu) / (K_BOLTZ * popt["T_prime"]), 0.0, 700.0)
    b_nu = (2.0 * H_PLANCK * nu ** 3 / C_CGS ** 2) / (np.exp(exp_factor) - 1.0)
    f_cont = b_nu * (C_CGS / lam_cm ** 2) * (popt["N_29"] * 1e-29)

    amp1 = popt.get("amp1", 0.0)
    amp2 = popt.get("amp2", 0.0)
    nir_mod = 1.0 + amp1 * np.exp(-0.5 * ((wave_AA - 15500.0) / 580.0) ** 2) \
              + amp2 * np.exp(-0.5 * ((wave_AA - 20200.0) / 800.0) ** 2)
    return f_cont * nir_mod


def plot_corner_figures(base_dir, results_summary, corner_labels):
    """1. 코너 플롯 생성 (수치 이산화 완화 및 가우시안 윤곽선 최적화)"""
    print("--> [1/3] Corner Plots 생성 중...")
    for res in results_summary:
        days = res["days"]
        samples_file = os.path.join(base_dir, f"samples_{days:.3f}d.npy")
        if not os.path.exists(samples_file):
            print(f"    [건너뜀] 체인 파일 부재: {samples_file}")
            continue

        flat_samples = np.load(samples_file)

        fig = corner.corner(
            flat_samples,
            labels=corner_labels,
            quantiles=[0.16, 0.50, 0.84],
            show_titles=True,
            title_fmt=".3f",
            title_kwargs={"fontsize": 10.5},
            label_kwargs={"fontsize": 11.5},
            smooth=1.2,  # 2D 등고선 미세 격자 노이즈 스무딩
            smooth1d=1.2,  # 1D 히스토그램 스무딩
            plot_datapoints=True,
            plot_density=True,
            fill_contours=True,
            color="#1b365d"
        )

        out_path = os.path.join(base_dir, f"Corner_Phase_{days:.2f}d.png")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"    저장 완료: {os.path.basename(out_path)}")


def plot_full_spectrum_fits(base_dir, spectra_data, results_summary, case_name):
    """2. 전 파장 대역 스펙트럼 피팅 플롯 (Plot1)"""
    print("--> [2/3] Plot1 (Full Spectrum Fits) 생성 중...")
    summary_map = {round(r["days"], 2): r for r in results_summary}

    for sd in spectra_data:
        days = sd["days"]
        label = sd["label"]
        wave = np.array(sd["wave"])
        flux = np.array(sd["flux"])
        x_fit = np.array(sd["x_fit"])
        time_s = days * 86400.0

        res_info = summary_map.get(round(days, 2), {})
        popt = res_info.get("popt", {})
        dl_med = res_info.get("dl_med", 42.0)

        # 모델 핏 및 연속광 기저선 계산
        model_fit = planck_with_mod_full_relativistic(
            x_fit, popt["T_prime"], popt["N_29"], popt["vmax"], popt["vphot"],
            tau=popt["tau"], trans=popt["trans"], ve=popt["ve"],
            amp1=popt["amp1"], amp2=popt["amp2"], t0=time_s
        )
        continuum_curve = compute_continuum(x_fit, popt)

        fig, ax = plt.subplots(figsize=(11, 6), dpi=300)

        # 지구 대기 흡수대 (Telluric Band) 음영 처리
        for idx, (t_low, t_high) in enumerate(TELLURIC_BANDS):
            ax.axvspan(
                t_low, t_high, color="gray", alpha=0.15,
                label="Telluric Band" if idx == 0 else None
            )

        # 관측 데이터 및 모델 플롯
        ax.plot(wave, flux, color="#7f7f7f", lw=0.7, alpha=0.6, label="Observed Data")
        ax.plot(x_fit, continuum_curve, "--", color="#000080", lw=1.8, label="Relativistic Continuum")
        ax.plot(
            x_fit, model_fit, "-", color="#d62728", lw=2.2,
            label=f"+{days:.2f}d Best-Fit ($D_L = {dl_med:.1f}$ Mpc, $T' = {popt['T_prime']:.0f}$ K)"
        )

        ax.set_xlim(3500, 22000)
        ax.set_ylim(bottom=0.0, top=np.max(flux[(wave >= 4000) & (wave <= 20000)]) * 1.15)
        ax.set_xlabel(r"Rest Wavelength [$\mathrm{\AA}$]")
        ax.set_ylabel(r"Flux [$\mathrm{erg/s/cm^2/\AA}$]")
        ax.set_title(f"AT2017gfo Full Spectrum Fit (+{days:.2f}d) [{case_name}]", fontweight="bold", pad=12)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper right", frameon=True, framealpha=0.9)

        plt.tight_layout()
        out_path = os.path.join(base_dir, f"Plot1_Spectrum_Fit_{days:.2f}d.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"    저장 완료: {os.path.basename(out_path)}")


def plot_line_profiles(base_dir, spectra_data, results_summary, case_name):
    """3. Sr II / He I 영역 정규화 라인 프로파일 (Plot2)"""
    print("--> [3/3] Plot2 (Normalized Line Profiles) 생성 중...")
    summary_map = {round(r["days"], 2): r for r in results_summary}

    for sd in spectra_data:
        days = sd["days"]
        wave = np.array(sd["wave"])
        flux = np.array(sd["flux"])
        time_s = days * 86400.0

        res_info = summary_map.get(round(days, 2), {})
        popt = res_info.get("popt", {})
        tau = popt.get("tau", 1.0)
        trans = popt.get("trans", 1.0)
        vmax = popt.get("vmax", 0.35)
        vphot = popt.get("vphot", 0.20)
        ve = popt.get("ve", 0.02)

        # 7,000 ~ 12,500 Å 영역 마스킹
        line_mask = (wave >= 7000.0) & (wave <= 12500.0)
        wave_line = wave[line_mask]
        flux_line = flux[line_mask]

        # 정규화: 연속광 기저선 계산 및 나눗셈
        cont_line = compute_continuum(wave_line, popt)
        norm_data = flux_line / cont_line

        # 모델 라인 프로파일 계산 (Best-fit vs Standard trans=1.0)
        best_profile = calc_combined_pcygni(wave_line, vmax, vphot, tau, trans, ve, time_s)
        standard_profile = calc_combined_pcygni(wave_line, vmax, vphot, tau, 1.00, ve, time_s)

        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

        # 데이터 및 합성선
        ax.plot(wave_line, norm_data, color="#b0b0b0", lw=0.9, alpha=0.7, label="Normalized Data")
        ax.plot(
            wave_line, best_profile, "-", color="#007a78", lw=2.3,
            label=f"Best-fit Profile (trans = {trans:.2f}, $\\tau = {tau:.2f}$)"
        )
        ax.plot(
            wave_line, standard_profile, "--", color="#ff7f0e", lw=1.9,
            label="Standard Profile (trans = 1.0)"
        )

        # 물리적 기준선 표시
        ax.axhline(1.0, linestyle=":", color="black", lw=1.4, label="Normalized Continuum (1.0)")
        ax.axvline(
            10327.31, linestyle="-.", color="#3b4992", lw=1.6, alpha=0.9,
            label=r"Rest Sr II ($10327\,\mathrm{\AA}$)"
        )
        ax.axvline(
            10833.30, linestyle="-.", color="#2e7d32", lw=1.6, alpha=0.9,
            label=r"Rest He I ($10833\,\mathrm{\AA}$)"
        )

        ax.set_xlim(7000, 12500)
        # 상단 방출선 피크 크기에 맞춰 y축 한계 동적 조절
        y_max_bound = max(2.2, np.percentile(norm_data, 99.5) * 1.15, np.max(best_profile) * 1.15)
        ax.set_ylim(0.4, min(y_max_bound, 3.5))

        ax.set_xlabel(r"Rest Wavelength [$\mathrm{\AA}$]")
        ax.set_ylabel(r"Normalized Flux ($F_\lambda / F_{\mathrm{cont}}$)")
        ax.set_title(f"Sr II Region Line Profile (+{days:.2f}d) [{case_name}]", fontweight="bold", pad=12)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper right", frameon=True, framealpha=0.9)

        plt.tight_layout()
        out_path = os.path.join(base_dir, f"Plot2_Line_Profile_{days:.2f}d.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"    저장 완료: {os.path.basename(out_path)}")


def main():
    for target_case in CASE_LIST:
        base_dir = os.path.join(PROJECT_ROOT, "output_results", target_case)
        if not os.path.exists(base_dir):
            print(f"[오류] 해당 결과 디렉터리가 존재하지 않습니다: {base_dir}")
            continue

        print(f"\n========================================================")
        print(f" [시각화 플롯 렌더링 시작: {target_case}]")
        print(f" 대상 디렉터리: {base_dir}")
        print(f"========================================================")

        summary_file = os.path.join(base_dir, "results_summary.json")
        spectra_file = os.path.join(base_dir, "spectra_data.json")
        labels_file = os.path.join(base_dir, "labels_dict.json")

        if not (os.path.exists(summary_file) and os.path.exists(spectra_file)):
            print(f"[오류] 필수 JSON 파일이 존재하지 않습니다. MCMC가 완료되었는지 확인하세요.")
            continue

        results_summary = load_json(summary_file)
        spectra_data = load_json(spectra_file)

        # 라벨 설정 로드
        if os.path.exists(labels_file):
            labels_info = load_json(labels_file)
            corner_labels = labels_info.get("corner_labels", [])
        else:
            corner_labels = [
                r"$T^\prime$", r"$N_{29}$", r"$v_{\max}$", r"$v_{\text{phot}}$",
                r"$\tau$", r"$\text{trans}$", r"$v_e$", r"$\text{amp}_1$", r"$\text{amp}_2$"
            ]

        # 3대 핵심 플롯 순차 생성
        plot_corner_figures(base_dir, results_summary, corner_labels)
        plot_full_spectrum_fits(base_dir, spectra_data, results_summary, target_case)
        plot_line_profiles(base_dir, spectra_data, results_summary, target_case)

        print(f"\n[완료] {target_case}에 대한 모든 시각화 플롯 생성이 완료되었습니다.\n")


if __name__ == "__main__":
    main()