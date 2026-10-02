#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
scripts/plot_results.py
MCMC 결과 시각화 스크립트 (전 파장 대역 3,500 ~ 22,000 Å 및 LTT 2개 케이스 전용)
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

from src.models import (
    planck_with_mod_full_relativistic,
    calc_relativistic_blackbody_continuum
)

try:
    import corner
except ImportError:
    corner = None

# 분석 대상 케이스 지정 (불필요한 구버전 케이스 자동 렌더링 방지)
TARGET_CASES = [
    "Case1_withLTT_PureLTE",
    "Case2_withLTT_NLTE"
]


def plot_all_results():
    base_dir = os.path.join(PROJECT_ROOT, "output_results")
    if not os.path.exists(base_dir):
        print(f"[경고] 결과 폴더를 찾을 수 없습니다: {base_dir}")
        return

    for case_id in TARGET_CASES:
        target_save_dir = os.path.join(base_dir, case_id)
        res_json = os.path.join(target_save_dir, 'results_summary.json')
        spec_json = os.path.join(target_save_dir, 'spectra_data.json')
        lbl_json = os.path.join(target_save_dir, 'labels_dict.json')

        if not os.path.exists(res_json):
            print(f"[건너뜀] 결과 파일 없음: {case_id}")
            continue

        use_nlte = "NLTE" in case_id and "PureLTE" not in case_id
        print(f"\n--> [{case_id}] 플롯 렌더링 시작 (LTT=True, NLTE={use_nlte})...")

        with open(res_json, 'r') as f:
            results_summary = json.load(f)
        with open(spec_json, 'r') as f:
            spectra_data = json.load(f)
        with open(lbl_json, 'r') as f:
            labels_dict = json.load(f)

        # -------------------------------------------------------------
        # 1. 코너 플롯 (파라미터 사후분포)
        # -------------------------------------------------------------
        for sdata in spectra_data:
            days = sdata["days"]
            samples_path = os.path.join(target_save_dir, f"samples_{days:.3f}d.npy")
            if os.path.exists(samples_path) and corner is not None:
                samples = np.load(samples_path)
                try:
                    safe_ranges = []
                    for col_idx in range(samples.shape[1]):
                        col_data = samples[:, col_idx]
                        ptp_val = np.ptp(col_data)
                        med_val = np.median(col_data)
                        if ptp_val < 1e-5:
                            span_pad = max(abs(med_val) * 0.05, 0.01)
                            safe_ranges.append((med_val - span_pad, med_val + span_pad))
                        else:
                            safe_ranges.append(0.999)

                    fig = corner.corner(
                        samples, labels=labels_dict.get("corner_labels", None),
                        range=safe_ranges,
                        quantiles=[0.16, 0.50, 0.84], show_titles=True, title_fmt=".3f",
                        plot_contours=False
                    )
                    fig.savefig(os.path.join(target_save_dir, f"Corner_Phase_{days:.2f}d.png"), dpi=200)
                    plt.close(fig)
                    print(f"    [완료] Corner_Phase_{days:.2f}d.png")
                except Exception as e:
                    print(f"    [참고] {days:.2f}d 코너 플롯 생성 생략 ({e})")

        # -------------------------------------------------------------
        # 2. 스펙트럼 핏 (전 파장 대역) & 라인 프로파일 핏 (Sr II)
        # -------------------------------------------------------------
        for idx, sdata in enumerate(spectra_data):
            days = sdata["days"]
            popt = results_summary[idx]["popt"]
            wave = np.array(sdata["wave"])
            flux = np.array(sdata["flux"])
            t_ph = days * 86400.0

            tau_val = float(popt.get("tau", popt.get("tau_sr", 1.5)))
            ve_val = float(popt.get("ve", 0.35))
            trans_val = float(popt.get("trans", 1.0))
            amp1_val = float(popt.get("amp1", 0.20))
            amp2_val = float(popt.get("amp2", 0.40))
            dl_val = float(results_summary[idx].get("dl_med", 40.0))

            # (1) 전 파장 대역(3,500 ~ 22,000 Å) 스펙트럼 핏
            # 관측 데이터 및 모델 플럭스 계산
            w_grid = np.linspace(3500.0, 22000.0, 1500)
            model_flux = planck_with_mod_full_relativistic(
                w_grid, popt["T_prime"], popt["N_29"], popt["vmax"], popt["vphot"],
                tau=tau_val, trans=trans_val, ve=ve_val,
                amp1=amp1_val, amp2=amp2_val, t0=t_ph,
                use_ltt=True, use_nlte=use_nlte
            )

            # 기저 상대론적 흑체 연속광
            cont_flux = (popt["N_29"] * 1e-29) * calc_relativistic_blackbody_continuum(
                w_grid, popt["T_prime"], popt["vphot"]
            )

            fig, ax = plt.subplots(figsize=(11, 6))
            # 텔루릭 흡수대 음영 표시
            ax.axvspan(13100, 14400, color="lightgray", alpha=0.4, label="Telluric Band")
            ax.axvspan(17550, 19200, color="lightgray", alpha=0.4)

            ax.plot(wave, flux, color="silver", label="Observed Data", lw=0.9, zorder=1)
            ax.plot(w_grid, cont_flux, color="navy", ls="--", lw=1.5, label="Relativistic Continuum", zorder=2)
            ax.plot(w_grid, model_flux, color="crimson", lw=2.0,
                    label=rf"+{days:.2f}d Best-Fit ($D_L={dl_val:.1f}\,\mathrm{{Mpc}}$, $T'={popt['T_prime']:.0f}\,\mathrm{{K}}$)",
                    zorder=3)

            ax.set_xlim(3500, 22000)
            ax.set_ylim(bottom=0.0)
            ax.set_xlabel(r"Rest Wavelength [$\AA$]", fontsize=12)
            ax.set_ylabel(r"Flux [$erg/s/cm^2/\AA$]", fontsize=12)
            ax.set_title(f"AT2017gfo Full Spectrum Fit (+{days:.2f}d) [{case_id}]", fontsize=13, fontweight="bold")
            ax.legend(loc="upper right", fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(target_save_dir, f"Plot1_Spectrum_Fit_{days:.2f}d.png"), dpi=200)
            plt.close(fig)

            # (2) 정규화 라인 프로파일 (7,000 ~ 12,500 Å 핵심 영역 확대)
            mask_zoom = (wave >= 7000) & (wave <= 12500)
            w_zoom, f_zoom = wave[mask_zoom], flux[mask_zoom]
            cont_zoom = (popt["N_29"] * 1e-29) * calc_relativistic_blackbody_continuum(
                w_zoom, popt["T_prime"], popt["vphot"]
            )

            norm_obs = f_zoom / np.maximum(cont_zoom, 1e-35)
            norm_best = planck_with_mod_full_relativistic(
                w_zoom, popt["T_prime"], popt["N_29"], popt["vmax"], popt["vphot"],
                tau=tau_val, trans=trans_val, ve=ve_val,
                amp1=amp1_val, amp2=amp2_val, t0=t_ph,
                use_ltt=True, use_nlte=use_nlte
            ) / np.maximum(cont_zoom, 1e-35)

            norm_std = planck_with_mod_full_relativistic(
                w_zoom, popt["T_prime"], popt["N_29"], popt["vmax"], popt["vphot"],
                tau=tau_val, trans=1.0, ve=ve_val,
                amp1=amp1_val, amp2=amp2_val, t0=t_ph,
                use_ltt=True, use_nlte=use_nlte
            ) / np.maximum(cont_zoom, 1e-35)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(w_zoom, norm_obs, color="lightgray", lw=1.2, label="Normalized Data", zorder=1)
            ax.plot(w_zoom, norm_best, color="teal", lw=2.2,
                    label=rf"Best-fit Profile ($\mathrm{{trans}}={trans_val:.2f}$, $\tau={tau_val:.2f}$)", zorder=3)
            ax.plot(w_zoom, norm_std, color="darkorange", ls="--", lw=1.8,
                    label=r"Standard Profile ($\mathrm{trans}=1.0$)", zorder=2)
            ax.axhline(1.0, color="black", ls=":", label="Normalized Continuum (1.0)")
            ax.axvline(10327.311, color="navy", ls="-.", alpha=0.7, label=r"Rest $\mathrm{Sr\ II}$ ($10327\,\AA$)")
            ax.axvline(10833.3, color="darkgreen", ls="-.", alpha=0.7, label=r"Rest $\mathrm{He\ I}$ ($10833\,\AA$)")

            ax.set_xlim(7000, 12500)
            ax.set_ylim(0.4, 2.2)
            ax.set_xlabel(r"Rest Wavelength [$\AA$]", fontsize=12)
            ax.set_ylabel(r"Normalized Flux ($F_{\lambda}/F_{\text{cont}}$)", fontsize=12)
            ax.set_title(f"Sr II Region Line Profile (+{days:.2f}d) [{case_id}]", fontsize=13, fontweight="bold")
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(target_save_dir, f"Plot2_Line_Profile_{days:.2f}d.png"), dpi=200)
            plt.close(fig)

        print(f"--> [성공] [{case_id}] 전체 플롯 저장 완료!")


if __name__ == "__main__":
    plot_all_results()