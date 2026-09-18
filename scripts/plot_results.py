#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from src.models import (
    planck_with_mod_full_relativistic,
    calc_relativistic_blackbody_continuum
)

try:
    import corner
except ImportError:
    corner = None


def plot_all_results():
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output_results")
    case_folders = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    if not case_folders:
        print("output_results 내에 피팅 결과 폴더가 없습니다.")
        return

    for case_id in case_folders:
        target_save_dir = os.path.join(base_dir, case_id)
        res_json = os.path.join(target_save_dir, 'results_summary.json')
        spec_json = os.path.join(target_save_dir, 'spectra_data.json')
        lbl_json = os.path.join(target_save_dir, 'labels_dict.json')

        if not os.path.exists(res_json):
            continue

        print(f"\n--> [{case_id}] 9D 물리 정규화 기반 플롯 일괄 생성 시작...")

        with open(res_json, 'r') as f:
            results_summary = json.load(f)
        with open(spec_json, 'r') as f:
            spectra_data = json.load(f)
        with open(lbl_json, 'r') as f:
            labels_dict = json.load(f)

        # 1. 코너 플롯 재생성
        for sdata in spectra_data:
            days = sdata["days"]
            samples_path = os.path.join(target_save_dir, f"samples_{days:.3f}d.npy")
            if os.path.exists(samples_path) and corner is not None:
                samples = np.load(samples_path)
                fig = corner.corner(
                    samples, labels=labels_dict["corner_labels"],
                    quantiles=[0.16, 0.50, 0.84], show_titles=True, title_fmt=".3f"
                )
                fig.savefig(os.path.join(target_save_dir, f"Corner_Phase_{days:.2f}d.png"), dpi=200)
                plt.close(fig)

        # 2. 개별 Spectrum Fit & Line Profile
        for idx, sdata in enumerate(spectra_data):
            days = sdata["days"]
            popt = results_summary[idx]["popt"]
            wave = np.array(sdata["wave"])
            flux = np.array(sdata["flux"])
            t_ph = days * 86400.0

            # (1) 개별 스펙트럼 핏
            model_flux = planck_with_mod_full_relativistic(
                wave, popt["T_prime"], popt["N_29"], popt["vmax"], popt["vphot"],
                tau=popt["tau"], trans=popt["trans"], ve=popt["ve"],
                amp1=popt["amp1"], amp2=popt["amp2"], t0=t_ph
            )
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(wave, flux, color="lightgray", label="Data", lw=1.0)
            ax.plot(wave, model_flux, color="crimson", lw=2.0,
                    label=rf"+{days:.2f}d Fit: $D_L={results_summary[idx]['dl_med']:.1f}\mathrm{{Mpc}}$, $\tau={popt['tau']:.2f}$")
            ax.set_xlabel(r"Rest Wavelength [$\AA$]", fontsize=12)
            ax.set_ylabel(r"Flux [$erg/s/cm^2/\AA$]", fontsize=12)
            ax.set_title(f"AT2017gfo Spectrum Fit (+{days:.2f}d) [{case_id}]", fontsize=13, fontweight="bold")
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(target_save_dir, f"Plot1_Spectrum_Fit_{days:.2f}d.png"), dpi=200)
            plt.close(fig)

            # (2) 개별 라인 프로파일 (물리 연속광 F_cont 정규화)
            mask_zoom = (wave >= 9500) & (wave <= 12500)
            w_zoom, f_zoom = wave[mask_zoom], flux[mask_zoom]
            cont_zoom = (popt["N_29"] * 1e-29) * calc_relativistic_blackbody_continuum(w_zoom, popt["T_prime"], popt["vphot"])

            norm_obs = f_zoom / cont_zoom
            norm_occulted = planck_with_mod_full_relativistic(
                w_zoom, popt["T_prime"], popt["N_29"], popt["vmax"], popt["vphot"],
                tau=popt["tau"], trans=popt["trans"], ve=popt["ve"], amp1=popt["amp1"], amp2=popt["amp2"], t0=t_ph
            ) / cont_zoom

            norm_no_occult = planck_with_mod_full_relativistic(
                w_zoom, popt["T_prime"], popt["N_29"], popt["vmax"], popt["vphot"],
                tau=popt["tau"], trans=1.0, ve=popt["ve"], amp1=popt["amp1"], amp2=popt["amp2"], t0=t_ph
            ) / cont_zoom

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(w_zoom, norm_obs, color="lightgray", lw=1.2, label="Normalized Observed Data")
            ax.plot(w_zoom, norm_occulted, color="teal", lw=2.2, label=rf"Best-fit Profile ($\mathrm{{trans}}={popt['trans']:.2f}$)")
            ax.plot(w_zoom, norm_no_occult, color="darkorange", ls="--", lw=2.0, label=r"Standard Line ($\mathrm{trans}=1.0$)")
            ax.axhline(1.0, color="black", ls=":", label="Normalized Continuum (1.0)")
            ax.axvline(10914.89, color="royalblue", ls=":", label=r"Rest $\mathrm{Sr\ II}$")
            ax.set_xlim(9500, 12500)
            ax.set_ylim(0.4, 2.0)
            ax.set_xlabel(r"Rest Wavelength [$\AA$]", fontsize=12)
            ax.set_ylabel(r"Normalized Flux ($F_{\lambda}/F_{\text{cont}}$)", fontsize=12)
            ax.set_title(f"Sr II Line Profile (+{days:.2f}d) [{case_id}]", fontsize=13, fontweight="bold")
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(target_save_dir, f"Plot2_Line_Profile_{days:.2f}d.png"), dpi=200)
            plt.close(fig)

        print(f"--> [성공] [{case_id}] 전체 플롯 렌더링 완료!")


if __name__ == "__main__":
    plot_all_results()