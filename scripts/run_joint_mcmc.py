#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
scripts/run_joint_mcmc.py
고속화(Fast-Sampling) 및 즉각 로깅이 적용된 4-에포크 결합 MCMC 파이프라인
"""

import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import multiprocessing as mp
import numpy as np
import emcee

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.settings import PHASES_CONFIG
from src.data_loader import load_clean_spectrum, TELLURIC_BANDS_AA
from src.models import synthesize_epoch_spectrum
from src.probability import JointMultiEpochProbability, sample_h0_from_dl_posterior

PARAM_LABELS = [
    "D_L", "cos_iota", "xi_asph",
    "T_1", "T_2", "T_3", "T_4",
    "vp_1", "vp_2", "vp_3", "vp_4",
    "dv_max", "ve_c",
    "tau_sr_1", "tau_sr_2", "tau_sr_3", "tau_sr_4",
    "tau_he_3", "tau_he_4", "uv_blanket"
]


def main():
    out_dir = os.path.join(PROJECT_ROOT, "output_results", "Joint_4Epoch_Relativistic_NLTE")
    os.makedirs(out_dir, exist_ok=True)

    nir_defaults = [(0.10, 0.35), (0.12, 0.22), (0.23, 0.45), (0.30, 0.47)]
    epochs_data = []

    print("=== [1/4] VLT/X-Shooter 4개 에포크 분광 데이터 로드 (고속 최적화 샘플링) ===", flush=True)
    for idx, p_cfg in enumerate(PHASES_CONFIG):
        local_path = os.path.join(out_dir, p_cfg["filename"])
        w_fit, f_fit, e_fit, w_raw, f_raw = load_clean_spectrum(
            p_cfg["url"], local_path, subsample_step=12
        )
        epochs_data.append({
            "label": p_cfg["label"],
            "days": p_cfg["days"],
            "wave": w_fit,
            "flux": f_fit,
            "err": e_fit,
            "wave_raw": w_raw,
            "flux_raw": f_raw,
            "nir_a1": nir_defaults[idx][0],
            "nir_a2": nir_defaults[idx][1],
        })
        print(f"  - {p_cfg['label']}: 피팅 픽셀 수 = {len(w_fit)} pts (최적화 완료)", flush=True)

    joint_prob = JointMultiEpochProbability(epochs_data, use_gw_prior=True, use_ltt=True)
    ndim = len(PARAM_LABELS)

    # 7800+ 스텝까지 도달하며 사전 검증된 고정밀 중심 파라미터 초기화
    theta_init = np.array([
        42.3,                         # D_L [Mpc]
        0.86,                         # cos_iota
        1.00,                         # xi_asph
        4620.0, 3450.0, 2920.0, 2610.0,  # T_1..T_4 [K]
        0.255, 0.196, 0.155, 0.125,   # vp_1..vp_4 [c] (단조 감소 제약)
        0.135,                        # dv_max [c]
        0.085,                        # ve_c [c]
        1.30, 1.20, 1.15, 0.49,       # tau_sr_1..4
        0.45, 1.35,                   # tau_he_3, tau_he_4
        1.10                          # uv_blanket
    ], dtype=np.float64)

    print("\n=== [2/4] 초기 결합 파라미터 구성 완료 ===", flush=True)
    center_pt = theta_init
    print(f"  - 기준 광도거리 D_L = {center_pt[0]:.2f} Mpc | vp_1..4 = {center_pt[7:11].round(3)}", flush=True)

    # 메모리 안정성을 고려한 32개 병렬 코어 배분
    n_cores = max(1, min(32, (os.cpu_count() or 4) - 2))
    n_walkers = 64
    n_steps = 8000
    burn_in = 2500

    spans = joint_prob.bounds[:, 1] - joint_prob.bounds[:, 0]
    pos = []
    rng = np.random.default_rng(2026)
    for _ in range(n_walkers):
        for _try in range(5000):
            cand = center_pt + 0.003 * spans * rng.standard_normal(ndim)
            if np.isfinite(joint_prob.log_prior(cand)):
                pos.append(cand)
                break
        else:
            pos.append(center_pt)
    pos = np.array(pos)

    moves = [
        (emcee.moves.DEMove(), 0.8),
        (emcee.moves.DESnookerMove(), 0.2),
    ]

    print(f"\n=== [3/4] 고속 Joint MCMC 샘플링 가동 ({n_walkers} Walkers x {n_steps} Steps, {n_cores} Cores) ===", flush=True)
    with mp.Pool(processes=n_cores) as pool:
        sampler = emcee.EnsembleSampler(n_walkers, ndim, joint_prob, pool=pool, moves=moves)
        sampler.run_mcmc(pos, n_steps, progress=True)

    flat_samples = sampler.get_chain(discard=burn_in, thin=10, flat=True)
    np.save(os.path.join(out_dir, "joint_flat_samples.npy"), flat_samples)

    dl_samples = flat_samples[:, 0]
    h0_samples = sample_h0_from_dl_posterior(dl_samples)

    dl_med, dl_lo, dl_hi = np.percentile(dl_samples, [50, 16, 84])
    h0_med, h0_lo, h0_hi = np.percentile(h0_samples, [50, 16, 84])

    popt_med = np.median(flat_samples, axis=0)
    summary_dict = {
        "D_L_Mpc": {
            "median": float(dl_med),
            "err_minus": float(dl_med - dl_lo),
            "err_plus": float(dl_hi - dl_med),
        },
        "H0_km_s_Mpc": {
            "median": float(h0_med),
            "err_minus": float(h0_med - h0_lo),
            "err_plus": float(h0_hi - h0_med),
        },
        "parameters": {
            name: {
                "median": float(np.percentile(flat_samples[:, i], 50)),
                "p16": float(np.percentile(flat_samples[:, i], 16)),
                "p84": float(np.percentile(flat_samples[:, i], 84)),
            }
            for i, name in enumerate(PARAM_LABELS)
        }
    }

    with open(os.path.join(out_dir, "joint_fit_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)

    print("\n========================================================", flush=True)
    print(f" [최종 결합 추론 결과]", flush=True)
    print(f"  * 전역 광도 거리 D_L = {dl_med:.2f} (+{dl_hi - dl_med:.2f} / -{dl_med - dl_lo:.2f}) Mpc", flush=True)
    print(f"  * 허블 상수 H_0      = {h0_med:.2f} (+{h0_hi - h0_med:.2f} / -{h0_med - h0_lo:.2f}) km/s/Mpc", flush=True)
    print("========================================================\n", flush=True)

    plot_joint_results(epochs_data, popt_med, h0_samples, summary_dict, out_dir)


def plot_joint_results(epochs_data, popt, h0_samples, summary_dict, out_dir):
    print("=== [4/4] 결과 도판(Spectra Fit 및 H_0 Posterior) 생성 중 ===", flush=True)
    dl_mpc, cos_iota, xi_asph = popt[0], popt[1], popt[2]
    temps = popt[3:7]
    vphots = popt[7:11]
    dv_max, ve_c = popt[11], popt[12]
    taus_sr = popt[13:17]
    taus_he = [0.0, 0.0, popt[17], popt[18]]
    uv_blanket = popt[19]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=300)
    axes = axes.flatten()

    for k, ep in enumerate(epochs_data):
        ax = axes[k]
        days = ep["days"]
        w_plot = np.linspace(3600.0, 22000.0, 1000)
        vp_k = vphots[k]
        vmax_k = min(0.55, vp_k + dv_max * ((days / 1.427) ** (-0.15)))

        f_mod, f_cont = synthesize_epoch_spectrum(
            w_plot, dl_mpc, temps[k], vp_k, vmax_k, ve_c,
            taus_sr[k], taus_he[k], days,
            cos_iota=cos_iota, axis_ratio_xi=xi_asph,
            tau_uv_blanket=uv_blanket if k == 0 else 0.3 * uv_blanket,
            nir_amp1=ep["nir_a1"], nir_amp2=ep["nir_a2"], use_ltt=True
        )

        for (t_low, t_high) in TELLURIC_BANDS_AA:
            ax.axvspan(t_low, t_high, color="lightgray", alpha=0.4, label="Telluric" if k == 0 else None)

        ax.plot(ep["wave_raw"], ep["flux_raw"], color="silver", lw=0.6, label="Observed Data")
        ax.plot(w_plot, f_cont, color="navy", ls="--", lw=1.5, label="Relativistic Continuum")
        ax.plot(
            w_plot, f_mod, color="#D81B60", lw=2.0,
            label=rf"Joint Fit ($D_L={dl_mpc:.1f}$ Mpc, $T'={temps[k]:.0f}$ K, $v_{{\mathrm{{phot}}}}={vp_k:.3f}c$)"
        )
        ax.set_title(f"{ep['label']} [Joint Relativistic NLTE]", fontsize=12, fontweight="bold")
        ax.set_xlim(3600, 22000)
        ax.set_xlabel(r"Rest Wavelength [$\mathrm{\AA}$]")
        ax.set_ylabel(r"Flux [$\mathrm{erg\ s^{-1}\ cm^{-2}\ \AA^{-1}}$]")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8.5, loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Joint_4Epoch_Spectra_Fit.png"), dpi=300)
    plt.close(fig)

    fig_h0, ax_h0 = plt.subplots(figsize=(9, 5.5), dpi=300)
    ax_h0.hist(h0_samples, bins=50, density=True, color="#D81B60", alpha=0.35, edgecolor="#D81B60", lw=1.2,
               label="This Work (4-Epoch Joint Relativistic NLTE + GW Prior)")
    h0_med = summary_dict["H0_km_s_Mpc"]["median"]
    h0_p = summary_dict["H0_km_s_Mpc"]["err_plus"]
    h0_m = summary_dict["H0_km_s_Mpc"]["err_minus"]
    ax_h0.axvline(h0_med, color="#D81B60", lw=2.2, label=rf"Median $H_0 = {h0_med:.1f}^{{+{h0_p:.1f}}}_{{-{h0_m:.1f}}}\ \mathrm{{km\ s^{{-1}}\ Mpc^{{-1}}}}$")
    ax_h0.axvspan(67.4 - 0.5, 67.4 + 0.5, color="#5E35B1", alpha=0.25, label=r"Planck 2018 ($67.4 \pm 0.5$)")
    ax_h0.axvspan(73.04 - 1.04, 73.04 + 1.04, color="#FFC107", alpha=0.30, label=r"SH0ES 2022 ($73.04 \pm 1.04$)")
    ax_h0.set_xlim(60, 82)
    ax_h0.set_xlabel(r"Hubble Constant $H_0\ [\mathrm{km\ s^{-1}\ Mpc^{-1}}]$", fontsize=12, fontweight="bold")
    ax_h0.set_ylabel("Posterior Probability Density", fontsize=12, fontweight="bold")
    ax_h0.set_title(r"$H_0$ Measurement via Joint Relativistic Kilonova EPM", fontsize=13, fontweight="bold")
    ax_h0.grid(True, ls=":", alpha=0.4)
    ax_h0.legend(fontsize=9.5, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Joint_H0_Posterior.png"), dpi=300)
    plt.close(fig_h0)
    print(f"--> 모든 최종 도판 및 요약 파일 저장 완료: '{out_dir}'", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
