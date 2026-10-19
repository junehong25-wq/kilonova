#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
scripts/run_mcmc_fit.py
킬로노바 AT2017gfo MCMC 피팅 엔진 (96 Walkers x 20,000 Steps)
- 1일차 부 모드(vphot=0.171) 제거 및 2/4일차 vphot 경계 여유 확보
"""

import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import warnings
import multiprocessing as mp
import numpy as np
from scipy.optimize import minimize
import emcee

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_data
from src.probability import MCMCProbabilityWrapper
from src.models import (
    planck_with_mod_full_relativistic,
    lum_dist_arr
)

warnings.filterwarnings("ignore")

TARGET_CASE = "Case1_withLTT_PureLTE"
N_CORES = 48
N_WALKERS = 96
N_STEPS = 20000
BURN_IN = 10000
THIN_STEP = 15

PARAM_NAMES = ["T_prime", "N_29", "vmax", "vphot", "tau", "trans", "ve", "amp1", "amp2"]
CORNER_LABELS = [
    r"$T^\prime$", r"$N_{29}$", r"$v_{\max}$", r"$v_{\text{phot}}$",
    r"$\tau$", r"$\text{trans}$", r"$v_e$", r"$\text{amp}_1$", r"$\text{amp}_2$"
]

PHASES = [
    {
        "label": "Phase +1.43d (OB1)", "days": 1.427,
        "url": "https://sid.erda.dk/share_redirect/df1fMhon6Z/dereddened%2Bderedshifted_spectra/AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57983.969_Phase%2B1.43d_deredz.dat",
        "file": "OB1_1.43d.dat",
        "bounds": [
            (4400.0, 4750.0),
            (0.45, 0.60),
            (0.37, 0.44),
            (0.182, 0.225),  # 0.171 부 모드 차단 -> 주 모드(0.196) 단일 가우시안 완성
            (0.60, 2.00),
            (0.00, 0.10),
            (0.08, 0.18),
            (0.04, 0.18),
            (0.28, 0.46),
        ],
        "p_init": [4566.0, 0.511, 0.391, 0.196, 1.20, 0.001, 0.120, 0.099, 0.368]
    },
    {
        "label": "Phase +2.42d (OB2)", "days": 2.417,
        "url": "https://sid.erda.dk/share_redirect/df1fMhon6Z/dereddened%2Bderedshifted_spectra/AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57984.969_Phase%2B2.42d_deredz.dat",
        "file": "OB2_2.42d.dat",
        "bounds": [
            (3150.0, 3600.0),
            (0.95, 1.30),
            (0.31, 0.40),
            (0.105, 0.175),  # 0.130 하한선 접촉 해소 -> 0.105로 여유 개방
            (0.60, 2.20),
            (0.10, 1.20),
            (0.06, 0.16),
            (0.05, 0.20),
            (0.15, 0.32),
        ],
        "p_init": [3398.0, 1.101, 0.345, 0.130, 1.22, 0.31, 0.109, 0.119, 0.225]
    },
    {
        "label": "Phase +3.41d (OB3)", "days": 3.413,
        "url": "https://sid.erda.dk/share_redirect/df1fMhon6Z/dereddened%2Bderedshifted_spectra/AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57985.974_Phase%2B3.41d_deredz.dat",
        "file": "OB3_3.41d.dat",
        "bounds": [
            (2750.0, 3100.0),
            (1.30, 1.65),
            (0.28, 0.37),
            (0.12, 0.18),
            (1.00, 3.00),
            (1.50, 3.00),
            (0.025, 0.075),
            (0.16, 0.32),
            (0.36, 0.56),
        ],
        "p_init": [2921.0, 1.435, 0.321, 0.151, 1.70, 2.11, 0.046, 0.236, 0.458]
    },
    {
        "label": "Phase +4.40d (OB4)", "days": 4.403,
        "url": "https://sid.erda.dk/share_redirect/df1fMhon6Z/dereddened%2Bderedshifted_spectra/AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57986.974_Phase%2B4.40d_deredz.dat",
        "file": "OB4_4.40d.dat",
        "bounds": [
            (2500.0, 2750.0),
            (1.45, 1.80),
            (0.22, 0.32),
            (0.14, 0.21),    # 0.180 상한선 접촉 해소 -> 0.21로 여유 개방
            (0.20, 0.90),
            (6.50, 12.00),
            (0.05, 0.14),
            (0.22, 0.40),
            (0.38, 0.56),
        ],
        "p_init": [2616.0, 1.622, 0.256, 0.178, 0.47, 8.72, 0.091, 0.304, 0.473]
    }
]


def run_single_epoch(p_info, pool, target_save_dir):
    label = p_info["label"]
    days = p_info["days"]
    url = p_info["url"]
    local_file = os.path.join(target_save_dir, p_info["file"])
    bounds = p_info["bounds"]
    p_init = np.array(p_info["p_init"], dtype=np.float64)
    time_s = days * 86400.0

    bounds_arr = np.array(bounds, dtype=np.float64)
    low_b, high_b = bounds_arr[:, 0], bounds_arr[:, 1]
    spans = high_b - low_b

    print(f"\n--> [{label}] 데이터 로드 및 전역 최적화...")
    wave, flux, err = load_data(url, local_file)
    eff_err = np.maximum(err, 0.05 * np.abs(flux))
    x_fit, y_fit, err_fit = wave[::6], flux[::6], eff_err[::6]

    prob_wrapper = MCMCProbabilityWrapper(
        x_fit, y_fit, err_fit, time_s, bounds, n_days=days
    )

    res = minimize(
        prob_wrapper.chi2_for_minimizer, p_init, method="Nelder-Mead",
        options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-3}
    )
    center = res.x if (res.success and np.isfinite(prob_wrapper.log_prior(res.x))) else p_init
    center = np.clip(center, low_b + 0.02 * spans, high_b - 0.02 * spans)

    ndim = len(bounds)
    pos = []
    for _ in range(N_WALKERS):
        for _try in range(2000):
            cand = center + 0.0008 * spans * np.random.randn(ndim)
            cand = np.clip(cand, low_b + 1e-4 * spans, high_b - 1e-4 * spans)
            if np.isfinite(prob_wrapper.log_prior(cand)):
                pos.append(cand)
                break
        else:
            pos.append(center)
    pos = np.array(pos)

    print(f"    MCMC 샘플링 중 ({N_WALKERS} Walkers x {N_STEPS} Steps, Processes: {N_CORES})...")
    sampler = emcee.EnsembleSampler(
        N_WALKERS, ndim, prob_wrapper, pool=pool,
        moves=emcee.moves.StretchMove(a=1.8)
    )
    sampler.run_mcmc(pos, N_STEPS, progress=True)

    flat_samples = sampler.get_chain(discard=BURN_IN, thin=THIN_STEP, flat=True)
    np.save(os.path.join(target_save_dir, f"samples_{days:.3f}d.npy"), flat_samples)

    popt = {}
    for i, name in enumerate(PARAM_NAMES):
        popt[name] = float(np.median(flat_samples[:, i]))

    model_fit = planck_with_mod_full_relativistic(
        x_fit, popt["T_prime"], popt["N_29"], popt["vmax"], popt["vphot"],
        tau=popt["tau"], trans=popt["trans"], ve=popt["ve"],
        amp1=popt["amp1"], amp2=popt["amp2"], t0=time_s
    )
    chi2_val = float(np.sum(((y_fit - model_fit) / err_fit) ** 2))
    red_chi2 = chi2_val / max(1, len(x_fit) - ndim)

    dl_samples = lum_dist_arr(flat_samples[:, 1], flat_samples[:, 3], n_days=days)
    dl_med = float(np.median(dl_samples))

    print(
        f"    [완료: {label}] Red.Chi2={red_chi2:.2f} | T'={popt['T_prime']:.0f}K | "
        f"N_29={popt['N_29']:.2f} | D_L={dl_med:.2f} Mpc | tau={popt['tau']:.2f} | trans={popt['trans']:.2f}"
    )

    return {
        "summary": {
            "days": days, "label": label, "popt": popt,
            "red_chi2": red_chi2, "dl_med": dl_med
        },
        "spectrum": {
            "days": days, "label": label,
            "wave": wave.tolist(), "flux": flux.tolist(),
            "x_fit": x_fit.tolist(), "model_fit": model_fit.tolist()
        }
    }


def main():
    target_save_dir = os.path.join(PROJECT_ROOT, "output_results", TARGET_CASE)
    os.makedirs(target_save_dir, exist_ok=True)

    print("========================================================")
    print(f" [가동 시작: {TARGET_CASE}] (96 Walkers x 20,000 Steps)")
    print(f" 병렬 프로세스 할당: {N_CORES} 코어 (1:1 매칭)")
    print("========================================================")

    results_summary = []
    spectra_data = []

    with mp.Pool(processes=N_CORES) as pool:
        for p_info in PHASES:
            res = run_single_epoch(p_info, pool, target_save_dir)
            results_summary.append(res["summary"])
            spectra_data.append(res["spectrum"])

    with open(os.path.join(target_save_dir, "results_summary.json"), "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    with open(os.path.join(target_save_dir, "spectra_data.json"), "w", encoding="utf-8") as f:
        json.dump(spectra_data, f)

    with open(os.path.join(target_save_dir, "labels_dict.json"), "w", encoding="utf-8") as f:
        json.dump({"labels": PARAM_NAMES, "corner_labels": CORNER_LABELS}, f, indent=2)

    print("\n[완료] 전체 에포크 MCMC 샘플링 및 데이터 저장이 정상 종료되었습니다.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
