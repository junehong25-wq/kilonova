#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
scripts/run_mcmc_fit.py
킬로노바 AT2017gfo MCMC 피팅 파이프라인
- Case 1 (Case1_withLTT_PureLTE) 전용
- 48 프로세스 1:1 매칭 멀티프로세싱
- 코너 플롯 경계벽 충돌 방지를 위한 Bounds 개방
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

from src.probability import MCMCProbabilityWrapper
from src.models import (
    planck_with_mod_full_relativistic,
    lum_dist_arr
)
from src.data_loader import load_data

warnings.filterwarnings("ignore")

CASE_LIST = [
    "Case1_withLTT_PureLTE"
]

def make_serializable(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list): return [make_serializable(v) for v in obj]
    if isinstance(obj, (np.int32, np.int64, np.integer)): return int(obj)
    if isinstance(obj, (np.float32, np.float64, np.floating)): return float(obj)
    return obj

def run_single_case(target_case):
    base_dir = os.path.join(PROJECT_ROOT, "output_results", target_case)
    os.makedirs(base_dir, exist_ok=True)

    nwalkers = 48
    ncpu = 48
    mcmc_steps = 10000
    burn_in = 5000      # 50% Burn-in으로 초기 산포 노이즈 완전 제거
    thin_step = 2

    # 코너 플롯 경계벽 충돌을 없애기 위해 충분히 개방된 물리적 Bounds
    phases = [
        {
            "label": "Phase +1.43d (OB1)",
            "days": 1.427,
            "url": "https://sid.erda.dk/share_redirect/df1fMhon6Z/dereddened%2Bderedshifted_spectra/AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57983.969_Phase%2B1.43d_deredz.dat",
            "local_file": os.path.join(base_dir, "OB1_1.43d.dat"),
            "bounds": [
                (4200.0, 4900.0), (0.40, 0.70), (0.35, 0.50), (0.16, 0.26),
                (0.50, 5.00), (0.00, 1.00), (0.05, 0.22), (0.02, 0.25), (0.20, 0.55)
            ],
            "init_guess": [4545.0, 0.514, 0.410, 0.205, 1.30, 0.05, 0.103, 0.097, 0.365]
        },
        {
            "label": "Phase +2.42d (OB2)",
            "days": 2.417,
            "url": "https://sid.erda.dk/share_redirect/df1fMhon6Z/dereddened%2Bderedshifted_spectra/AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57984.969_Phase%2B2.42d_deredz.dat",
            "local_file": os.path.join(base_dir, "OB2_2.42d.dat"),
            # vphot 상한을 0.33까지 개방, tau 폭주 방지를 위해 상한 12.0 설정
            "bounds": [
                (2850.0, 3300.0), (0.90, 1.45), (0.35, 0.50), (0.18, 0.33),
                (0.50, 12.00), (0.30, 3.00), (0.005, 0.08), (0.02, 0.25), (0.10, 0.35)
            ],
            "init_guess": [3033.0, 1.23, 0.435, 0.275, 4.20, 1.55, 0.015, 0.102, 0.202]
        },
        {
            "label": "Phase +3.41d (OB3)",
            "days": 3.413,
            "url": "https://sid.erda.dk/share_redirect/df1fMhon6Z/dereddened%2Bderedshifted_spectra/AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57985.974_Phase%2B3.41d_deredz.dat",
            "local_file": os.path.join(base_dir, "OB3_3.41d.dat"),
            # T' 하한 2650K, vmax 상한 0.42c, vphot 상한 0.25c까지 개방
            "bounds": [
                (2650.0, 3050.0), (1.20, 1.75), (0.28, 0.42), (0.13, 0.25),
                (1.00, 8.00), (0.80, 3.50), (0.008, 0.06), (0.10, 0.35), (0.30, 0.55)
            ],
            "init_guess": [2810.0, 1.50, 0.360, 0.195, 3.80, 2.05, 0.020, 0.226, 0.442]
        },
        {
            "label": "Phase +4.40d (OB4)",
            "days": 4.403,
            "url": "https://sid.erda.dk/share_redirect/df1fMhon6Z/dereddened%2Bderedshifted_spectra/AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57986.974_Phase%2B4.40d_deredz.dat",
            "local_file": os.path.join(base_dir, "OB4_4.40d.dat"),
            # vmax 상한 0.35c, vphot (0.08 ~ 0.20)으로 개방
            "bounds": [
                (2500.0, 2900.0), (1.30, 1.90), (0.20, 0.36), (0.08, 0.20),
                (0.20, 4.00), (3.00, 12.00), (0.02, 0.15), (0.15, 0.45), (0.35, 0.60)
            ],
            "init_guess": [2704.0, 1.60, 0.265, 0.135, 0.70, 7.50, 0.080, 0.300, 0.469]
        }
    ]

    labels = ["T_prime", "N_29", "vmax", "vphot", "tau", "trans", "ve", "amp1", "amp2"]
    corner_labels = [
        r"$T^\prime$", r"$N_{29}$", r"$v_{\max}$", r"$v_{\text{phot}}$",
        r"$\tau$", r"$\text{trans}$", r"$v_e$", r"$\text{amp}_1$", r"$\text{amp}_2$"
    ]
    labels_dict = {"labels": labels, "corner_labels": corner_labels}

    results_summary = []
    spectra_data = []

    print("\n========================================================")
    print(f" [가동 시작: {target_case}] (48-Core MCMC)")
    print("========================================================\n")

    for p_info in phases:
        label, days, url, local_file, bnds = p_info["label"], p_info["days"], p_info["url"], p_info["local_file"], p_info["bounds"]
        init_guess = p_info["init_guess"]
        time_s = days * 86400.0

        bounds_arr = np.array(bnds)
        low_b, high_b = bounds_arr[:, 0], bounds_arr[:, 1]

        print(f"--> [{label}] 데이터 로드 및 초기 최적화...")
        wave, flux, err = load_data(url, local_file)
        eff_err = np.maximum(err, 0.05 * np.abs(flux))

        x_fit, y_fit, err_fit = wave[::4], flux[::4], eff_err[::4]

        # 단일 인자 5개로 전달 (TypeError 방지)
        prob_wrapper = MCMCProbabilityWrapper(x_fit, y_fit, err_fit, time_s, bnds)

        opt_res = minimize(
            prob_wrapper.chi2_for_minimizer, init_guess, method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-4, 'fatol': 1e-2}
        )
        center_point = opt_res.x if (opt_res.success and prob_wrapper.log_prior(opt_res.x) > -1e10) else init_guess

        valid_center = center_point.copy()
        ndim = len(bnds)
        spans = high_b - low_b
        pos = []

        # 워커 분열 방지: 최적점 주변 1% 가우시안 볼 집속 초기화
        for _ in range(nwalkers):
            p_cand = None
            for _ in range(1000):
                u_offset = np.random.normal(0.0, 0.01, size=ndim)
                cand = valid_center + spans * u_offset
                cand = np.clip(cand, low_b + 0.002 * spans, high_b - 0.002 * spans)
                if prob_wrapper.log_prior(cand) > -1e10:
                    p_cand = cand
                    break
            pos.append(p_cand if p_cand is not None else valid_center)
        pos = np.array(pos)

        print(f"    MCMC 샘플링 중 ({nwalkers} Walkers x {mcmc_steps} Steps, Processes: {ncpu})...")
        with mp.Pool(processes=ncpu) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, prob_wrapper, pool=pool)
            sampler.run_mcmc(pos, mcmc_steps, progress=True)

        flat_samples = sampler.get_chain(discard=burn_in, thin=thin_step, flat=True)
        np.save(os.path.join(base_dir, f"samples_{days:.3f}d.npy"), flat_samples)

        popt = {}
        for i in range(ndim):
            mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
            popt[labels[i]] = float(mcmc[1])

        model_fit = planck_with_mod_full_relativistic(
            x_fit, popt["T_prime"], popt["N_29"], popt["vmax"], popt["vphot"],
            tau=popt["tau"], trans=popt["trans"], ve=popt["ve"], amp1=popt["amp1"], amp2=popt["amp2"], t0=time_s
        )
        chi2_fit = np.sum(((y_fit - model_fit) / err_fit) ** 2)
        red_chi2_fit = chi2_fit / (len(x_fit) - ndim)

        dl_samples = lum_dist_arr(flat_samples[:, 1], flat_samples[:, 3], flat_samples[:, 5], n_days=days)
        dl_med = float(np.median(dl_samples))

        print(f"    [완료: {label}] Red.Chi2={red_chi2_fit:.2f} | T'={popt['T_prime']:.0f}K | N_29={popt['N_29']:.2f} | D_L={dl_med:.2f} Mpc | trans={popt['trans']:.2f}\n")

        results_summary.append({
            "days": days, "label": label, "popt": popt, "red_chi2": red_chi2_fit, "dl_med": dl_med
        })
        spectra_data.append({
            "days": days, "label": label, "wave": wave, "flux": flux, "x_fit": x_fit, "model_fit": model_fit
        })

    with open(os.path.join(base_dir, 'spectra_data.json'), 'w') as f:
        json.dump(make_serializable(spectra_data), f)
    with open(os.path.join(base_dir, 'labels_dict.json'), 'w') as f:
        json.dump(make_serializable(labels_dict), f)
    with open(os.path.join(base_dir, 'results_summary.json'), 'w') as f:
        json.dump(make_serializable(results_summary), f)
    with open(os.path.join(base_dir, 'fit_summary_all.json'), 'w') as f:
        json.dump(make_serializable(results_summary), f)

def run_all_cases():
    for case_name in CASE_LIST:
        run_single_case(case_name)

if __name__ == "__main__":
    mp.freeze_support()
    run_all_cases()