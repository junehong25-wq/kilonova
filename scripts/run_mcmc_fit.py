#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
scripts/run_mcmc_fit.py
킬로노바 AT2017gfo MCMC 피팅 파이프라인
- Case 1 (Case1_withLTT_PureLTE) 전용
- 2일차 vphot 경계 개방 (0.16 ~ 0.28)
- 3일차 초기화 집중 (분산 2%, burn-in 4000)으로 바이모달리티 제거
- OpenBLAS 스레드 격리 및 48 프로세스 1:1 매칭
"""

import os
import sys

# BLAS 단일 스레드 격리로 메모리 오버헤드 방지
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


def get_case_options(case_name):
    return {
        "use_ltt": True,
        "use_nlte": "NLTE" in case_name and "PureLTE" not in case_name
    }


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

    case_opts = get_case_options(target_case)

    nwalkers = 48
    ncpu = 48
    mcmc_steps = 10000
    burn_in = 4000      # 3일차 이중 모드 수렴을 위해 4,000으로 상향
    thin_step = 2

    phases = [
        {
            "label": "Phase +1.43d (OB1)",
            "days": 1.427,
            "url": "https://sid.erda.dk/share_redirect/df1fMhon6Z/dereddened%2Bderedshifted_spectra/AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57983.969_Phase%2B1.43d_deredz.dat",
            "local_file": os.path.join(base_dir, "OB1_1.43d.dat"),
            "bounds": [
                (3000.0, 7500.0), (0.20, 5.0), (0.22, 0.45), (0.19, 0.28),
                (0.01, 30.0), (0.00, 5.00), (0.01, 0.65), (0.00, 1.00), (0.00, 1.00)
            ],
            "init_guess": [4540.0, 0.52, 0.39, 0.20, 1.27, 0.00, 0.11, 0.09, 0.36]
        },
        {
            "label": "Phase +2.42d (OB2)",
            "days": 2.417,
            "url": "https://sid.erda.dk/share_redirect/df1fMhon6Z/dereddened%2Bderedshifted_spectra/AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57984.969_Phase%2B2.42d_deredz.dat",
            "local_file": os.path.join(base_dir, "OB2_2.42d.dat"),
            # [수정]: vphot 상한 0.24 -> 0.28 개방, vmax 0.45 개방
            "bounds": [
                (2500.0, 5500.0), (0.50, 6.0), (0.22, 0.45), (0.16, 0.28),
                (0.01, 30.0), (0.10, 6.00), (0.01, 0.45), (0.00, 1.00), (0.00, 1.00)
            ],
            "init_guess": [3130.0, 1.17, 0.38, 0.24, 4.50, 1.20, 0.025, 0.13, 0.23]
        },
        {
            "label": "Phase +3.41d (OB3)",
            "days": 3.413,
            "url": "https://sid.erda.dk/share_redirect/df1fMhon6Z/dereddened%2Bderedshifted_spectra/AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57985.974_Phase%2B3.41d_deredz.dat",
            "local_file": os.path.join(base_dir, "OB3_3.41d.dat"),
            "bounds": [
                (2200.0, 4500.0), (0.60, 8.0), (0.18, 0.38), (0.13, 0.22),
                (0.01, 30.0), (0.50, 8.00), (0.01, 0.45), (0.00, 1.00), (0.00, 1.00)
            ],
            # [수정]: Mode 1 중심값으로 타겟 고정
            "init_guess": [2853.0, 1.46, 0.32, 0.18, 3.32, 2.11, 0.023, 0.24, 0.46]
        },
        {
            "label": "Phase +4.40d (OB4)",
            "days": 4.403,
            "url": "https://sid.erda.dk/share_redirect/df1fMhon6Z/dereddened%2Bderedshifted_spectra/AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57986.974_Phase%2B4.40d_deredz.dat",
            "local_file": os.path.join(base_dir, "OB4_4.40d.dat"),
            "bounds": [
                (2000.0, 4000.0), (0.80, 10.0), (0.14, 0.30), (0.10, 0.17),
                (0.01, 30.0), (0.50, 10.00), (0.01, 0.45), (0.00, 1.00), (0.00, 1.00)
            ],
            "init_guess": [2708.0, 1.59, 0.25, 0.13, 0.61, 7.76, 0.087, 0.31, 0.48]
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
    print(f" [가동 시작: {target_case}] (LTT=True, NLTE={case_opts['use_nlte']})")
    print(f" 병렬 프로세스 할당: {ncpu} 코어 (1:1 매칭)")
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

        prob_wrapper = MCMCProbabilityWrapper(
            x_fit, y_fit, err_fit, time_s, bnds, n_days=days,
            use_ltt=True, use_nlte=case_opts["use_nlte"]
        )

        opt_res = minimize(
            prob_wrapper.chi2_for_minimizer, init_guess, method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-4, 'fatol': 1e-2}
        )
        center_point = opt_res.x if (opt_res.success and prob_wrapper.log_prior(opt_res.x) > -1e10) else init_guess

        valid_center = center_point.copy()
        if prob_wrapper.log_prior(valid_center) <= -1e10:
            for _ in range(20000):
                cand_u = np.random.uniform(low_b, high_b)
                if prob_wrapper.log_prior(cand_u) > -1e10:
                    valid_center = cand_u
                    break

        ndim = len(bnds)
        spans = high_b - low_b
        pos = []

        # [수정]: 3일차 바이모달리티 방지를 위해 초기 분산 폭을 ±2%로 제한
        for _ in range(nwalkers):
            p_cand = None
            for _ in range(1000):
                u_offset = np.random.uniform(-0.02, 0.02, size=ndim)
                cand = valid_center + spans * u_offset
                cand = np.clip(cand, low_b + 0.002 * spans, high_b - 0.002 * spans)
                if prob_wrapper.log_prior(cand) > -1e10:
                    p_cand = cand
                    break
            if p_cand is None:
                for _ in range(10000):
                    cand = np.random.uniform(low_b, high_b)
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
            tau=popt["tau"], trans=popt["trans"], ve=popt["ve"], amp1=popt["amp1"], amp2=popt["amp2"], t0=time_s,
            use_ltt=True, use_nlte=case_opts["use_nlte"]
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