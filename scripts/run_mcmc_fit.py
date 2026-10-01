#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import warnings
import multiprocessing as mp
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import emcee

from src.probability import MCMCProbabilityWrapper
from src.models import (
    planck_with_mod_full_relativistic,
    lum_dist_arr
)
from src.data_loader import load_data

warnings.filterwarnings("ignore")


def make_serializable(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list): return [make_serializable(v) for v in obj]
    if isinstance(obj, (np.int32, np.int64, np.integer)): return int(obj)
    if isinstance(obj, (np.float32, np.float64, np.floating)): return float(obj)
    return obj


def run_mcmc():
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output_results", "Case1_noLTT_PureLTE")
    os.makedirs(base_dir, exist_ok=True)

    # 48 워커에 맞춰 단일 소켓 물리 코어 32개로 고정 (서버 프로세스 과점 방지)
    nwalkers = 48
    ncpu = min(nwalkers, 32)
    mcmc_steps = 10000
    burn_in = 3000
    thin_step = 2

    # -------------------------------------------------------------
    # 바운드 완전 개방 (Flat Prior / Unconstrained Bounds)
    # 물리적 하한/광속 한계만 유지하고 탐색 공간을 완전히 개방
    # -------------------------------------------------------------
    unconstrained_bounds = [
        (1000.0, 25000.0), # T_prime (K)
        (0.01, 50.0),      # N_29
        (0.15, 0.70),      # vmax
        (0.05, 0.55),      # vphot
        (0.01, 50.0),      # tau
        (0.00, 5.00),      # trans
        (0.01, 0.80),      # ve
        (0.00, 2.00),      # amp1
        (0.00, 2.00)       # amp2
    ]

    phases = [
        {
            "label": "Phase +1.43d (OB1)",
            "days": 1.427,
            "url": "https://sid.erda.dk/share_redirect/df1fMhon6Z/dereddened%2Bderedshifted_spectra/AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57983.969_Phase%2B1.43d_deredz.dat",
            "local_file": os.path.join(base_dir, "OB1_1.43d.dat"),
            "bounds": unconstrained_bounds,
            "init_guess": [4800.0, 1.20, 0.38, 0.28, 4.0, 0.05, 0.18, 0.10, 0.35]
        },
        {
            "label": "Phase +2.42d (OB2)",
            "days": 2.417,
            "url": "https://sid.erda.dk/share_redirect/df1fMhon6Z/dereddened%2Bderedshifted_spectra/AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57984.969_Phase%2B2.42d_deredz.dat",
            "local_file": os.path.join(base_dir, "OB2_2.42d.dat"),
            "bounds": unconstrained_bounds,
            "init_guess": [3200.0, 2.50, 0.32, 0.24, 3.0, 0.10, 0.20, 0.08, 0.22]
        },
        {
            "label": "Phase +3.41d (OB3)",
            "days": 3.413,
            "url": "https://sid.erda.dk/share_redirect/df1fMhon6Z/dereddened%2Bderedshifted_spectra/AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57985.974_Phase%2B3.41d_deredz.dat",
            "local_file": os.path.join(base_dir, "OB3_3.41d.dat"),
            "bounds": unconstrained_bounds,
            "init_guess": [2800.0, 3.20, 0.26, 0.19, 1.5, 0.50, 0.20, 0.12, 0.30]
        },
        {
            "label": "Phase +4.40d (OB4)",
            "days": 4.403,
            "url": "https://sid.erda.dk/share_redirect/df1fMhon6Z/dereddened%2Bderedshifted_spectra/AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57986.974_Phase%2B4.40d_deredz.dat",
            "local_file": os.path.join(base_dir, "OB4_4.40d.dat"),
            "bounds": unconstrained_bounds,
            "init_guess": [2600.0, 4.00, 0.22, 0.15, 0.8, 0.80, 0.18, 0.15, 0.25]
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
    print(" [순수 MCMC 샘플링 파이프라인 가동 (바운드 완전 개방 모드)]")
    print("========================================================\n")

    for p_info in phases:
        label, days, url, local_file, bounds = p_info["label"], p_info["days"], p_info["url"], p_info["local_file"], p_info["bounds"]
        init_guess = p_info["init_guess"]
        time_s = days * 24.0 * 3600.0

        bounds_arr = np.array(bounds)
        low_b, high_b = bounds_arr[:, 0], bounds_arr[:, 1]

        print(f"--> [{label}] 데이터 로드 및 초기 최적화...")
        wave, flux, err = load_data(url, local_file)
        eff_err = np.maximum(err, 0.05 * np.abs(flux))
        x_fit, y_fit, err_fit = wave[::6], flux[::6], eff_err[::6]

        prob_wrapper = MCMCProbabilityWrapper(x_fit, y_fit, err_fit, time_s, bounds)

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

        ndim = len(bounds)
        spans = high_b - low_b
        pos = []

        for _ in range(nwalkers):
            p_cand = None
            for _ in range(500):
                cand = valid_center + spans * 0.010 * np.random.randn(ndim)
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

    print("========================================================")
    print(f" [MCMC 완료] 체인 및 결과 데이터가 저장되었습니다: {base_dir}")
    print(" 이제 'python scripts/plot_results.py'를 실행하여 플롯을 생성하십시오.")
    print("========================================================\n")


if __name__ == "__main__":
    mp.freeze_support()
    run_mcmc()
