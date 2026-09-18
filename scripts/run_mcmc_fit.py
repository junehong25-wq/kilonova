#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import warnings
import urllib.request
import multiprocessing as mp
import numpy as np
import pandas as pd
from scipy import constants
from scipy.optimize import minimize
import emcee

from src.probability import MCMCProbabilityWrapper
from src.models import (
    planck_with_mod_full_relativistic,
    lum_dist_arr
)

warnings.filterwarnings("ignore")


def load_data(url, local_filename="temp_spectrum.dat"):
    os.makedirs(os.path.dirname(os.path.abspath(local_filename)), exist_ok=True)
    if not os.path.exists(local_filename):
        print(f"    [다운로드 중] 관측 데이터 생성: {os.path.basename(local_filename)}")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as resp, open(local_filename, 'wb') as out_f:
                out_f.write(resp.read())
        except Exception:
            alt_path = os.path.join(os.getcwd(), "output", os.path.basename(local_filename))
            if os.path.exists(alt_path):
                import shutil
                shutil.copy(alt_path, local_filename)

    df = pd.read_csv(local_filename, sep=r'\s+', comment='#', header=None, on_bad_lines='skip')
    wave_raw = pd.to_numeric(df[0], errors='coerce').values
    flux_raw = pd.to_numeric(df[1], errors='coerce').values
    err_raw = pd.to_numeric(df[3], errors='coerce').values

    valid = ~np.isnan(wave_raw) & ~np.isnan(flux_raw) & ~np.isnan(err_raw)
    wave, flux, err = wave_raw[valid], flux_raw[valid], err_raw[valid]
    if wave.max() < 3000:
        wave = wave * 10.0

    exc_reg = (
            ~((wave > 13100) & (wave < 14400))
            & ~((wave > 17550) & (wave < 19200))
            & ~((wave > 5330) & (wave < 5740))
            & ~((wave > 9840) & (wave < 10300))
            & (wave >= 3800) & (wave <= 21500)
    )
    return wave[exc_reg], flux[exc_reg], err[exc_reg]


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

    ncpu = max(1, (os.cpu_count() or 4) - 2)
    mcmc_steps = 6000

    phases = [
        {
            "label": "Phase +1.43d (OB1)", "days": 1.427,
            "url": "https://sid.erda.dk/share_redirect/df1fMhon6Z/dereddened%2Bderedshifted_spectra/AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57983.969_Phase%2B1.43d_deredz.dat",
            "local_file": os.path.join(base_dir, "OB1_1.43d.dat"),
            "bounds": [(4500.0, 6000.0), (1.00, 1.50), (0.32, 0.40), (0.260, 0.295), (1.0, 10.0), (0.0, 1.5),
                       (0.01, 0.50), (0.0, 0.60), (0.20, 0.70)]
        },
        {
            "label": "Phase +2.42d (OB2)", "days": 2.417,
            "url": "https://sid.erda.dk/share_redirect/df1fMhon6Z/dereddened%2Bderedshifted_spectra/AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57984.969_Phase%2B2.42d_deredz.dat",
            "local_file": os.path.join(base_dir, "OB2_2.42d.dat"),
            "bounds": [(3000.0, 3400.0), (1.80, 2.90), (0.28, 0.36), (0.230, 0.270), (1.0, 10.0), (0.0, 2.0),
                       (0.01, 0.50), (0.0, 0.80), (0.0, 0.80)]
        },
        {
            "label": "Phase +3.41d (OB3)", "days": 3.413,
            "url": "https://sid.erda.dk/share_redirect/df1fMhon6Z/dereddened%2Bderedshifted_spectra/AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57985.974_Phase%2B3.41d_deredz.dat",
            "local_file": os.path.join(base_dir, "OB3_3.41d.dat"),
            "bounds": [(2629.0, 3029.0), (2.10, 3.10), (0.24, 0.32), (0.180, 0.220), (1.0, 10.0), (0.0, 2.5),
                       (0.01, 0.50), (0.0, 1.00), (0.0, 1.00)]
        },
        {
            "label": "Phase +4.40d (OB4)", "days": 4.403,
            "url": "https://sid.erda.dk/share_redirect/df1fMhon6Z/dereddened%2Bderedshifted_spectra/AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57986.974_Phase%2B4.40d_deredz.dat",
            "local_file": os.path.join(base_dir, "OB4_4.40d.dat"),
            "bounds": [(2407.0, 2807.0), (2.50, 4.00), (0.20, 0.28), (0.150, 0.185), (1.0, 15.0), (0.0, 3.5),
                       (0.01, 0.50), (0.0, 1.20), (0.0, 1.20)]
        }
    ]

    labels = ["T_prime", "N_29", "vmax", "vphot", "tau", "trans", "ve", "amp1", "amp2"]
    corner_labels = [r"$T^\prime$", r"$N_{29}$", r"$v_{\max}$", r"$v_{\text{phot}}$", r"$\tau$", r"$\text{trans}$",
                     r"$v_e$", r"$\text{amp}_1$", r"$\text{amp}_2$"]
    labels_dict = {"labels": labels, "corner_labels": corner_labels}

    results_summary = []
    spectra_data = []

    print("\n========================================================")
    print(" [순수 MCMC 샘플링 파이프라인 가동]")
    print("========================================================\n")

    for p_info in phases:
        label, days, url, local_file, bounds = p_info["label"], p_info["days"], p_info["url"], p_info["local_file"], \
        p_info["bounds"]
        time_s = days * 24.0 * 3600.0

        bounds_arr = np.array(bounds)
        low_b, high_b = bounds_arr[:, 0], bounds_arr[:, 1]
        midpoint_guess = 0.5 * (low_b + high_b)

        print(f"--> [{label}] 피팅 시작...")
        wave, flux, err = load_data(url, local_file)
        eff_err = np.maximum(err, 0.05 * np.abs(flux))
        x_fit, y_fit, err_fit = wave[::6], flux[::6], eff_err[::6]

        prob_wrapper = MCMCProbabilityWrapper(x_fit, y_fit, err_fit, time_s, bounds)

        opt_res = minimize(
            prob_wrapper.chi2_for_minimizer, midpoint_guess, method='Nelder-Mead',
            options={'maxiter': 2500, 'xatol': 1e-4, 'fatol': 1e-2}
        )
        center_point = opt_res.x if (opt_res.success and prob_wrapper.log_prior(opt_res.x) > -1e10) else midpoint_guess

        # Prior 유효 중심점 탐색 (워커 초기화 교착 차단)
        valid_center = center_point.copy()
        if prob_wrapper.log_prior(valid_center) <= -1e10:
            for _ in range(20000):
                cand_u = np.random.uniform(low_b, high_b)
                if prob_wrapper.log_prior(cand_u) > -1e10:
                    valid_center = cand_u
                    break

        ndim, nwalkers = len(bounds), 40
        spans = high_b - low_b
        pos = []
        for _ in range(nwalkers):
            p_cand = None
            for _ in range(500):
                cand = valid_center + spans * 0.02 * np.random.randn(ndim)
                cand = np.clip(cand, low_b + 0.001 * spans, high_b - 0.001 * spans)
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

        print(f"    MCMC 샘플링 진행 중 ({nwalkers} Walkers x {mcmc_steps} Steps)...")
        with mp.Pool(processes=ncpu) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, prob_wrapper, pool=pool)
            sampler.run_mcmc(pos, mcmc_steps, progress=True)

        flat_samples = sampler.get_chain(discard=2000, thin=2, flat=True)

        # MCMC 사후분포 샘플 원본 저장
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

        print(
            f"    결과 요약 [{label}]: Red.Chi2={red_chi2_fit:.2f} | D_L={dl_med:.2f} Mpc | T'={popt['T_prime']:.0f}K | trans={popt['trans']:.2f}\n")

        results_summary.append({
            "days": days, "label": label, "popt": popt, "red_chi2": red_chi2_fit, "dl_med": dl_med
        })
        spectra_data.append({
            "days": days, "label": label, "wave": wave, "flux": flux, "x_fit": x_fit, "model_fit": model_fit
        })

    # 플롯 생성 없이 순수 데이터 파일만 즉시 저장
    with open(os.path.join(base_dir, 'spectra_data.json'), 'w') as f:
        json.dump(make_serializable(spectra_data), f)
    with open(os.path.join(base_dir, 'labels_dict.json'), 'w') as f:
        json.dump(make_serializable(labels_dict), f)
    with open(os.path.join(base_dir, 'results_summary.json'), 'w') as f:
        json.dump(make_serializable(results_summary), f)

    print("========================================================")
    print(f" [MCMC 완료] 샘플 및 요약 데이터가 저장되었습니다: {base_dir}")
    print(" 이제 'python scripts/plot_results_2.py'를 실행하여 플롯을 생성하세요.")
    print("========================================================\n")


if __name__ == "__main__":
    mp.freeze_support()
    run_mcmc()