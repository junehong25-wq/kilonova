#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import multiprocessing as mp
import os
import re
import urllib.request
import warnings
import json
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import constants
from scipy.optimize import minimize

from src.probability import MCMCProbabilityWrapper
from src.continuum import calc_relativistic_blackbody_continuum
from src.models import planck_with_mod_full_relativistic
#

try:
    import corner
except ImportError:
    corner = None

try:
    import emcee
except ImportError:
    raise ImportError("MCMC 실행을 위해 'emcee' 라이브러리가 필요합니다.")

warnings.filterwarnings("ignore")


def lum_dist_arr(N_29_array, vphot_array, trans_array=1.0, n_days=1.427, dt=0.0):
    c_m = constants.c
    N = np.maximum(N_29_array * 1e-29, 1e-35)
    theta = 2.0 * np.sqrt(N * 5.48e6)
    v = vphot_array * c_m
    t = (n_days - dt) * (3600.0 * 24.0)
    r = v * t
    D = (r / theta) * 2.0
    D_mpc = D * (3.2408e-23)
    return D_mpc


def load_data(url, local_filename="temp_spectrum.dat"):
    if os.path.exists(local_filename):
        try:
            df = pd.read_csv(
                local_filename,
                sep=r"\s+",
                comment="#",
                header=None,
                on_bad_lines="skip",
            )
            wave_raw = pd.to_numeric(df[0], errors="coerce").values
            flux_raw = pd.to_numeric(df[1], errors="coerce").values
            err_raw = pd.to_numeric(df[3], errors="coerce").values
            valid = (
                ~np.isnan(wave_raw) & ~np.isnan(flux_raw) & ~np.isnan(err_raw)
            )
            wave, flux, err = wave_raw[valid], flux_raw[valid], err_raw[valid]
            if wave.max() < 3000:
                wave = wave * 10.0
            exc_reg = (
                (~((wave > 13100) & (wave < 14400)))
                & (~((wave > 17550) & (wave < 19200)))
                & (~((wave > 5330) & (wave < 5740)))
                & (~((wave > 9950) & (wave < 10250)))
                & (wave >= 3800)
                & (wave <= 21500)
            )
            return wave[exc_reg], flux[exc_reg], err[exc_reg]
        except Exception:
            pass

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response:
        raw_data = response.read().decode("utf-8")

    try:
        with open(local_filename, "w", encoding="utf-8") as f:
            f.write(raw_data)
    except Exception:
        pass

    raw_data = re.sub(r"(?<=\d)[Dd](?=[+-]?\d)", "E", raw_data)
    df = pd.read_csv(
        io.StringIO(raw_data),
        sep=r"\s+",
        comment="#",
        header=None,
        on_bad_lines="skip",
    )
    wave_raw, flux_raw, err_raw = (
        pd.to_numeric(df[0], errors="coerce").values,
        pd.to_numeric(df[1], errors="coerce").values,
        pd.to_numeric(df[3], errors="coerce").values,
    )
    valid = ~np.isnan(wave_raw) & ~np.isnan(flux_raw) & ~np.isnan(err_raw)
    wave, flux, err = wave_raw[valid], flux_raw[valid], err_raw[valid]
    if wave.max() < 3000:
        wave = wave * 10.0
    exc_reg = (
        (~((wave > 13100) & (wave < 14400)))
        & (~((wave > 17550) & (wave < 19200)))
        & (~((wave > 5330) & (wave < 5740)))
        & (~((wave > 9950) & (wave < 10250)))
        & (wave >= 3800)
        & (wave <= 21500)
    )
    return wave[exc_reg], flux[exc_reg], err[exc_reg]


def main():
    target_save_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(target_save_dir, exist_ok=True)

    phases = [
        {
            "label": "Phase +1.43d (OB1)",
            "days": 1.427,
            "url": "https://sid.erda.dk/share_redirect/df1fMhon6Z/dereddened%2Bderedshifted_spectra/AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57983.969_Phase%2B1.43d_deredz.dat",
            "local_file": os.path.join(target_save_dir, "OB1_1.43d.dat"),
            "bounds": [
                (4200.0, 6000.0), (0.80, 1.80), (0.35, 0.50), (0.24, 0.35),
                (1.0, 3.0), (0.0, 0.30), (0.01, 0.30)
            ],
            "init_guess": [4900.0, 1.30, 0.40, 0.27, 2.00, 0.05, 0.15],
        },
        {
            "label": "Phase +2.42d (OB2)",
            "days": 2.417,
            "url": "https://sid.erda.dk/share_redirect/df1fMhon6Z/dereddened%2Bderedshifted_spectra/AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57984.969_Phase%2B2.42d_deredz.dat",
            "local_file": os.path.join(target_save_dir, "OB2_2.42d.dat"),
            "bounds": [
                (3100.0, 3800.0), (1.50, 3.20), (0.25, 0.42), (0.16, 0.29),
                (0.7, 2.0), (0.05, 0.40), (0.35, 0.65)
            ],
            "init_guess": [3450.0, 2.20, 0.33, 0.21, 1.20, 0.20, 0.45],
        },
        {
            "label": "Phase +3.41d (OB3)",
            "days": 3.413,
            "url": "https://sid.erda.dk/share_redirect/df1fMhon6Z/dereddened%2Bderedshifted_spectra/AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57985.974_Phase%2B3.41d_deredz.dat",
            "local_file": os.path.join(target_save_dir, "OB3_3.41d.dat"),
            "bounds": [
                (2700.0, 3300.0), (1.0, 5.0), (0.18, 0.37), (0.13, 0.20),
                (0.4, 1.3), (0.2, 0.9), (0.35, 0.65)
            ],
            "init_guess": [2950.0, 2.90, 0.25, 0.16, 0.85, 0.50, 0.48],
        },
        {
            "label": "Phase +4.40d (OB4)",
            "days": 4.403,
            "url": "https://sid.erda.dk/share_redirect/df1fMhon6Z/dereddened%2Bderedshifted_spectra/AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57986.974_Phase%2B4.40d_deredz.dat",
            "local_file": os.path.join(target_save_dir, "OB4_4.40d.dat"),
            "bounds": [
                (2400.0, 2900.0), (2.00, 4.50), (0.15, 0.30), (0.10, 0.16),
                (0.2, 0.9), (0.5, 1.8), (0.35, 0.65)
            ],
            "init_guess": [2650.0, 3.40, 0.21, 0.13, 0.50, 1.00, 0.50],
        },
    ]

    labels = ["T_prime", "N_29", "vmax", "vphot", "tau_sr", "tau_he", "trans"]
    corner_labels = [
        r"$T^\prime$",
        r"$N_{29}$",
        r"$v_{\max}$",
        r"$v_{\text{phot}}$",
        r"$\tau_{\text{Sr II}}$",
        r"$\tau_{\text{He I}}$",
        r"$\text{trans}$",
    ]
    ncpu = max(1, mp.cpu_count() - 2)

    results_summary = []
    spectra_data = []

    print("\n========================================================")
    print(" [Arya+2026 7D Pure NLTE 복합 MCMC 피팅 시작]")
    print("========================================================\n")

    for p_info in phases:
        label, days, url, local_file, bounds = (
            p_info["label"],
            p_info["days"],
            p_info["url"],
            p_info["local_file"],
            p_info["bounds"],
        )
        init_guess = p_info["init_guess"]
        time_s = days * 24.0 * 3600.0

        bounds_arr = np.array(bounds)
        low_b, high_b = bounds_arr[:, 0], bounds_arr[:, 1]

        print(f"--> [{label}] 데이터 로드 및 초기 피팅 시작...")
        wave, flux, err = load_data(url, local_file)
        eff_err = np.maximum(err, 0.05 * np.abs(flux))
        x_fit, y_fit, err_fit = wave[::5], flux[::5], eff_err[::5]

        prob_wrapper = MCMCProbabilityWrapper(
            x_fit, y_fit, err_fit, time_s, bounds, days
        )

        opt_res = minimize(
            prob_wrapper.chi2_for_minimizer,
            init_guess,
            method="Nelder-Mead",
            options={"maxiter": 3000, "xatol": 1e-4, "fatol": 1e-2},
        )
        center_point = (
            opt_res.x
            if (opt_res.success and prob_wrapper.log_prior(opt_res.x) > -1e10)
            else init_guess
        )

        ndim, nwalkers = len(bounds), 32
        nsteps = 10000
        spans = high_b - low_b
        pos = []
        for _ in range(nwalkers):
            while True:
                cand = center_point + spans * 0.005 * np.random.randn(ndim)
                cand = np.clip(
                    cand, low_b + 0.005 * spans, high_b - 0.005 * spans
                )
                if prob_wrapper.log_prior(cand) > -1e10:
                    pos.append(cand)
                    break
        pos = np.array(pos)

        print(
            f"    MCMC 샘플링 진행 중 ({nwalkers} Walkers x {nsteps} Steps)..."
        )
        with mp.Pool(processes=ncpu) as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, prob_wrapper, pool=pool
            )
            sampler.run_mcmc(pos, nsteps, progress=True)

        flat_samples = sampler.get_chain(discard=3000, thin=15, flat=True)

        popt = {}
        for i in range(ndim):
            mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
            popt[labels[i]] = mcmc[1]

        if corner is not None:
            fig_corner = corner.corner(
                flat_samples,
                labels=corner_labels,
                quantiles=[0.16, 0.50, 0.84],
                show_titles=True,
                title_fmt=".3f",
                smooth=1.0,
                levels=(0.68, 0.95),
                fill_contours=True,
                plot_datapoints=False,
            )
            safe_label = re.sub(r"[^a-zA-Z0-9_.]", "_", label)
            safe_label = re.sub(r"_+", "_", safe_label).strip("_")
            fig_corner.savefig(
                os.path.join(target_save_dir, f"{safe_label}_7D_NLTE_corner.png"),
                dpi=200,
            )
            plt.close(fig_corner)

        dl_samples = lum_dist_arr(
            flat_samples[:, 1],
            flat_samples[:, 3],
            flat_samples[:, 6],
            n_days=days,
        )
        dl_med = np.median(dl_samples)

        # Compute chi2 for current best fit
        t_ph = days * 86400.0
        model_fit = planck_with_mod_full_relativistic(
            x_fit, popt["T_prime"], popt["N_29"], popt["vmax"], popt["vphot"],
            tau_sr=popt["tau_sr"], tau_he=popt["tau_he"], trans=popt["trans"], t0=t_ph
        )
        chi2_fit = np.sum(((y_fit - model_fit) / err_fit) ** 2)
        red_chi2_fit = chi2_fit / (len(x_fit) - ndim)

        print(f"    결과 요약 [{label}]: Red.Chi2={red_chi2_fit:.2f} | D_L={dl_med:.2f} Mpc | T'={popt['T_prime']:.0f}K\n")

        epoch_summary = {
            "days": float(days),
            "label": str(label),
            "reduced_chi2": float(red_chi2_fit),
            "luminosity_distance_mpc": {
                "median": float(np.median(dl_samples)),
                "lower_1sigma": float(np.percentile(dl_samples, 16)),
                "upper_1sigma": float(np.percentile(dl_samples, 84))
            },
            "parameters": {
                name: {
                    "median": float(np.percentile(flat_samples[:, i], 50)),
                    "lower_1sigma": float(np.percentile(flat_samples[:, i], 16)),
                    "upper_1sigma": float(np.percentile(flat_samples[:, i], 84))
                }
                for i, name in enumerate(labels)
            }
        }
        if "epoch_summaries" not in locals():
            epoch_summaries = []
        epoch_summaries.append(epoch_summary)

        # Save flat samples for future plotting
        np.save(os.path.join(target_save_dir, f"samples_{days:.3f}d.npy"), flat_samples)

        results_summary.append(
            {"days": days, "label": label, "popt": popt, "red_chi2": red_chi2_fit, "dl_med": dl_med}
        )
        spectra_data.append(
            {"days": days, "label": label, "wave": wave, "flux": flux, "x_fit": x_fit, "model_fit": model_fit}
        )

    print(f"--> MCMC Run completed. Saving lightweight data...")
    with open(os.path.join(target_save_dir, 'fit_summary_all.json'), 'w') as f:
        json.dump(epoch_summaries, f, indent=4)

    def make_serializable(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list): return [make_serializable(v) for v in obj]
        if isinstance(obj, (np.int32, np.int64)): return int(obj)
        if isinstance(obj, (np.float32, np.float64)): return float(obj)
        return obj

    with open(os.path.join(target_save_dir, 'spectra_data.json'), 'w') as f:
        json.dump(make_serializable(spectra_data), f)

    labels_dict = {"labels": labels, "corner_labels": corner_labels}
    with open(os.path.join(target_save_dir, 'labels_dict.json'), 'w') as f:
        json.dump(make_serializable(labels_dict), f)

    with open(os.path.join(target_save_dir, 'results_summary.json'), 'w') as f:
        json.dump(make_serializable(results_summary), f)

    for sd in spectra_data:
        d = sd["days"]
        csv_file = os.path.join(target_save_dir, f"spectral_summary_{d:.2f}d.csv")
        np.savetxt(csv_file, np.column_stack((sd["wave"], sd["flux"])), delimiter=",", header="wave,flux")


if __name__ == "__main__":
    mp.freeze_support()
    main()
