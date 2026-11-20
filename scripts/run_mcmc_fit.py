#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
scripts/run_mcmc_fit.py
10-에포크(+1.43d ~ +10.40d) torch.compile 가속 GPU 상대론적 NLTE MCMC 피팅 및 수렴 진단 실행기
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import torch
from scipy.optimize import minimize
import emcee

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.settings import (
    PHASES_CONFIG, PARAM_NAMES, N_WALKERS, N_STEPS, BURN_IN, THIN_STEP,
    Z_COSMIC_MEAN, Z_COSMIC_ERR, Q0_DECEL
)
from src.data_loader import load_clean_spectrum
from src.continuum import DEVICE
from src.models import planck_with_mod_full_relativistic, lum_dist_arr
from src.probability import MCMCProbabilityWrapper

warnings.filterwarnings("ignore")


def compute_convergence_diagnostics(sampler, burn_in=10000):
    """
    9차원 MCMC 체인의 평균 수락률(a_f), 자기상관 시간(tau), Split Gelman-Rubin 통계량(R-hat) 산출
    """
    acc_frac = float(np.mean(sampler.acceptance_fraction))

    try:
        tau_arr = sampler.get_autocorr_time(discard=burn_in, tol=0)
        tau_list = [float(t) for t in tau_arr]
        tau_max = float(np.max(tau_arr))
    except Exception:
        tau_list = [None] * sampler.ndim
        tau_max = np.nan

    post_chain = sampler.get_chain(discard=burn_in)
    n_half = post_chain.shape[0] // 2
    split_chains = np.concatenate([post_chain[:n_half], post_chain[n_half:2 * n_half]], axis=1)
    N = split_chains.shape[0]

    W = np.mean(np.var(split_chains, axis=0, ddof=1), axis=0)
    B_over_n = np.var(np.mean(split_chains, axis=0), axis=0, ddof=1)
    var_hat = ((N - 1.0) / N) * W + B_over_n
    r_hat = np.sqrt(var_hat / np.maximum(W, 1e-30))

    total_steps = post_chain.shape[0] + burn_in
    return {
        "acceptance_fraction": acc_frac,
        "autocorr_time_tau_per_param": tau_list,
        "max_autocorr_time_tau": tau_max if np.isfinite(tau_max) else None,
        "n_steps_over_tau": float(total_steps / tau_max) if np.isfinite(tau_max) else None,
        "r_hat_per_param": [float(r) for r in r_hat],
        "max_r_hat": float(np.max(r_hat)),
    }


def main():
    out_dir = os.path.join(PROJECT_ROOT, "output_results", "GPU_10Epoch_Relativistic_NLTE")
    os.makedirs(out_dir, exist_ok=True)

    print("==========================================================================", flush=True)
    print(" [AT2017gfo 10-에포크 torch.compile 가속 GPU MCMC & 정량 수렴 진단 파이프라인]", flush=True)
    print(f"  * 연산 디바이스 : {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""), flush=True)
    print(f"  * MCMC 규격    : {N_WALKERS} Walkers (Half-Batch={N_WALKERS//2}) x {N_STEPS} Steps (Burn-in: {BURN_IN})", flush=True)
    print("==========================================================================\n", flush=True)

    rng = np.random.default_rng(2026)
    results_summary = []
    photospheric_dl_chains = []

    for idx, p_cfg in enumerate(PHASES_CONFIG):
        t_start = time.time()
        label = p_cfg["label"]
        days = p_cfg["days"]
        local_path = os.path.join(out_dir, p_cfg["filename"])

        print(f"--> [{idx+1}/10] {label} 데이터 로드 및 Nelder-Mead 전역 최적화...", flush=True)
        w_fit, f_fit, e_fit, _, _ = load_clean_spectrum(p_cfg["url"], local_path, subsample_step=8)

        prob_fn = MCMCProbabilityWrapper(
            w_fit, f_fit, e_fit, days=days,
            bounds=p_cfg["bounds"], is_photospheric=p_cfg["is_photospheric"]
        )

        init_guess = np.array(p_cfg["init_guess"], dtype=np.float64)
        best_opt_x = init_guess.copy()
        best_opt_val = prob_fn.chi2_for_minimizer(init_guess)

        for trial in range(4):
            start_pt = init_guess if trial == 0 else np.clip(
                init_guess + 0.04 * prob_fn.spans * rng.standard_normal(len(init_guess)),
                prob_fn.low_b + 0.02 * prob_fn.spans,
                prob_fn.high_b - 0.02 * prob_fn.spans
            )
            opt_res = minimize(
                prob_fn.chi2_for_minimizer, start_pt, method="Nelder-Mead",
                options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-4}
            )
            if opt_res.fun < best_opt_val and np.isfinite(prob_fn(opt_res.x)):
                best_opt_val = opt_res.fun
                best_opt_x = opt_res.x.copy()

        ndim = len(PARAM_NAMES)
        pos = []
        for _ in range(N_WALKERS):
            for _try in range(3000):
                cand = best_opt_x + 0.01 * prob_fn.spans * rng.standard_normal(ndim)
                cand = np.clip(cand, prob_fn.low_b + 0.003 * prob_fn.spans, prob_fn.high_b - 0.003 * prob_fn.spans)
                if cand[3] < cand[2] - 0.022 and np.isfinite(prob_fn(cand)):
                    pos.append(cand)
                    break
            else:
                pos.append(best_opt_x)
        pos = np.array(pos, dtype=np.float64)

        # 고정 배치 크기(N_WALKERS // 2 = 64) 1회 워밍업으로 torch.compile 커널 컴파일 완료
        _ = prob_fn.evaluate_batch(pos[: N_WALKERS // 2])
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        print(f"    최적 중심점: T'={best_opt_x[0]:.0f}K, N29={best_opt_x[1]:.3f}, vphot={best_opt_x[3]:.3f}c, trans={best_opt_x[5]:.2f}", flush=True)
        print(f"    GPU MCMC 샘플링 가동 ({N_WALKERS} Walkers x {N_STEPS} Steps)...", flush=True)

        moves = [
            (emcee.moves.DEMove(), 0.8),
            (emcee.moves.DESnookerMove(), 0.2),
        ]
        sampler = emcee.EnsembleSampler(N_WALKERS, ndim, prob_fn, vectorize=True, moves=moves)
        sampler.run_mcmc(pos, N_STEPS, progress=True)

        conv_stats = compute_convergence_diagnostics(sampler, burn_in=BURN_IN)
        flat_samples = sampler.get_chain(discard=BURN_IN, thin=THIN_STEP, flat=True)
        samples_path = os.path.join(out_dir, f"samples_{days:.2f}d.npy")
        np.save(samples_path, flat_samples)

        popt = {}
        param_stats = {}
        for j, pname in enumerate(PARAM_NAMES):
            p16, p50, p84 = np.percentile(flat_samples[:, j], [16, 50, 84])
            popt[pname] = float(p50)
            param_stats[pname] = {
                "median": float(p50),
                "err_minus": float(p50 - p16),
                "err_plus": float(p84 - p50),
            }

        dl_chain = lum_dist_arr(flat_samples[:, 1], flat_samples[:, 3], n_days=days)
        dl_p16, dl_p50, dl_p84 = np.percentile(dl_chain, [16, 50, 84])
        if p_cfg["is_photospheric"]:
            photospheric_dl_chains.append(dl_chain)

        mod_best = planck_with_mod_full_relativistic(
            w_fit, popt["T_prime"], popt["N_29"], popt["vmax"], popt["vphot"],
            tau=popt["tau"], trans=popt["trans"], ve=popt["ve"],
            amp1=popt["amp1"], amp2=popt["amp2"], t0=days * 86400.0, use_nlte=True
        )
        red_chi2 = float(np.sum(((f_fit - mod_best) / e_fit) ** 2) / max(1, len(w_fit) - ndim))
        elapsed = time.time() - t_start

        tau_str = f"{conv_stats['max_autocorr_time_tau']:.1f}" if conv_stats["max_autocorr_time_tau"] else "N/A"
        print(
            f"  ✓ [{label}] 완료 ({elapsed:.1f}s) | Red.Chi2={red_chi2:.2f} | "
            f"a_f={conv_stats['acceptance_fraction']:.2f} | max_tau={tau_str} | max_Rhat={conv_stats['max_r_hat']:.4f}\n"
            f"    -> T'={popt['T_prime']:.0f}K | N29={popt['N_29']:.3f} | vphot={popt['vphot']:.3f}c | "
            f"trans={popt['trans']:.2f} | D_L={dl_p50:.1f} Mpc\n",
            flush=True
        )

        results_summary.append({
            "label": label,
            "days": float(days),
            "filename": p_cfg["filename"],
            "is_photospheric": p_cfg["is_photospheric"],
            "reduced_chi2": red_chi2,
            "dl_med": float(dl_p50),
            "dl_err_minus": float(dl_p50 - dl_p16),
            "dl_err_plus": float(dl_p84 - dl_p50),
            "convergence": conv_stats,
            "popt": popt,
            "parameters": param_stats,
        })

    min_len = min(len(c) for c in photospheric_dl_chains)
    chains_trim = np.array([c[:min_len] for c in photospheric_dl_chains])
    vars_ep = np.var(chains_trim, axis=1, keepdims=True)
    inv_vars = 1.0 / np.maximum(vars_ep, 0.1)
    weights_ep = inv_vars / np.sum(inv_vars)
    dl_comb = np.sum(chains_trim * weights_ep, axis=0)

    z_samples = rng.normal(Z_COSMIC_MEAN, Z_COSMIC_ERR, size=min_len)
    c_kms = 299792.458
    h0_comb = (c_kms * z_samples / dl_comb) * (1.0 + 0.5 * (1.0 - Q0_DECEL) * z_samples)

    dl_c16, dl_c50, dl_c84 = np.percentile(dl_comb, [16, 50, 84])
    h0_c16, h0_c50, h0_c84 = np.percentile(h0_comb, [16, 50, 84])

    final_output = {
        "Combined_Photospheric_DL_Mpc": {
            "median": float(dl_c50),
            "err_minus": float(dl_c50 - dl_c16),
            "err_plus": float(dl_c84 - dl_c50),
        },
        "Combined_H0_km_s_Mpc": {
            "median": float(h0_c50),
            "err_minus": float(h0_c50 - h0_c16),
            "err_plus": float(h0_c84 - h0_c50),
        },
        "epochs": results_summary,
    }

    summary_path = os.path.join(out_dir, "fit_summary_10epochs.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    print("==========================================================================", flush=True)
    print(" [10-에포크 torch.compile 가속 MCMC 샘플링 및 수렴 진단 완료]", flush=True)
    print(f"  * 결합 광도 거리 D_L (1~4일차) = {dl_c50:.2f} (+{dl_c84 - dl_c50:.2f} / -{dl_c50 - dl_c16:.2f}) Mpc", flush=True)
    print(f"  * 결합 허블 상수 H_0           = {h0_c50:.2f} (+{h0_c84 - h0_c50:.2f} / -{h0_c50 - h0_c16:.2f}) km/s/Mpc", flush=True)
    print("==========================================================================\n", flush=True)


if __name__ == "__main__":
    main()
