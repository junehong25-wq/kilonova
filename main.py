#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import inspect
import json
import warnings
import multiprocessing as mp
import numpy as np
from scipy.optimize import minimize
import emcee

from config.settings import fit_cases, phases_template
from src.data_loader import load_data
from src.models import planck_with_mod_full_relativistic, lum_dist_arr
from utils.plotting import generate_all_plots

from src.probability import MCMCProbabilityWrapper
warnings.filterwarnings("ignore")


def make_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    if isinstance(obj, (np.int32, np.int64, np.integer)):
        return int(obj)
    if isinstance(obj, (np.float32, np.float64, np.floating)):
        return float(obj)
    return obj


def main():
    target_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_results")
    os.makedirs(target_base_dir, exist_ok=True)

    # 로컬 CPU 자원 분배 (PC 멈춤 방지용 여유 코어 확보)
    ncpu = max(1, (os.cpu_count() or 4) - 2)
    mcmc_steps = 10000

    print(f"=== Kilonova 4-Case Matrix MCMC 파이프라인 가동 (CPU Cores: {ncpu}) ===")

    for case_cfg in fit_cases:
        case_id = case_cfg["case_id"]
        use_nlte = case_cfg["use_nlte"]
        use_he = case_cfg["use_he"]
        use_ltt = case_cfg.get("use_ltt", "withLTT" in case_id)

        target_save_dir = os.path.join(target_base_dir, case_id)
        os.makedirs(target_save_dir, exist_ok=True)

        if use_he:
            labels = ["T_prime", "N_29", "vmax", "vphot", "tau_sr", "tau_he", "trans"]
            corner_labels = [
                r"$T^\prime$", r"$N_{29}$", r"$v_{\max}$", r"$v_{\text{phot}}$",
                r"$\tau_{\text{Sr II}}$", r"$\tau_{\text{He I}}$", r"$\text{trans}$"
            ]
        else:
            labels = ["T_prime", "N_29", "vmax", "vphot", "tau_sr", "trans"]
            corner_labels = [
                r"$T^\prime$", r"$N_{29}$", r"$v_{\max}$", r"$v_{\text{phot}}$",
                r"$\tau_{\text{Sr II}}$", r"$\text{trans}$"
            ]

        # 각 케이스별 데이터 구조 초기화
        epoch_summaries = []
        results_summary = []
        spectra_data = []
        flat_samples_dict = {}
        labels_dict = {"labels": labels, "corner_labels": corner_labels}

        print(f"\n========================================================")
        print(f" [피팅 진행 케이스: {case_id} (NLTE={use_nlte}, He={use_he}, LTT={use_ltt})]")
        print(f"========================================================\n")

        for p_info in phases_template:
            label, days, url = p_info["label"], p_info["days"], p_info["url"]
            local_file = os.path.join(target_save_dir, f"OB_{days}d.dat")
            bounds = p_info["bounds_withHe"] if use_he else p_info["bounds_noHe"]
            time_s = days * 24.0 * 3600.0

            bounds_arr = np.array(bounds)
            low_b, high_b = bounds_arr[:, 0], bounds_arr[:, 1]
            midpoint_guess = 0.5 * (low_b + high_b)

            print(f"--> [{label}] 데이터 로드 및 초기 피팅 시작...")
            wave, flux, err = load_data(url, local_file)
            eff_err = np.maximum(err, 0.05 * np.abs(flux))
            x_fit, y_fit, err_fit = wave[::6], flux[::6], eff_err[::6]

            # 래퍼 인자 시그니처 자동 대응
            wrapper_kwargs = {
                "use_nlte": use_nlte,
                "use_he": use_he,
                "days": days
            }
            sig_wrap = inspect.signature(MCMCProbabilityWrapper.__init__).parameters
            if "use_ltt" in sig_wrap:
                wrapper_kwargs["use_ltt"] = use_ltt

            prob_wrapper = MCMCProbabilityWrapper(
                x_fit, y_fit, err_fit, time_s, bounds, **wrapper_kwargs
            )

            # 사전 최적화 (Nelder-Mead)
            opt_res = minimize(
                prob_wrapper.chi2_for_minimizer, midpoint_guess, method='Nelder-Mead',
                options={'maxiter': 2500, 'xatol': 1e-4, 'fatol': 1e-2}
            )
            center_point = opt_res.x if (opt_res.success and prob_wrapper.log_prior(opt_res.x) > -1e10) else midpoint_guess

            # Prior 경계 검증 및 유효 중심점 탐색 (Phase 4 교착 상태 원천 차단)
            valid_center = center_point.copy()
            if prob_wrapper.log_prior(valid_center) <= -1e10:
                for _ in range(20000):
                    cand_u = np.random.uniform(low_b, high_b)
                    if prob_wrapper.log_prior(cand_u) > -1e10:
                        valid_center = cand_u
                        break

            ndim, nwalkers = len(bounds), 32
            spans = high_b - low_b
            pos = []
            for _w in range(nwalkers):
                p_cand = None
                # 1. 유효 중심점 주변 탐색 (2% 스케일)
                for _ in range(500):
                    cand = valid_center + spans * 0.02 * np.random.randn(ndim)
                    cand = np.clip(cand, low_b + 0.001 * spans, high_b - 0.001 * spans)
                    if prob_wrapper.log_prior(cand) > -1e10:
                        p_cand = cand
                        break
                # 2. 중심점 주변 미발견 시 균등 분포 영역 탐색
                if p_cand is None:
                    for _ in range(10000):
                        cand = np.random.uniform(low_b, high_b)
                        if prob_wrapper.log_prior(cand) > -1e10:
                            p_cand = cand
                            break
                if p_cand is None:
                    p_cand = valid_center
                pos.append(p_cand)
            pos = np.array(pos)

            print(f"    MCMC 샘플링 진행 중 ({nwalkers} Walkers x {mcmc_steps} Steps)...")
            with mp.Pool(processes=ncpu) as pool:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, prob_wrapper, pool=pool)
                # 터미널에서 진행률 실시간 확인 (progress=True)
                sampler.run_mcmc(pos, mcmc_steps, progress=True)

            flat_samples = sampler.get_chain(
                discard=min(1500, int(mcmc_steps * 0.15)),
                thin=max(1, int(mcmc_steps * 0.0002)),
                flat=True
            )
            flat_samples_dict[days] = flat_samples

            popt = {}
            for i in range(ndim):
                mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
                popt[labels[i]] = mcmc[1]
            if not use_he:
                popt["tau_he"] = 0.0

            model_kwargs = {
                "tau_sr": popt["tau_sr"],
                "tau_he": popt["tau_he"],
                "trans": popt["trans"],
                "t0": time_s,
                "use_nlte": use_nlte,
                "use_he": use_he
            }
            if "amp1" in popt:
                model_kwargs["amp1"] = popt["amp1"]
            if "amp2" in popt:
                model_kwargs["amp2"] = popt["amp2"]

            model_fit = planck_with_mod_full_relativistic(
                x_fit, popt["T_prime"], popt["N_29"], popt["vmax"], popt["vphot"],
                **model_kwargs
            )
            chi2_fit = np.sum(((y_fit - model_fit) / err_fit) ** 2)
            red_chi2_fit = chi2_fit / (len(x_fit) - ndim)

            dl_idx = 5 if not use_he else 6
            dl_samples = lum_dist_arr(flat_samples[:, 1], flat_samples[:, 3], flat_samples[:, dl_idx], n_days=days)
            dl_med = np.median(dl_samples)

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
            epoch_summaries.append(epoch_summary)

            # MCMC 체인 배열 저장
            np.save(os.path.join(target_save_dir, f"samples_{days:.3f}d.npy"), flat_samples)

            results_summary.append({
                "days": days, "label": label, "popt": popt, "red_chi2": red_chi2_fit, "dl_med": dl_med
            })
            spectra_data.append({
                "days": days, "label": label, "wave": wave, "flux": flux, "x_fit": x_fit, "model_fit": model_fit
            })

        print(f"--> [{case_id}] MCMC 연산 완료. 요약 JSON 및 스펙트럼 데이터 저장 중...")
        with open(os.path.join(target_save_dir, 'fit_summary_all.json'), 'w') as f:
            json.dump(epoch_summaries, f, indent=4)

        with open(os.path.join(target_save_dir, 'spectra_data.json'), 'w') as f:
            json.dump(make_serializable(spectra_data), f)

        with open(os.path.join(target_save_dir, 'labels_dict.json'), 'w') as f:
            json.dump(make_serializable(labels_dict), f)

        with open(os.path.join(target_save_dir, 'results_summary.json'), 'w') as f:
            json.dump(make_serializable(results_summary), f)

        for sd in spectra_data:
            d = sd["days"]
            csv_file = os.path.join(target_save_dir, f"spectral_summary_{d:.2f}d.csv")
            np.savetxt(csv_file, np.column_stack((sd["wave"], sd["flux"])), delimiter=",", header="wave,flux")

        # 4개 케이스별 시각화 플롯 (총 15종) 자동 생성 및 호출
        print(f"--> [{case_id}] 시각화 플롯 15종 생성 및 저장 시작...")
        try:
            sig_plot = inspect.signature(generate_all_plots).parameters
            if len(sig_plot) == 8:
                generate_all_plots(
                    spectra_data, results_summary, case_id, use_nlte, use_he,
                    flat_samples_dict, labels_dict, target_save_dir
                )
            else:
                plot_kw = {
                    "spectra_data": spectra_data,
                    "results_summary": results_summary,
                    "case_id": case_id,
                    "use_nlte": use_nlte,
                    "use_he": use_he,
                    "flat_samples_dict": flat_samples_dict,
                    "labels_dict": labels_dict,
                    "target_save_dir": target_save_dir
                }
                filtered_kw = {k: v for k, v in plot_kw.items() if k in sig_plot}
                generate_all_plots(**filtered_kw)
            print(f"    [{case_id}] 모든 도판 저장 완료.")
        except Exception as e:
            print(f"    [경고] 플롯 자동 생성 중 오류 발생 ({e}). 데이터는 안전하게 보존되었습니다.")

    print("\n========================================================")
    print(" [총 4개 피팅 케이스 연산 및 각 케이스별 15종 플롯 완전 저장 완료]")
    print("========================================================\n")


if __name__ == "__main__":
    mp.freeze_support()
    main()