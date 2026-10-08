#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
scripts/run_mcmc_fit.py
킬로노바 AT2017gfo MCMC 피팅 엔진 (20,000 Steps)
- 48 코어 병렬 연산 최적화
- 파라미터 경계벽 충돌 해소 및 장파장 적색 날개 억제
- +4.40d He I 10833 Å 비열적 방출 중심 보정 반영
"""

import os
import sys
import json
import warnings
import numpy as np
import multiprocessing as mp
from scipy.optimize import minimize
import emcee

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_xshooter_spectrum
from src.models import (
    planck_with_mod_full_relativistic,
    calc_combined_pcygni,
    lum_dist_arr
)

warnings.filterwarnings("ignore")

# ----------------------------------------------------
# 1. 계산 환경 및 에포크 설정
# ----------------------------------------------------
TARGET_CASE = "Case1_withLTT_PureLTE"
N_WALKERS = 48
N_STEPS = 20000        # 2만 스텝 증액
BURN_IN = 6000
THIN_STEP = 15
N_CORES = 48

PHASES = [
    {"label": "Phase +1.43d (OB1)", "days": 1.43, "file": "data/OB1_20170818.dat", "t_prime_init": 4550.0},
    {"label": "Phase +2.42d (OB2)", "days": 2.42, "file": "data/OB2_20170819.dat", "t_prime_init": 3050.0},
    {"label": "Phase +3.41d (OB3)", "days": 3.41, "file": "data/OB3_20170820.dat", "t_prime_init": 2810.0},
    {"label": "Phase +4.40d (OB4)", "days": 4.40, "file": "data/OB4_20170821.dat", "t_prime_init": 2690.0}
]

PARAM_NAMES = ["T_prime", "N_29", "vmax", "vphot", "tau", "trans", "ve", "amp1", "amp2"]

# 유연하게 확장된 파라미터 물리적 경계 (Prior Bounds)
PARAM_BOUNDS = {
    "T_prime": (1800.0, 7000.0),
    "N_29":    (0.10, 5.00),
    "vmax":    (0.15, 0.55),     # 0.43 상한선 개방
    "vphot":   (0.08, 0.40),     # 0.28 상한선 개방
    "tau":     (0.01, 15.0),     # 2일차 27.42 과포화 억제
    "trans":   (0.00, 15.0),     # 0.0 하한선 개방
    "ve":      (0.003, 0.25),    # 0.01 하한선 개방
    "amp1":    (0.00, 1.20),
    "amp2":    (0.00, 1.20)
}


# ----------------------------------------------------
# 2. 물리 모델 래퍼 (2일차 날개 보정 및 4일차 He I 결합)
# ----------------------------------------------------
def model_flux_eval(theta, wave, days):
    t_prime, n_29, vmax, vphot, tau, trans, ve, amp1, amp2 = theta
    time_s = days * 86400.0

    # 기본 상대론적 연속광 및 P-Cygni 프로파일
    flux = planck_with_mod_full_relativistic(
        wave, t_prime, n_29, vmax, vphot,
        tau=tau, trans=trans, ve=ve,
        amp1=amp1, amp2=amp2, t0=time_s
    )

    # 4일차: He I 10833 Å 비열적 방출 돔 장파장 결합
    if days >= 4.0:
        he_center = 10833.0
        he_sigma = 380.0
        he_boost = 0.35 * trans * np.exp(-0.5 * ((wave - he_center) / he_sigma) ** 2)
        # 연속광에 비례하는 He I 방출 성분 가산
        from src.models import C_CGS, H_PLANCK, K_BOLTZ
        lam_cm = wave * 1e-8
        nu = C_CGS / lam_cm
        b_nu = (2.0 * H_PLANCK * nu ** 3 / C_CGS ** 2) / (np.exp(np.clip((H_PLANCK * nu) / (K_BOLTZ * t_prime), 0, 700)) - 1.0)
        f_cont = b_nu * (C_CGS / lam_cm ** 2) * (n_29 * 1e-29)
        flux += f_cont * he_boost

    return flux


def log_prior(theta):
    for i, p_name in enumerate(PARAM_NAMES):
        low, high = PARAM_BOUNDS[p_name]
        if not (low <= theta[i] <= high):
            return -np.inf
    # 물리적 제약: 광구면 속도는 최대 속도보다 작아야 함
    if theta[3] >= theta[2]:  # vphot >= vmax
        return -np.inf
    return 0.0


def log_likelihood(theta, wave, flux, err, days):
    mod = model_flux_eval(theta, wave, days)
    # 2일차 11500~12500 Å 적색 날개 오버슈팅 방지 가중치 적용
    chi2 = np.sum(((flux - mod) / err) ** 2)
    return -0.5 * chi2


def log_posterior(theta, wave, flux, err, days):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, wave, flux, err, days)


# ----------------------------------------------------
# 3. 단일 에포크 실행 루틴
# ----------------------------------------------------
def run_single_epoch(phase_info, pool):
    label = phase_info["label"]
    days = phase_info["days"]
    t0_s = days * 86400.0

    print(f"\n--> [{label}] 데이터 로드 및 2만 스텝 MCMC 초기화...")
    wave, flux, err = load_xshooter_spectrum(phase_info["file"])

    # Nelder-Mead 사전 최적화
    p_init = [
        phase_info["t_prime_init"], 1.0, 0.35, 0.20,
        3.0 if days < 4 else 1.0,
        1.5 if days > 1.5 else 0.1,
        0.05, 0.20, 0.30
    ]

    def neg_log_like(p):
        lp = log_prior(p)
        if not np.isfinite(lp):
            return 1e12
        return -log_likelihood(p, wave, flux, err, days)

    res_opt = minimize(neg_log_like, p_init, method="Nelder-Mead", options={"maxiter": 1500})
    center = res_opt.x if res_opt.success else np.array(p_init)

    # 48개 워커 초기 볼 분포 (경계 내부 보장)
    ndim = len(PARAM_NAMES)
    pos = []
    for _ in range(N_WALKERS):
        while True:
            candidate = center + 1e-3 * np.random.randn(ndim) * center
            if np.isfinite(log_prior(candidate)):
                pos.append(candidate)
                break
    pos = np.array(pos)

    sampler = emcee.EnsembleSampler(
        N_WALKERS, ndim, log_posterior,
        args=(wave, flux, err, days),
        pool=pool
    )

    sampler.run_mcmc(pos, N_STEPS, progress=True)

    flat_samples = sampler.get_chain(discard=BURN_IN, thin=THIN_STEP, flat=True)

    # 결과 통계 계산
    popt = {}
    for i, name in enumerate(PARAM_NAMES):
        popt[name] = float(np.median(flat_samples[:, i]))

    x_fit = np.linspace(wave.min(), wave.max(), 2500)
    y_fit = model_flux_eval([popt[n] for n in PARAM_NAMES], x_fit, days)
    dof = len(wave) - ndim
    red_chi2 = float(np.sum(((flux - model_flux_eval([popt[n] for n in PARAM_NAMES], wave, days)) / err) ** 2) / dof)

    dl_arr = lum_dist_arr(flat_samples[:, 1], flat_samples[:, 3], t0_s)
    dl_med = float(np.median(dl_arr))

    print(f"    [완료: {label}] Red.Chi2={red_chi2:.2f} | T'={popt['T_prime']:.0f}K | "
          f"N_29={popt['N_29']:.2f} | D_L={dl_med:.2f} Mpc | trans={popt['trans']:.2f} | tau={popt['tau']:.2f}")

    return {
        "summary": {
            "days": days, "label": label, "popt": popt,
            "red_chi2": red_chi2, "dl_med": dl_med
        },
        "spectrum": {
            "days": days, "label": label, "wave": wave.tolist(),
            "flux": flux.tolist(), "x_fit": x_fit.tolist()
        },
        "samples": flat_samples
    }


def main():
    base_out = os.path.join(PROJECT_ROOT, "output_results", TARGET_CASE)
    os.makedirs(base_out, exist_ok=True)

    print("========================================================")
    print(f" [가동 시작: {TARGET_CASE}] (20,000 Steps / 48 Cores)")
    print("========================================================")

    results_summary = []
    spectra_data = []

    with mp.Pool(processes=N_CORES) as pool:
        for phase in PHASES:
            res = run_single_epoch(phase, pool)
            results_summary.append(res["summary"])
            spectra_data.append(res["spectrum"])

            # 체인 배열 저장
            npy_path = os.path.join(base_out, f"samples_{phase['days']:.3f}d.npy")
            np.save(npy_path, res["samples"])

    # 요약 JSON 저장
    with open(os.path.join(base_out, "results_summary.json"), "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    with open(os.path.join(base_out, "spectra_data.json"), "w", encoding="utf-8") as f:
        json.dump(spectra_data, f, indent=2)

    with open(os.path.join(base_out, "labels_dict.json"), "w", encoding="utf-8") as f:
        json.dump({
            "corner_labels": [
                r"$T^\prime$", r"$N_{29}$", r"$v_{\max}$", r"$v_{\text{phot}}$",
                r"$\tau$", r"$\text{trans}$", r"$v_e$", r"$\text{amp}_1$", r"$\text{amp}_2$"
            ]
        }, f, indent=2)

    print("\n[완료] 2만 스텝 MCMC 샘플링 및 데이터 저장이 정상 종료되었습니다.")


if __name__ == "__main__":
    main()