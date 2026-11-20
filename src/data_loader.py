#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
src/data_loader.py
X-Shooter 10개 에포크 분광 데이터 로드, 텔루릭/아티팩트 마스킹 및 KPS 10% 오차 모델 적용
"""

import os
import shutil
import subprocess
import urllib.request
import numpy as np
import pandas as pd
from config.settings import TELLURIC_BANDS_AA, ARTIFACT_BANDS_AA


def ensure_spectrum_file(url, local_path):
    if os.path.exists(local_path) and os.path.getsize(local_path) > 1000:
        return local_path

    os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
    phase_tag = os.path.basename(local_path).split("_")[-1].replace(".dat", "")

    search_dirs = [
        "output_results/GPU_10Epoch_Relativistic_NLTE",
        "output_results/Joint_4Epoch_Relativistic_NLTE",
        "output_results/GPU_Relativistic_EPM",
        "output_results/Case1_withLTT_PureLTE",
        "output",
        "data",
    ]
    for sdir in search_dirs:
        if os.path.exists(sdir):
            for fname in os.listdir(sdir):
                if phase_tag in fname and fname.endswith(".dat"):
                    cand = os.path.join(sdir, fname)
                    if os.path.getsize(cand) > 1000 and os.path.abspath(cand) != os.path.abspath(local_path):
                        shutil.copy(cand, local_path)
                        print(f"    [로컬 캐시 복사] {cand} -> {local_path}", flush=True)
                        return local_path

    print(f"    [원격 다운로드] {os.path.basename(local_path)} 수신 중...", flush=True)
    ret = subprocess.run(
        ["curl", "-k", "-L", "--retry", "3", "--connect-timeout", "30", "-s", "-o", local_path, url]
    )
    if ret.returncode == 0 and os.path.exists(local_path) and os.path.getsize(local_path) > 1000:
        return local_path

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=45) as resp, open(local_path, "wb") as out_f:
        out_f.write(resp.read())

    return local_path


def load_clean_spectrum(url, local_path, subsample_step=8):
    ensure_spectrum_file(url, local_path)

    df = pd.read_csv(
        local_path, sep=r"\s+", comment="#", header=None, on_bad_lines="skip"
    )
    wave_raw = pd.to_numeric(df[0], errors="coerce").values.astype(np.float64)
    flux_raw = pd.to_numeric(df[1], errors="coerce").values.astype(np.float64)
    var_col = pd.to_numeric(df[3], errors="coerce").values.astype(np.float64) if df.shape[1] > 3 else (0.10 * np.abs(flux_raw))

    valid = np.isfinite(wave_raw) & np.isfinite(flux_raw) & np.isfinite(var_col)
    wave_all = wave_raw[valid]
    flux_all = flux_raw[valid]
    var_all = var_col[valid]

    if wave_all.max() < 3000.0:
        wave_all = wave_all * 10.0

    if np.median(np.abs(var_all)) < 1e-25:
        sigma_stat = np.sqrt(np.maximum(var_all, 1e-40))
    else:
        sigma_stat = np.maximum(var_all, 1e-21)

    fit_mask = (wave_all >= 3800.0) & (wave_all <= 21500.0) & (flux_all > 0.0)
    for (t_low, t_high) in TELLURIC_BANDS_AA + ARTIFACT_BANDS_AA:
        fit_mask &= ~((wave_all >= t_low) & (wave_all <= t_high))

    wave_fit = wave_all[fit_mask][::subsample_step]
    flux_fit = flux_all[fit_mask][::subsample_step]
    stat_fit = sigma_stat[fit_mask][::subsample_step]

    # KPS Slide 7 규격: 10% 체계적 모델링 오차를 반영하여 과적합 스파이크를 방지하고 자연스러운 사후분포 도출
    char_flux = np.percentile(flux_fit, 85)
    err_fit = np.sqrt(stat_fit ** 2 + (0.10 * flux_fit) ** 2 + (0.04 * char_flux) ** 2)

    return wave_fit, flux_fit, err_fit, wave_all, flux_all
