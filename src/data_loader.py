#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
src/data_loader.py
X-shooter 전 파장 대역 (3,500 ~ 22,500 Å) 분광 데이터 전처리
(텔루릭 흡수대 및 UV 극한 노이즈 정밀 노치 마스킹)
"""

import os
import ssl
import shutil
import urllib.request
import numpy as np
import pandas as pd


def load_data(url, local_filename="temp_spectrum.dat"):
    local_dir = os.path.dirname(os.path.abspath(local_filename))
    os.makedirs(local_dir, exist_ok=True)
    filename = os.path.basename(local_filename)

    if not os.path.exists(local_filename):
        found = False
        search_dirs = [
            os.path.join(os.getcwd(), "output"),
            os.path.join(os.getcwd(), "..", "output"),
            os.path.join(os.getcwd(), "output_results"),
            os.getcwd(),
        ]
        for sdir in search_dirs:
            candidate = os.path.join(sdir, filename)
            if os.path.exists(candidate):
                shutil.copy(candidate, local_filename)
                found = True
                break

        if not found:
            try:
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE

                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, context=ctx) as resp, open(local_filename, "wb") as out_f:
                    out_f.write(resp.read())
            except Exception as e:
                raise RuntimeError(f"다운로드 실패: {e}")

    df = pd.read_csv(local_filename, sep=r"\s+", comment="#", header=None, on_bad_lines="skip")
    wave_raw = pd.to_numeric(df[0], errors="coerce").values
    flux_raw = pd.to_numeric(df[1], errors="coerce").values
    err_raw = pd.to_numeric(df[3], errors="coerce").values

    valid = ~np.isnan(wave_raw) & ~np.isnan(flux_raw) & ~np.isnan(err_raw)
    wave, flux, err = wave_raw[valid], flux_raw[valid], err_raw[valid]
    if wave.max() < 3000:
        wave = wave * 10.0

    # [전 파장 대역 정규 피팅]: 3,500 ~ 22,000 Å 전역을 피팅하되,
    # 지상 대기 수증기 텔루릭 흡수대 및 기기 경계 잡음 구간만 제외
    exc_reg = (
        (wave >= 3500.0) & (wave <= 22000.0)
        & ~((wave > 13100.0) & (wave < 14400.0))  # 텔루릭 J/H 밴드 사이
        & ~((wave > 17550.0) & (wave < 19200.0))  # 텔루릭 H/K 밴드 사이
        & ~((wave > 5330.0) & (wave < 5740.0))    # 2일차 VIS/NIR 접합 노이즈
    )
    return wave[exc_reg], flux[exc_reg], err[exc_reg]