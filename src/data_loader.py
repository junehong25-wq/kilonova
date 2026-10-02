#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
src/data_loader.py
관측 분광 데이터 전처리 (가시광선 블랭킷팅 간섭 배제: 7000 ~ 21500 Å)
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

    # [핵심]: 가시광선 블랭킷팅 붕괴 영역(wave < 7000 Å)을 배제하여 단일 흑체 물리 정합성 확보
    exc_reg = (
        ~((wave > 13100) & (wave < 14400))
        & ~((wave > 17550) & (wave < 19200))
        & ~((wave > 5330) & (wave < 5740))
        & (wave >= 7000) & (wave <= 21500)
    )
    return wave[exc_reg], flux[exc_reg], err[exc_reg]