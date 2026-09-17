import os
import io
import re
import urllib.request
import pandas as pd
import numpy as np

def load_data(url, local_filename="temp_spectrum.dat"):
    if os.path.exists(local_filename):
        try:
            df = pd.read_csv(local_filename, sep=r'\s+', comment='#', header=None, on_bad_lines='skip')
            wave_raw = pd.to_numeric(df[0], errors='coerce').values
            flux_raw = pd.to_numeric(df[1], errors='coerce').values
            err_raw = pd.to_numeric(df[3], errors='coerce').values
            valid = ~np.isnan(wave_raw) & ~np.isnan(flux_raw) & ~np.isnan(err_raw)
            wave, flux, err = wave_raw[valid], flux_raw[valid], err_raw[valid]
            if wave.max() < 3000:
                wave = wave * 10.0
            exc_reg = (~((wave > 13100) & (wave < 14400))) & (~((wave > 17550) & (wave < 19200))) & (
                ~((wave > 5330) & (wave < 5740))) & (~((wave > 9950) & (wave < 10250))) & (wave >= 3800) & (wave <= 21500)
            return wave[exc_reg], flux[exc_reg], err[exc_reg]
        except Exception:
            pass

    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        raw_data = response.read().decode('utf-8')

    try:
        with open(local_filename, 'w', encoding='utf-8') as f:
            f.write(raw_data)
    except Exception:
        pass

    raw_data = re.sub(r'(?<=\d)[Dd](?=[+-]?\d)', 'E', raw_data)
    df = pd.read_csv(io.StringIO(raw_data), sep=r'\s+', comment='#', header=None, on_bad_lines='skip')
    wave_raw, flux_raw, err_raw = pd.to_numeric(df[0], errors='coerce').values, pd.to_numeric(df[1], errors='coerce').values, pd.to_numeric(df[3], errors='coerce').values
    valid = ~np.isnan(wave_raw) & ~np.isnan(flux_raw) & ~np.isnan(err_raw)
    wave, flux, err = wave_raw[valid], flux_raw[valid], err_raw[valid]
    if wave.max() < 3000:
        wave = wave * 10.0
    exc_reg = (~((wave > 13100) & (wave < 14400))) & (~((wave > 17550) & (wave < 19200))) & (
        ~((wave > 5330) & (wave < 5740))) & (~((wave > 9950) & (wave < 10250))) & (wave >= 3800) & (wave <= 21500)
    return wave[exc_reg], flux[exc_reg], err[exc_reg]
