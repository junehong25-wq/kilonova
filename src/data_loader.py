import os
import urllib.request
import numpy as np
import pandas as pd

def load_data(url, local_filename="temp_spectrum.dat"):
    # 1. 대상 폴더 생성 및 파일이 없으면 URL에서 자동 다운로드
    os.makedirs(os.path.dirname(os.path.abspath(local_filename)), exist_ok=True)
    if not os.path.exists(local_filename):
        print(f"    [다운로드 중] 관측 데이터 캐시 생성: {os.path.basename(local_filename)}")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as resp, open(local_filename, 'wb') as out_f:
                out_f.write(resp.read())
        except Exception as e:
            # 루트 output/ 폴더에 동일 데이터가 있으면 복사해서 fallback
            alt_path = os.path.join(os.getcwd(), "output", os.path.basename(local_filename))
            if os.path.exists(alt_path):
                import shutil
                shutil.copy(alt_path, local_filename)
            else:
                raise RuntimeError(f"데이터 다운로드 및 캐시 탐색 실패: {local_filename} (에러: {e})")

    # 2. 데이터 로드 및 전처리
    df = pd.read_csv(local_filename, sep=r'\s+', comment='#', header=None, on_bad_lines='skip')
    wave_raw = pd.to_numeric(df[0], errors='coerce').values
    flux_raw = pd.to_numeric(df[1], errors='coerce').values
    err_raw = pd.to_numeric(df[3], errors='coerce').values

    valid = ~np.isnan(wave_raw) & ~np.isnan(flux_raw) & ~np.isnan(err_raw)
    wave, flux, err = wave_raw[valid], flux_raw[valid], err_raw[valid]
    if wave.max() < 3000:
        wave = wave * 10.0

    # 대기 텔루릭 흡수선 영역 마스킹
    exc_reg = (
        ~((wave > 13100) & (wave < 14400))
        & ~((wave > 17550) & (wave < 19200))
        & ~((wave > 5330) & (wave < 5740))
        & ~((wave > 9840) & (wave < 10300))
        & (wave >= 3800) & (wave <= 21500)
    )
    return wave[exc_reg], flux[exc_reg], err[exc_reg]