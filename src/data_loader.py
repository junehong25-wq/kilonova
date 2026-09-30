import os
import ssl
import shutil
import urllib.request
import numpy as np
import pandas as pd


def load_data(url, local_filename="temp_spectrum.dat"):
    """
    관측 분광 데이터를 다운로드하거나 로컬 캐시에서 불러오며,
    대기 흡수선 마스킹 및 전처리를 수행하여 (wave, flux, err) 배열을 반환합니다.
    """
    local_dir = os.path.dirname(os.path.abspath(local_filename))
    os.makedirs(local_dir, exist_ok=True)
    filename = os.path.basename(local_filename)

    # 1. 파일이 없으면 기존 프로젝트 폴더(output 등)에서 탐색하거나 웹에서 다운로드
    if not os.path.exists(local_filename):
        found = False
        search_dirs = [
            os.path.join(os.getcwd(), "output"),
            os.path.join(os.getcwd(), "..", "output"),
            os.getcwd(),
        ]
        for sdir in search_dirs:
            candidate = os.path.join(sdir, filename)
            if os.path.exists(candidate):
                shutil.copy(candidate, local_filename)
                found = True
                print(f"    [로컬 캐시 사용] {filename} 발견")
                break

        # 로컬에도 없으면 웹 URL에서 자동 다운로드
        if not found:
            print(f"    [다운로드 중] 관측 데이터 다운로드: {filename}")
            try:
                # Windows 환경의 SSL 인증서 검증 오류 방지
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE

                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, context=ctx) as resp, open(local_filename, "wb") as out_f:
                    out_f.write(resp.read())
            except Exception as e:
                raise RuntimeError(
                    f"관측 데이터 다운로드 실패 ({filename}): {e}\n"
                    f"인터넷 연결을 확인하거나 {local_filename} 경로에 파일을 수동으로 배치하세요."
                )

    # 2. 데이터 파일 파싱
    if not os.path.exists(local_filename):
        raise FileNotFoundError(f"데이터 파일이 존재하지 않습니다: {local_filename}")

    df = pd.read_csv(local_filename, sep=r"\s+", comment="#", header=None, on_bad_lines="skip")
    wave_raw = pd.to_numeric(df[0], errors="coerce").values
    flux_raw = pd.to_numeric(df[1], errors="coerce").values
    err_raw = pd.to_numeric(df[3], errors="coerce").values

    valid = ~np.isnan(wave_raw) & ~np.isnan(flux_raw) & ~np.isnan(err_raw)
    wave, flux, err = wave_raw[valid], flux_raw[valid], err_raw[valid]
    if wave.max() < 3000:
        wave = wave * 10.0

    # 3. 대기 텔루릭 흡수선 마스킹 (Sr II 1.0um 방출 피크 9840~10300Å는 온전히 보존)
    exc_reg = (
        ~((wave > 13100) & (wave < 14400))
        & ~((wave > 17550) & (wave < 19200))
        & ~((wave > 5330) & (wave < 5740))
        & (wave >= 3800) & (wave <= 21500)
    )
    return wave[exc_reg], flux[exc_reg], err[exc_reg]