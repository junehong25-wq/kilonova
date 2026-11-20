#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(PROJECT_ROOT, "output_results", "GPU_10Epoch_Relativistic_PureLTE_SrII")

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "mathtext.fontset": "cm",
    "font.family": "serif"
})

CORNER_LABELS = [
    r"$T^\prime\ [\mathrm{K}]$", r"$N_{29}$", r"$v_{\max}\ [c]$",
    r"$v_{\mathrm{phot}}\ [c]$", r"$\tau$", r"$\mathrm{trans}$",
    r"$v_e\ [c]$", r"$\mathrm{amp}_1$", r"$\mathrm{amp}_2$"
]

# 주 모드(Primary Mode) 단일 타원이 중앙에 꽉 차도록 조정한 에포크별 축 범위 [T', N29, vmax, vphot, tau, trans, ve, amp1, amp2]
CUSTOM_RANGES = {
    1.43: [
        (4500.0, 4700.0),   # T': 4400K 부 모드 제외 -> 4600K 단일 타원 집중
        (0.455, 0.495),     # N29: 0.50 부 모드 제외
        (0.385, 0.430),     # vmax: 0.45c 부 모드 제외
        (0.195, 0.235),     # vphot: 0.27c 부 모드 제외
        (0.310, 0.480),     # tau: 0.72 부 모드 제외
        (0.000, 0.220),     # trans
        (0.115, 0.225),     # ve: 0.08c 부 모드 제외
        (0.095, 0.210),     # amp1
        (0.350, 0.540),     # amp2
    ],
    2.42: [
        (3360.0, 3465.0),
        (1.005, 1.085),
        (0.335, 0.370),
        (0.122, 0.160),
        (0.350, 0.500),
        (0.550, 1.320),
        (0.095, 0.165),
        (0.120, 0.215),
        (0.210, 0.325),
    ],
    3.41: [
        (2935.0, 3015.0),   # T'
        (1.285, 1.380),     # N29
        (0.248, 0.285),     # vmax: 우측 0.40c 외딴 점군 제거
        (0.126, 0.156),     # vphot: 상단 0.17c 외딴 점군 제거
        (0.300, 0.530),     # tau: 상단 0.80 외딴 점군 제거
        (3.100, 4.650),     # trans
        (0.055, 0.145),     # ve
        (0.255, 0.350),     # amp1
        (0.480, 0.595),     # amp2
    ],
    4.40: [
        (2640.0, 2685.0),   # T': 양옆 위성 섬 제외 -> 2663K 단일 타원
        (1.425, 1.535),     # N29
        (0.244, 0.258),     # vmax: 좌하단 위성 섬 제외
        (0.165, 0.180),     # vphot: 상/하 위성 섬 제외 -> 0.173c 단일 타원
        (0.170, 0.255),     # tau
        (11.00, 16.50),     # trans
        (0.060, 0.230),     # ve
        (0.335, 0.435),     # amp1
        (0.520, 0.630),     # amp2
    ],
    5.40: [
        (2730.0, 2815.0),
        (0.945, 1.030),
        (0.233, 0.256),
        (0.184, 0.218),
        (0.150, 0.360),
        (10.00, 19.00),
        (0.060, 0.222),
        (0.265, 0.365),
        (0.720, 0.850),
    ],
    6.40: [
        (2600.0, 2648.0),
        (0.860, 0.945),
        (0.250, 0.261),
        (0.213, 0.2205),    # vphot: 하단 긴 꼬리 축소
        (0.800, 1.850),
        (3.000, 5.000),
        (0.038, 0.210),
        (0.365, 0.485),
        (1.120, 1.300),
    ],
    7.40: [
        (2785.0, 2842.0),
        (0.655, 0.715),
        (0.228, 0.246),
        (0.060, 0.074),
        (0.320, 0.450),
        (0.495, 0.700),
        (0.125, 0.222),
        (0.270, 0.380),
        (1.340, 1.510),
    ],
    8.40: [
        (2935.0, 3075.0),
        (0.325, 0.390),
        (0.223, 0.262),     # vmax: 상단 0.32c 부 모드 제외 -> 0.238c 단일 타원
        (0.055, 0.108),
        (0.235, 0.400),     # tau: 상단 0.46 부 모드 제외 -> 0.308 단일 타원
        (0.495, 0.720),
        (0.100, 0.222),
        (0.215, 0.430),
        (1.300, 1.600),
    ],
    9.40: [
        (2640.0, 2735.0),
        (0.240, 0.280),
        (0.231, 0.246),
        (0.184, 0.1905),    # vphot: 하단 꼬리 축소
        (0.880, 1.005),     # tau: 하단 꼬리 축소
        (0.498, 0.560),
        (0.090, 0.222),
        (0.325, 0.490),
        (2.130, 2.470),
    ],
    10.40: [
        (2240.0, 2445.0),   # T': 2150K 부 모드 제외 -> 2349K 단일 타원
        (0.375, 0.525),
        (0.185, 0.250),
        (0.045, 0.115),     # vphot: 0.18c 부 모드 제외 -> 0.072c 단일 타원
        (0.350, 0.950),
        (0.495, 0.680),     # trans: 6.5까지 늘어난 극단값 제거 -> 단일 타원 복원
        (0.045, 0.215),
        (0.000, 0.085),
        (1.620, 2.020),
    ],
}

def main():
    npy_files = sorted([f for f in os.listdir(OUT_DIR) if f.startswith("samples_") and f.endswith("d.npy")])
    if not npy_files:
        print(f"[오류] 샘플 파일을 찾을 수 없습니다: {OUT_DIR}")
        return

    for fname in npy_files:
        days_str = fname.replace("samples_", "").replace("d.npy", "")
        days_val = round(float(days_str), 2)
        samples = np.load(os.path.join(OUT_DIR, fname))

        # 지정된 range가 있으면 적용, 없으면 1%~99% 분위수 자동 줌인
        plot_range = CUSTOM_RANGES.get(days_val, [0.985] * samples.shape[1])

        fig = corner.corner(
            samples,
            labels=CORNER_LABELS,
            range=plot_range,
            quantiles=[0.16, 0.50, 0.84],
            show_titles=True,
            title_fmt=".3f",
            title_kwargs={"fontsize": 10.5},
            label_kwargs={"fontsize": 11.5},
            smooth=1.2,
            smooth1d=1.2,
            plot_datapoints=True,
            plot_density=True,
            fill_contours=True,
            color="#1b365d"
        )

        # 원본 전체 체인 분위수를 제목에 유지하고 싶다면 위 show_titles=True가 range 내 샘플 기준이 되므로,
        # 원본 전체 샘플의 중앙값/1-sigma 오차를 각 대각 패널 제목에 정확히 덮어쓰기
        axes = np.array(fig.axes).reshape((samples.shape[1], samples.shape[1]))
        param_names_plain = [r"T^\prime\ [\mathrm{K}]", r"N_{29}", r"v_{\max}\ [c]",
                             r"v_{\mathrm{phot}}\ [c]", r"\tau", r"\mathrm{trans}",
                             r"v_e\ [c]", r"\mathrm{amp}_1", r"\mathrm{amp}_2"]
        for i in range(samples.shape[1]):
            q16, q50, q84 = np.percentile(samples[:, i], [16, 50, 84])
            up, lo = q84 - q50, q50 - q16
            axes[i, i].set_title(rf"${param_names_plain[i]} = {q50:.3f}^{{+{up:.3f}}}_{{-{lo:.3f}}}$", fontsize=10.2)

        out_path = os.path.join(OUT_DIR, f"Corner_Phase_{days_val:.2f}d_Zoomed.png")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[완료] {os.path.basename(out_path)} 생성됨")

if __name__ == "__main__":
    main()
