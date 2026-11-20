#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
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

samples_all = np.load(os.path.join(OUT_DIR, "samples_1.43d.npy"))

# T' (~4520K) 및 v_phot (~0.200c) 잔여 부 모드 꼬리를 제거한 주 모드 마스크
primary_mask = (
    (samples_all[:, 0] >= 4545.0) &
    (samples_all[:, 2] <= 0.435) &
    (samples_all[:, 3] >= 0.205) &
    (samples_all[:, 4] <= 0.500) &
    (samples_all[:, 6] >= 0.115)
)
samples_primary = samples_all[primary_mask].copy()

plot_range = [
    (4530.0, 4695.0),    # T': ~4520K 잔여 꼬리 배제 및 우측 여백 확보
    (0.456, 0.494),      # N29
    (0.386, 0.428),      # vmax
    (0.203, 0.232),      # vphot: ~0.200c 잔여 꼬리 배제
    (0.315, 0.475),      # tau
    (-0.025, 0.160),     # trans: 0.00 좌측 기준선 여백 확보
    (0.120, 0.220),      # ve
    (0.095, 0.205),      # amp1
    (0.355, 0.530),      # amp2
]

fig = corner.corner(
    samples_primary,
    labels=CORNER_LABELS,
    range=plot_range,
    bins=28,
    quantiles=[0.16, 0.50, 0.84],
    show_titles=True,
    title_fmt=".3f",
    title_kwargs={"fontsize": 10.5},
    label_kwargs={"fontsize": 11.5},
    smooth=1.4,
    smooth1d=1.4,
    plot_datapoints=True,
    plot_density=True,
    fill_contours=True,
    color="#1b365d"
)

# 전체 체인 기준 원래 중앙값 및 1-sigma 오차를 대각 패널 제목에 그대로 표기
axes = np.array(fig.axes).reshape((samples_all.shape[1], samples_all.shape[1]))
param_names_plain = [r"T^\prime\ [\mathrm{K}]", r"N_{29}", r"v_{\max}\ [c]",
                     r"v_{\mathrm{phot}}\ [c]", r"\tau", r"\mathrm{trans}",
                     r"v_e\ [c]", r"\mathrm{amp}_1", r"\mathrm{amp}_2"]
for i in range(samples_all.shape[1]):
    q16, q50, q84 = np.percentile(samples_all[:, i], [16, 50, 84])
    up, lo = q84 - q50, q50 - q16
    axes[i, i].set_title(rf"${param_names_plain[i]} = {q50:.3f}^{{+{up:.3f}}}_{{-{lo:.3f}}}$", fontsize=10.2)

out_path = os.path.join(OUT_DIR, "Corner_Phase_1.43d_Zoomed.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[완료] {out_path} 재생성 완료 (주 모드 샘플 비율: {len(samples_primary)/len(samples_all)*100:.1f}%)")
