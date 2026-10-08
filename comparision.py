#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------------------
# 1. 데이터 정의 (연구명, H0 값, 1-sigma 오차, 범주/기법)
# -------------------------------------------------------------
data = [
    # (라벨, H0, err_lower, err_upper, 구분 카테고리)
    (
        "This Work (Combined 4-Epoch)",
        74.7,
        4.6,
        4.6,
        "This Work",
    ),
    (
        "This Work (+1.43d Single)",
        71.8,
        4.0,
        4.0,
        "This Work",
    ),
    (
        "Sneppen et al. (2023, A&A)",
        67.0,
        3.6,
        3.6,
        "Kilonova EPM",
    ),
    (
        "Sneppen et al. (2023, Nature)",
        65.6,
        3.4,
        3.4,
        "Kilonova EPM",
    ),
    (
        "Hotokezaka et al. (2019, GW+Radio)",
        70.3,
        5.0,
        5.3,
        "GW + Jet",
    ),
    (
        "Mooley et al. (2018, GW+VLBI)",
        71.5,
        4.6,
        4.6,
        "GW + Jet",
    ),
    (
        "Abbott et al. (2017, GW170817 standard siren)",
        70.0,
        8.0,
        12.0,
        "Pure GW",
    ),
    (
        "SH0ES (Riess et al. 2022, Cepheid-SN Ia)",
        73.04,
        1.04,
        1.04,
        "Local Distance Ladder",
    ),
    (
        "CCHP (Freedman et al. 2021, TRGB-SN Ia)",
        69.8,
        1.7,
        1.7,
        "Local Distance Ladder",
    ),
    (
        "Planck Collaboration (2018, CMB LCDM)",
        67.4,
        0.5,
        0.5,
        "Early Universe (CMB)",
    ),
]

labels = [d[0] for d in data]
h0_vals = np.array([d[1] for d in data])
err_low = np.array([d[2] for d in data])
err_high = np.array([d[3] for d in data])
categories = [d[4] for d in data]

# 범주별 색상 및 마커 스타일 지정
color_map = {
    "This Work": ("#D81B60", "D"),  # 강조 진분홍 / 다이아몬드
    "Kilonova EPM": ("#1E88E5", "o"),  # 파랑 / 원
    "GW + Jet": ("#004D40", "s"),  # 딥 틸 / 사각형
    "Pure GW": ("#757575", "^"),  # 회색 / 삼각형
    "Local Distance Ladder": ("#FFC107", "v"),  # 주황-노랑 / 역삼각형
    "Early Universe (CMB)": ("#5E35B1", "P"),  # 보라 / 플러스
}

# -------------------------------------------------------------
# 2. 플롯 설정 및 렌더링
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
y_indices = np.arange(len(data))[::-1]  # 위에서 아래로 순차 배치

# (1) 배경 참조 밴드: Planck(초기 우주) vs SH0ES(국소 우주) 허블 장력 영역 음영 표시
ax.axvspan(
    67.4 - 0.5,
    67.4 + 0.5,
    color="#5E35B1",
    alpha=0.15,
    label="Planck 2018 (68% CL)",
)
ax.axvspan(
    73.04 - 1.04,
    73.04 + 1.04,
    color="#FFC107",
    alpha=0.18,
    label="SH0ES 2022 (68% CL)",
)

# (2) 기준선: 본 연구 통합 측정치 (수직 점선)
ax.axvline(
    74.7,
    color="#D81B60",
    ls="--",
    lw=1.2,
    alpha=0.7,
    label=r"This Work Best-fit ($74.7\ \mathrm{km\ s^{-1}\ Mpc^{-1}}$)",
)

# (3) 개별 측정치 에러바 플롯
plotted_categories = set()
for idx, y_pos in enumerate(y_indices):
    cat = categories[idx]
    c, m = color_map[cat]
    is_this_work = cat == "This Work"

    # 범례 중복 방지 레이블 설정
    lbl = cat if cat not in plotted_categories else None
    plotted_categories.add(cat)

    ax.errorbar(
        h0_vals[idx],
        y_pos,
        xerr=[[err_low[idx]], [err_high[idx]]],
        fmt=m,
        color=c,
        ecolor=c,
        elinewidth=2.2 if is_this_work else 1.5,
        capsize=4.5,
        capthick=1.5,
        markersize=9 if is_this_work else 7,
        zorder=5 if is_this_work else 3,
    )

    # 오차 범위 수치 표기 (플롯 우측 정렬 텍스트)
    if err_low[idx] == err_high[idx]:
        val_str = rf"${h0_vals[idx]:.1f} \pm {err_low[idx]:.1f}$"
    else:
        val_str = (
            rf"${h0_vals[idx]:.1f}_{{-{err_low[idx]:.1f}}}^{{+{err_high[idx]:.1f}}}$"
        )

    ax.text(
        h0_vals[idx] + err_high[idx] + 1.0,
        y_pos,
        val_str,
        va="center",
        ha="left",
        fontsize=9,
        fontweight="bold" if is_this_work else "normal",
        color=c if is_this_work else "#333333",
    )

# -------------------------------------------------------------
# 3. 축 서식 및 레이아웃 정리
# -------------------------------------------------------------
ax.set_yticks(y_indices)
ax.set_yticklabels(labels, fontsize=10.5)

# 본 연구 라벨 폰트 굵게 강조
# 수정 코드
for y_tick in ax.get_yticklabels():
    if "This Work" in y_tick.get_text():
        y_tick.set_fontweight("bold")
        y_tick.set_color("#D81B60")

ax.set_xlim(55, 95)
ax.set_xlabel(
    r"$H_0\ [\mathrm{km\ s^{-1}\ Mpc^{-1}}]$", fontsize=13, fontweight="bold"
)
ax.set_title(
    "Hubble Constant Comparison: Kilonova AT2017gfo & Cosmological Probes",
    fontsize=14,
    fontweight="bold",
    pad=15,
)

# 눈금 및 그리드 설정
ax.xaxis.grid(True, linestyle=":", alpha=0.4)
ax.set_axisbelow(True)

# 범례
handles, leg_labels = ax.get_legend_handles_labels()
ax.legend(
    handles=handles,
    labels=leg_labels,
    loc="lower right",
    fontsize=9.5,
    frameon=True,
    framealpha=0.9,
)

plt.tight_layout()

# -------------------------------------------------------------
# 4. 저장 및 출력
# -------------------------------------------------------------
plt.savefig("H0_Comparison_Plot.png", dpi=300, bbox_inches="tight")
print(
    "[완료] 'H0_Comparison_Plot.png' 고해상도 플롯이 현재 디렉토리에 저장되었습니다."
)
plt.show()