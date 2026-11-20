#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
config/settings.py
물리 상수(SI/CGS), 전이선 파장 및 AT2017gfo 10개 에포크(+1.43d ~ +10.40d) 설정
- 경계 충돌(Boundary Pegging) 방지를 위해 물리적으로 허용되는 넉넉한 Flat Prior 범위 부여
"""

from scipy import constants

H_PLANCK_SI = constants.h
C_SPEED_SI = constants.c
K_BOLTZ_SI = constants.k

C_CGS = constants.c * 100.0
H_PLANCK = constants.h * 1e7
K_BOLTZ = constants.k * 1e7
MPC_TO_CM = 3.085677581491367e24

# Sr II 근적외선 삼중항 및 비열적 NLTE He I 정지 파장 [Å]
LAM_SR_10036_AA = 10036.65
LAM_SR_10327_AA = 10327.311
LAM_SR_10914_AA = 10914.887
LAM_HE_10833_AA = 10833.31

# 근적외선(NIR) 란타넘족 방출 특징 중심 및 폭 [Å]
CEN1_AA, SIG1_AA = 15500.0, 580.0
CEN2_AA, SIG2_AA = 20200.0, 800.0

# 지구 대기 텔루릭 흡수대 및 기기 경계 아티팩트 구간 [Å]
TELLURIC_BANDS_AA = [
    (13100.0, 14400.0),
    (17550.0, 19200.0),
]
ARTIFACT_BANDS_AA = [
    (5330.0, 5740.0),
]

# 모은하 NGC 4993 우주론적 적색편이
Z_COSMIC_MEAN = 0.00986
Z_COSMIC_ERR = 0.00049
Q0_DECEL = -0.53

# MCMC 하이퍼파라미터 (96 Walkers, 20000 Steps, 10000 Burn-in)
N_WALKERS = 128
N_STEPS = 20000
BURN_IN = 10000
THIN_STEP = 10

PARAM_NAMES = ["T_prime", "N_29", "vmax", "vphot", "tau", "trans", "ve", "amp1", "amp2"]
LATEX_LABELS = [
    r"$T'\ [\mathrm{K}]$", r"$N_{29}$", r"$v_{\max}\ [c]$", r"$v_{\mathrm{phot}}\ [c]$",
    r"$\tau$", r"$\mathrm{trans}$", r"$v_e\ [c]$", r"$\mathrm{amp}_1$", r"$\mathrm{amp}_2$"
]

# 파라미터 순서: [T_prime, N_29, vmax, vphot, tau, trans, ve, amp1, amp2]
PHASES_CONFIG = [
    {
        "label": "Phase +1.43d (OB1)",
        "days": 1.427,
        "filename": "OB1_1.43d.dat",
        "is_photospheric": True,
        "url": "https://sid.erda.dk/share_redirect/df1fMhon6Z/dereddened%2Bderedshifted_spectra/AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57983.969_Phase%2B1.43d_deredz.dat",
        "init_guess": [4560.0, 0.515, 0.385, 0.198, 1.25, 0.002, 0.115, 0.100, 0.360],
        "bounds": [(3900.0, 5300.0), (0.25, 0.90), (0.28, 0.48), (0.12, 0.27), (0.20, 3.50), (0.00, 0.25), (0.02, 0.25), (0.00, 0.50), (0.00, 0.80)],
    },
    {
        "label": "Phase +2.42d (OB2)",
        "days": 2.417,
        "filename": "OB2_2.42d.dat",
        "is_photospheric": True,
        "url": "https://sid.erda.dk/share_redirect/df1fMhon6Z/dereddened%2Bderedshifted_spectra/AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57984.969_Phase%2B2.42d_deredz.dat",
        "init_guess": [3410.0, 1.105, 0.335, 0.125, 1.20, 0.350, 0.120, 0.120, 0.220],
        "bounds": [(2900.0, 4000.0), (0.60, 1.80), (0.22, 0.44), (0.07, 0.22), (0.20, 3.50), (0.00, 1.80), (0.02, 0.25), (0.00, 0.60), (0.00, 0.80)],
    },
    {
        "label": "Phase +3.41d (OB3)",
        "days": 3.413,
        "filename": "OB3_3.41d.dat",
        "is_photospheric": True,
        "url": "https://sid.erda.dk/share_redirect/df1fMhon6Z/dereddened%2Bderedshifted_spectra/AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57985.974_Phase%2B3.41d_deredz.dat",
        "init_guess": [2890.0, 1.460, 0.315, 0.152, 1.60, 2.150, 0.065, 0.240, 0.450],
        "bounds": [(2450.0, 3450.0), (0.80, 2.30), (0.20, 0.42), (0.08, 0.23), (0.20, 4.00), (0.20, 6.00), (0.015, 0.22), (0.00, 0.80), (0.00, 1.00)],
    },
    {
        "label": "Phase +4.40d (OB4)",
        "days": 4.403,
        "filename": "OB4_4.40d.dat",
        "is_photospheric": True,
        "url": "https://sid.erda.dk/share_redirect/df1fMhon6Z/dereddened%2Bderedshifted_spectra/AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57986.974_Phase%2B4.40d_deredz.dat",
        "init_guess": [2610.0, 1.600, 0.265, 0.178, 0.55, 6.500, 0.090, 0.320, 0.480],
        "bounds": [(2200.0, 3200.0), (0.80, 2.60), (0.18, 0.38), (0.09, 0.24), (0.10, 2.50), (1.00, 16.0), (0.015, 0.22), (0.00, 1.00), (0.00, 1.20)],
    },
    {
        "label": "Phase +5.40d (OB5)",
        "days": 5.400,
        "filename": "OB5_5.40d.dat",
        "is_photospheric": False,
        "url": "https://sid.erda.dk/share_redirect/df1fMhon6Z/dereddened%2Bderedshifted_spectra/AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57987.980_Phase%2B5.40d_deredz.dat",
        "init_guess": [2650.0, 1.150, 0.260, 0.165, 0.42, 7.200, 0.090, 0.420, 0.520],
        "bounds": [(2000.0, 3300.0), (0.20, 2.50), (0.16, 0.36), (0.08, 0.23), (0.05, 2.00), (1.00, 20.0), (0.015, 0.22), (0.00, 1.50), (0.00, 1.60)],
    },
    {
        "label": "Phase +6.40d (OB6)",
        "days": 6.400,
        "filename": "OB6_6.40d.dat",
        "is_photospheric": False,
        "url": "https://sid.erda.dk/share_redirect/df1fMhon6Z/dereddened%2Bderedshifted_spectra/AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57988.990_Phase%2B6.40d_deredz.dat",
        "init_guess": [2680.0, 0.780, 0.250, 0.155, 0.35, 7.800, 0.085, 0.500, 0.650],
        "bounds": [(1900.0, 3300.0), (0.10, 2.20), (0.15, 0.35), (0.07, 0.22), (0.03, 1.80), (1.00, 22.0), (0.015, 0.22), (0.00, 1.80), (0.00, 2.00)],
    },
    {
        "label": "Phase +7.40d (OB7)",
        "days": 7.400,
        "filename": "OB7_7.40d.dat",
        "is_photospheric": False,
        "url": "https://sid.erda.dk/share_redirect/df1fMhon6Z/dereddened%2Bderedshifted_spectra/AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57990.000_Phase%2B7.40d_deredz.dat",
        "init_guess": [2700.0, 0.520, 0.240, 0.145, 0.28, 8.200, 0.080, 0.580, 0.780],
        "bounds": [(1800.0, 3300.0), (0.05, 1.80), (0.14, 0.34), (0.06, 0.21), (0.02, 1.50), (0.50, 25.0), (0.015, 0.22), (0.00, 2.20), (0.00, 2.50)],
    },
    {
        "label": "Phase +8.40d (OB8)",
        "days": 8.400,
        "filename": "OB8_8.40d.dat",
        "is_photospheric": False,
        "url": "https://sid.erda.dk/share_redirect/df1fMhon6Z/dereddened%2Bderedshifted_spectra/AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57991.000_Phase%2B8.40d_deredz.dat",
        "init_guess": [2720.0, 0.350, 0.230, 0.135, 0.22, 8.500, 0.080, 0.650, 0.900],
        "bounds": [(1700.0, 3300.0), (0.02, 1.50), (0.13, 0.33), (0.05, 0.20), (0.01, 1.20), (0.50, 28.0), (0.015, 0.22), (0.00, 2.60), (0.00, 3.00)],
    },
    {
        "label": "Phase +9.40d (OB9)",
        "days": 9.400,
        "filename": "OB9_9.40d.dat",
        "is_photospheric": False,
        "url": "https://sid.erda.dk/share_redirect/df1fMhon6Z/dereddened%2Bderedshifted_spectra/AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57992.000_Phase%2B9.40d_deredz.dat",
        "init_guess": [2750.0, 0.220, 0.220, 0.125, 0.18, 8.800, 0.075, 0.720, 1.050],
        "bounds": [(1600.0, 3300.0), (0.01, 1.20), (0.12, 0.32), (0.05, 0.19), (0.01, 1.00), (0.50, 30.0), (0.015, 0.22), (0.00, 3.00), (0.00, 3.50)],
    },
    {
        "label": "Phase +10.40d (OB10)",
        "days": 10.400,
        "filename": "OB10_10.40d.dat",
        "is_photospheric": False,
        "url": "https://sid.erda.dk/share_redirect/df1fMhon6Z/dereddened%2Bderedshifted_spectra/AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57993.000_Phase%2B10.40d_deredz.dat",
        "init_guess": [2750.0, 0.140, 0.210, 0.118, 0.15, 9.000, 0.075, 0.800, 1.200],
        "bounds": [(1500.0, 3300.0), (0.005, 1.00), (0.11, 0.31), (0.04, 0.18), (0.005, 0.90), (0.50, 32.0), (0.015, 0.22), (0.00, 3.50), (0.00, 4.00)],
    },
]
