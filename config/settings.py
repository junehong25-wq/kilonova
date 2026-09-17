import astropy.constants as csts
from scipy import constants

h_planck = constants.h
c_speed = constants.c
k_B = constants.k
C_CGS = csts.c.cgs.value

# Sneppen et al. (2023) 근적외선 가우시안 보정 성분
CEN1_AA, SIG1_AA = 15500.0, 580.0
CEN2_AA, SIG2_AA = 20200.0, 800.0

# Chiba et al. (2026) / Arya et al. (2026) 원소별 중심 파장 (Rest Wavelength)
LAM_SR_10036_AA = 10036.65
LAM_SR_10327_AA = 10327.311
LAM_SR_10914_AA = 10914.887
LAM_HE_10833_AA = 10833.3  # He I 2^3S - 2^3P

fit_cases = [
    {"case_id": "Case1_noLTT_PureLTE", "use_nlte": False, "use_he": False},
    {"case_id": "Case2_withLTT_PureLTE", "use_nlte": False, "use_he": True},
    {"case_id": "Case3_noLTT_NLTE_He", "use_nlte": True, "use_he": False},
    {"case_id": "Case4_withLTT_NLTE_He", "use_nlte": True, "use_he": True},
]

phases_template = [
    {
        "label": "Phase +1.43d (OB1)", "days": 1.427,
        "url": "https://sid.erda.dk/share_redirect/df1fMhon6Z/dereddened%2Bderedshifted_spectra/AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57983.969_Phase%2B1.43d_deredz.dat",
        "bounds_withHe": [(4200.0, 6000.0), (0.80, 1.80), (0.35, 0.50), (0.24, 0.35), (1.0, 3.0), (0.0, 0.30), (0.01, 0.30)],
        "bounds_noHe": [(4200.0, 6000.0), (0.80, 1.80), (0.35, 0.50), (0.24, 0.35), (1.0, 3.0), (0.01, 0.30)]
    },
    {
        "label": "Phase +2.42d (OB2)", "days": 2.417,
        "url": "https://sid.erda.dk/share_redirect/df1fMhon6Z/dereddened%2Bderedshifted_spectra/AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57984.969_Phase%2B2.42d_deredz.dat",
        "bounds_withHe": [(3100.0, 3800.0), (1.50, 3.20), (0.25, 0.42), (0.130, 0.250), (0.7, 2.5), (0.05, 0.60), (0.35, 0.65)],
        "bounds_noHe": [(3100.0, 3800.0), (1.50, 3.20), (0.25, 0.42), (0.130, 0.250), (0.7, 2.5), (0.35, 0.65)]
    },
    {
        "label": "Phase +3.41d (OB3)", "days": 3.413,
        "url": "https://sid.erda.dk/share_redirect/df1fMhon6Z/dereddened%2Bderedshifted_spectra/AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57985.974_Phase%2B3.41d_deredz.dat",
        "bounds_withHe": [(2700.0, 3300.0), (1.0, 5.0), (0.18, 0.37), (0.100, 0.220), (0.4, 2.0), (0.2, 1.5), (0.35, 0.65)],
        "bounds_noHe": [(2700.0, 3300.0), (1.0, 5.0), (0.18, 0.37), (0.100, 0.220), (0.4, 2.0), (0.35, 0.65)]
    },
    {
        "label": "Phase +4.40d (OB4)", "days": 4.403,
        "url": "https://sid.erda.dk/share_redirect/df1fMhon6Z/dereddened%2Bderedshifted_spectra/AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57986.974_Phase%2B4.40d_deredz.dat",
        "bounds_withHe": [(2300.0, 2900.0), (2.00, 4.50), (0.15, 0.30), (0.070, 0.160), (0.2, 1.5), (0.5, 2.5), (0.35, 0.65)],
        "bounds_noHe": [(2300.0, 2900.0), (2.00, 4.50), (0.15, 0.30), (0.070, 0.160), (0.2, 1.5), (0.35, 0.65)]
    }
]