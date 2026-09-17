import os
import matplotlib.pyplot as plt
import numpy as np

try:
    import corner
except ImportError:
    corner = None

from src.models import planck_with_mod_full_relativistic
from src.continuum import calc_relativistic_blackbody_continuum


def plot_spectrum_fit(wave, flux, wave_grid, model_grid, days, dl_med, tau_sr, tau_he, case_id, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(wave, flux, color='#cccccc', alpha=0.7, lw=0.8, zorder=2, label='Data')
    ax.plot(wave_grid, model_grid, color='#c2185b', lw=2.0, zorder=3,
             label=rf"+{days:.2f}d Fit: $D_L={dl_med:.1f}\mathrm{{Mpc}}$, $\tau_{{\mathrm{{Sr}}}}={tau_sr:.2f}, \tau_{{\mathrm{{He}}}}={tau_he:.2f}$")
    ax.set_xlim(3500, 22000)

    # Expand y-axis boundaries
    y_min = np.min(flux)
    y_max = np.max(flux)
    padding = (y_max - y_min) * 0.2
    ax.set_ylim(min(-0.02e-15, y_min - padding), y_max + padding)

    ax.set_xlabel('Rest Wavelength [Å]', fontsize=12)
    ax.set_ylabel(r'Flux [$erg / s / cm^2 / \AA$]', fontsize=12)
    ax.set_title(f'AT2017gfo Spectrum Fit (+{days:.2f}d) ({case_id})', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=250)
    plt.close(fig)

def plot_stacked_spectra_fit(spectra_data, results_summary, case_id, use_nlte, use_he, save_path):
    fig1, ax1 = plt.subplots(figsize=(10, 11))
    offset_step = 4.0e-16
    telluric_bands = [(5330, 5740), (9800, 10250), (13100, 14400), (17550, 19200)]
    for b_low, b_high in telluric_bands:
        ax1.axvspan(b_low, b_high, color='gray', alpha=0.15, zorder=1)

    ax1.text(13750, 1.50e-15, 'Telluric Band', rotation=90, color='gray', fontsize=8.5, ha='center', va='top')
    ax1.text(18375, 1.50e-15, 'Telluric Band', rotation=90, color='gray', fontsize=8.5, ha='center', va='top')

    for idx, sdata in enumerate(spectra_data):
        offset = (len(spectra_data) - 1 - idx) * offset_step
        res = results_summary[idx]
        popt = res["popt"]

        wave_grid = np.linspace(3800, 21500, 1000)
        model_grid = planck_with_mod_full_relativistic(
            wave_grid, popt["T_prime"], popt["N_29"], popt["vmax"], popt["vphot"],
            tau_sr=popt["tau_sr"], tau_he=popt["tau_he"], trans=popt["trans"],
             t0=sdata["days"] * 86400.0,
            use_nlte=use_nlte, use_he=use_he
        )

        ax1.plot(sdata["wave"], sdata["flux"] + offset, color='#cccccc', alpha=0.7, lw=0.8, zorder=2)
        ax1.plot(wave_grid, model_grid + offset, color='#c2185b', lw=2.0, zorder=3,
                 label=rf"{sdata['label']}: $D_L={res['dl_med']:.1f}\mathrm{{Mpc}}$, $\tau_{{\mathrm{{Sr}}}}={popt['tau_sr']:.2f}, \tau_{{\mathrm{{He}}}}={popt['tau_he']:.2f}$")
        ax1.text(4000, offset + 0.35e-16, f"+{sdata['days']:.3f}d", fontsize=11, fontweight='bold', color='black')

    ax1.set_xlim(3500, 22000)
    # Expand y-axis boundaries
    ax1.set_ylim(-0.02e-15, 1.68e-15 + offset_step * 0.5)
    ax1.set_xlabel('Rest Wavelength [Å]', fontsize=12)
    ax1.set_ylabel(r'Flux [$erg / s / cm^2 / \AA$] + Offset', fontsize=12)
    ax1.set_title(f'AT2017gfo Stacked Spectra Fit ({case_id})', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8.2, framealpha=0.9, facecolor='white')
    ax1.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=250)
    plt.close(fig1)

def plot_line_profile(wave_zoom, prof_full, prof_no_occ, obs_norm, days, case_id, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axvspan(9950, 10250, color='gray', alpha=0.18, zorder=1)
    ax.axhline(1.0, color='black', ls=':', lw=1.1, zorder=2, label='Normalized Continuum (1.0)')
    ax.axvline(10327.311, color='slateblue', ls='-.', lw=1.2, zorder=2, label=r'Rest $\mathrm{Sr\ II}\ (1.0327\mu\mathrm{m})$')
    ax.axvline(10833.3, color='forestgreen', ls='-.', lw=1.2, zorder=2, label=r'Rest $\mathrm{He\ I}\ (1.0833\mu\mathrm{m})$')

    ax.plot(wave_zoom, obs_norm, color='#cccccc', alpha=0.8, lw=1.0, label='Normalized Observed Data')
    ax.plot(wave_zoom, prof_full, color='#1f77b4', ls='-', lw=2.2, zorder=4, label='Occulted (Full Fit)')
    ax.plot(wave_zoom, prof_no_occ, color='#ff7f0e', ls='--', lw=1.3, zorder=3, label='No-Occultation')

    ax.set_xlim(7000, 12500)
    ax.set_xlabel('Rest Wavelength [Å]', fontsize=12)
    ax.set_ylabel(r'Normalized Flux ($F_\lambda / F_{\mathrm{cont}}$)', fontsize=12)
    ax.set_title(f'Line Profile (+{days:.2f}d) ({case_id})', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=250)
    plt.close(fig)

def plot_stacked_line_profiles(spectra_data, results_summary, case_id, use_nlte, use_he, save_path):
    fig2, ax2 = plt.subplots(figsize=(11, 7.8))
    wave_zoom = np.linspace(7000, 12500, 500)
    ax2.axvspan(9650, 10300, color='gray', alpha=0.18, zorder=1)

    ax2.axhline(1.0, color='black', ls=':', lw=1.1, zorder=2, label='Normalized Continuum (1.0)')
    ax2.axvline(10327.311, color='slateblue', ls='-.', lw=1.2, zorder=2, label=r'Rest $\mathrm{Sr\ II}\ (1.0327\mu\mathrm{m})$')
    ax2.axvline(10833.3, color='forestgreen', ls='-.', lw=1.2, zorder=2, label=r'Rest $\mathrm{He\ I}\ (1.0833\mu\mathrm{m})$')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    global_max_flux = 1.0
    global_min_flux = 1.0

    for idx, sdata in enumerate(spectra_data):
        res = results_summary[idx]
        popt = res["popt"]
        t_ph = sdata["days"] * 86400.0
        c = colors[idx % len(colors)]

        model_full = planck_with_mod_full_relativistic(
            wave_zoom, popt["T_prime"], popt["N_29"], popt["vmax"], popt["vphot"],
            tau_sr=popt["tau_sr"], tau_he=popt["tau_he"], trans=popt["trans"],
             t0=t_ph, use_nlte=use_nlte, use_he=use_he
        )
        model_no_occ = planck_with_mod_full_relativistic(
            wave_zoom, popt["T_prime"], popt["N_29"], popt["vmax"], popt["vphot"],
            tau_sr=popt["tau_sr"], tau_he=popt["tau_he"], trans=1.0, # trans=1.0 is no-occultation equivalent loosely for illustration
             t0=t_ph, use_nlte=use_nlte, use_he=use_he
        )

        cont_zoom = (popt["N_29"] * 1e-29) * calc_relativistic_blackbody_continuum(wave_zoom, popt["T_prime"], popt["vphot"])
        prof_full = model_full / cont_zoom
        prof_no_occ = model_no_occ / cont_zoom

        global_max_flux = max(global_max_flux, np.max(prof_full))
        global_min_flux = min(global_min_flux, np.min(prof_full))

        ax2.plot(wave_zoom, prof_full, color=c, ls='-', lw=2.2, zorder=4,
                 label=rf"+{sdata['days']:.2f}d (Full Fit: $\tau_{{\mathrm{{Sr}}}}={popt['tau_sr']:.2f}, \tau_{{\mathrm{{He}}}}={popt['tau_he']:.2f}$)")
        ax2.plot(wave_zoom, prof_no_occ, color=c, ls='--', lw=1.3, alpha=0.75, zorder=3,
                 label=rf"+{sdata['days']:.2f}d (No-Occultation)")

    y_upper_limit = max(2.10, global_max_flux * 1.10)
    y_lower_limit = max(0.40, global_min_flux - 0.10)
    ax2.set_xlim(7000, 12500)
    ax2.set_ylim(y_lower_limit, y_upper_limit)
    ax2.text(9975, y_upper_limit - 0.08, 'Masked', ha='center', va='center', color='gray', fontsize=10, fontweight='bold')

    ax2.set_xlabel('Rest Wavelength [Å]', fontsize=12)
    ax2.set_ylabel(r'Normalized Flux ($F_\lambda / F_{\mathrm{cont}}$)', fontsize=12)
    ax2.set_title(f'Normalized Line Profile Evolution ({case_id})', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=7.2, ncol=2, frameon=True, facecolor='white', framealpha=0.95, borderpad=0.3, labelspacing=0.25, handlelength=1.5)
    ax2.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=250)
    plt.close(fig2)

def plot_optical_depth_evolution(results_summary, use_he, case_id, save_path):
    days_arr = [r["days"] for r in results_summary]
    tau_sr_arr = [r["popt"]["tau_sr"] for r in results_summary]
    tau_he_arr = [r["popt"]["tau_he"] for r in results_summary]

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.plot(days_arr, tau_sr_arr, 'o-', color='royalblue', ms=8, lw=2.2, label=r'$\mathrm{Sr\ II}$ Optical Depth ($\tau_{\mathrm{Sr}}$)')
    if use_he:
        ax3.plot(days_arr, tau_he_arr, 's--', color='darkgreen', ms=8, lw=2.2, label=r'$\mathrm{He\ I}$ Optical Depth ($\tau_{\mathrm{He}}$)')

    for d, ts, th in zip(days_arr, tau_sr_arr, tau_he_arr):
        ax3.annotate(f"{ts:.2f}", (d, ts), textcoords="offset points", xytext=(0, 8), ha='center', fontweight='bold', color='royalblue')
        if use_he:
            ax3.annotate(f"{th:.2f}", (d, th), textcoords="offset points", xytext=(0, -15), ha='center', fontweight='bold', color='darkgreen')

    ax3.set_xlabel('Phase [Days post-merger]', fontsize=12)
    ax3.set_ylabel(r'Optical Depth ($\tau$)', fontsize=12)
    ax3.set_title(f'Temporal Evolution of Optical Depth ({case_id})', fontsize=13, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=250)
    plt.close(fig3)

def generate_all_plots(spectra_data, results_summary, case_id, use_nlte, use_he, flat_samples_dict, labels_dict, target_save_dir):
    # Plot 1: Stacked spectra fit (1 plot)
    plot_stacked_spectra_fit(spectra_data, results_summary, case_id, use_nlte, use_he, os.path.join(target_save_dir, "Plot1_Stacked_Spectra_Fit.png"))

    # Plot 2: Stacked line profiles (1 plot)
    plot_stacked_line_profiles(spectra_data, results_summary, case_id, use_nlte, use_he, os.path.join(target_save_dir, "Plot2_Line_Profile_Evolution.png"))

    # Plot 3: Optical depth evolution (1 plot)
    plot_optical_depth_evolution(results_summary, use_he, case_id, os.path.join(target_save_dir, "Plot3_Optical_Depth_Evolution.png"))

    for idx, sdata in enumerate(spectra_data):
        res = results_summary[idx]
        popt = res["popt"]
        days = sdata["days"]
        t_ph = days * 86400.0

        wave_grid = np.linspace(3800, 21500, 1000)
        model_grid = planck_with_mod_full_relativistic(
            wave_grid, popt["T_prime"], popt["N_29"], popt["vmax"], popt["vphot"],
            tau_sr=popt["tau_sr"], tau_he=popt["tau_he"], trans=popt["trans"],
             t0=t_ph,
            use_nlte=use_nlte, use_he=use_he
        )

        # Plot 1 singles: 4 individual plots
        plot_spectrum_fit(sdata["wave"], sdata["flux"], wave_grid, model_grid, days, res["dl_med"], popt["tau_sr"], popt["tau_he"], case_id, os.path.join(target_save_dir, f"Plot1_Spectrum_Fit_{days:.3f}d.png"))

        # Plot 2 singles: 4 individual plots
        wave_zoom = np.linspace(7000, 12500, 500)
        model_full = planck_with_mod_full_relativistic(
            wave_zoom, popt["T_prime"], popt["N_29"], popt["vmax"], popt["vphot"],
            tau_sr=popt["tau_sr"], tau_he=popt["tau_he"], trans=popt["trans"],
             t0=t_ph, use_nlte=use_nlte, use_he=use_he
        )
        model_no_occ = planck_with_mod_full_relativistic(
            wave_zoom, popt["T_prime"], popt["N_29"], popt["vmax"], popt["vphot"],
            tau_sr=popt["tau_sr"], tau_he=popt["tau_he"], trans=1.0,
             t0=t_ph, use_nlte=use_nlte, use_he=use_he
        )
        cont_zoom = (popt["N_29"] * 1e-29) * calc_relativistic_blackbody_continuum(wave_zoom, popt["T_prime"], popt["vphot"])

        prof_full = model_full / cont_zoom
        prof_no_occ = model_no_occ / cont_zoom

        # interpolate obs data for normalization
        obs_interp = np.interp(wave_zoom, sdata["wave"], sdata["flux"], left=np.nan, right=np.nan)
        obs_norm = obs_interp / cont_zoom

        plot_line_profile(wave_zoom, prof_full, prof_no_occ, obs_norm, days, case_id, os.path.join(target_save_dir, f"Plot2_Line_Profile_{days:.3f}d.png"))

        # Corner plots (4 plots)
        if corner is not None:
            flat_samples = flat_samples_dict[days]
            corner_labels = labels_dict["corner_labels"]

            safe_ranges = []
            for col_idx in range(flat_samples.shape[1]):
                col_data = flat_samples[:, col_idx]
                ptp_val = np.ptp(col_data)
                med_val = np.median(col_data)
                if ptp_val < 1e-5:
                    span_pad = max(abs(med_val) * 0.05, 0.01)
                    safe_ranges.append((med_val - span_pad, med_val + span_pad))
                else:
                    safe_ranges.append(0.999)

            fig_corner = corner.corner(
                flat_samples,
                labels=corner_labels,
                range=safe_ranges,
                quantiles=[0.16, 0.50, 0.84],
                show_titles=True,
                title_fmt='.3f',
                smooth=1.0,
                levels=(0.68, 0.95),
                fill_contours=True,
                plot_datapoints=True
            )
            label_fn = sdata['label'].replace(" ", "_").replace("+", "").replace("(", "").replace(")", "")
            fig_corner.savefig(os.path.join(target_save_dir, f"Corner_{label_fn}.png"), dpi=200)
            plt.close(fig_corner)
