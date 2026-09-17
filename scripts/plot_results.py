import os
import json
import numpy as np
import argparse
import multiprocessing as mp
from config.settings import fit_cases, phases_template
from utils.plotting import generate_all_plots

def plot_results(data_dir=None):
    if data_dir is None:
        target_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output_results")
    else:
        target_base_dir = os.path.abspath(data_dir)

    if not os.path.exists(target_base_dir):
        print(f"Directory {target_base_dir} does not exist. Run MCMC fit first!")
        return

    for case_cfg in fit_cases:
        case_id = case_cfg["case_id"]
        use_nlte = case_cfg["use_nlte"]
        use_he = case_cfg["use_he"]
        target_save_dir = os.path.join(target_base_dir, case_id)
        if not os.path.exists(os.path.join(target_save_dir, 'fit_summary_all.json')):
            continue

        with open(os.path.join(target_save_dir, 'fit_summary_all.json'), 'r') as f:
            epoch_summaries = json.load(f)

        with open(os.path.join(target_save_dir, 'spectra_data.json'), 'r') as f:
            spectra_data_raw = json.load(f)
            spectra_data = []
            for sd in spectra_data_raw:
                spectra_data.append({
                    'days': sd['days'],
                    'label': sd['label'],
                    'wave': np.array(sd['wave']),
                    'flux': np.array(sd['flux']),
                    'x_fit': np.array(sd['x_fit']) if 'x_fit' in sd else None,
                    'model_fit': np.array(sd['model_fit']) if 'model_fit' in sd else None
                })

        with open(os.path.join(target_save_dir, 'results_summary.json'), 'r') as f:
            results_summary = json.load(f)

        with open(os.path.join(target_save_dir, 'labels_dict.json'), 'r') as f:
            labels_dict = json.load(f)

        flat_samples_dict = {}
        for es in epoch_summaries:
            days = es["days"]
            samples_path = os.path.join(target_save_dir, f"samples_{days:.3f}d.npy")
            if os.path.exists(samples_path):
                flat_samples_dict[days] = np.load(samples_path)

        print(f"--> [{case_id}] Generating plots...")
        generate_all_plots(spectra_data, results_summary, case_id, use_nlte, use_he, flat_samples_dict, labels_dict, target_save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot MCMC results from directory.')
    parser.add_argument('--data_dir', type=str, default=None, help='Data directory containing MCMC outputs')
    args = parser.parse_args()
    mp.freeze_support()
    plot_results(data_dir=args.data_dir)
