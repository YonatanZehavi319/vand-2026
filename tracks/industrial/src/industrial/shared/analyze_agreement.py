"""Analyze agreement between INP-Former and CPR heatmaps per category.

Usage:
    python -m industrial.shared.analyze_agreement \
        --inp_dir path/to/inp/val_heatmaps \
        --cpr_dir path/to/cpr/val_heatmaps
"""

import argparse
import os
import numpy as np
import cv2
from glob import glob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp_dir', type=str, required=True)
    parser.add_argument('--cpr_dir', type=str, required=True)
    parser.add_argument('--save_size', type=int, default=512)
    args = parser.parse_args()

    categories = sorted(os.listdir(args.inp_dir))

    print(f"{'Category':<15} {'Corr':>8} {'INP std':>10} {'CPR std':>10} {'INP max':>10} {'CPR max':>10} {'Ratio std':>10}")
    print("-" * 80)

    for category in categories:
        inp_good = os.path.join(args.inp_dir, category, 'good')
        cpr_good = os.path.join(args.cpr_dir, category, 'good')

        if not os.path.isdir(inp_good) or not os.path.isdir(cpr_good):
            print(f"  {category}: skipped (missing dir)")
            continue

        inp_files = sorted(glob(os.path.join(inp_good, '*_heatmap_raw.npy')))

        correlations = []
        inp_stds = []
        cpr_stds = []
        inp_maxes = []
        cpr_maxes = []

        for npy_path in inp_files:
            fname = os.path.basename(npy_path).replace('_heatmap_raw.npy', '')
            cpr_npy = os.path.join(cpr_good, f'{fname}_heatmap_raw.npy')

            if not os.path.exists(cpr_npy):
                continue

            inp_map = np.load(npy_path).astype(np.float32)
            cpr_map = np.load(cpr_npy).astype(np.float32)

            inp_map = cv2.resize(inp_map, (args.save_size, args.save_size)).flatten()
            cpr_map = cv2.resize(cpr_map, (args.save_size, args.save_size)).flatten()

            # Normalize each to 0-1 for fair comparison
            inp_norm = (inp_map - inp_map.min()) / (inp_map.max() - inp_map.min() + 1e-8)
            cpr_norm = (cpr_map - cpr_map.min()) / (cpr_map.max() - cpr_map.min() + 1e-8)

            corr = np.corrcoef(inp_norm, cpr_norm)[0, 1]
            correlations.append(corr)
            inp_stds.append(inp_map.std())
            cpr_stds.append(cpr_map.std())
            inp_maxes.append(inp_map.max())
            cpr_maxes.append(cpr_map.max())

        if correlations:
            mean_corr = np.mean(correlations)
            mean_inp_std = np.mean(inp_stds)
            mean_cpr_std = np.mean(cpr_stds)
            mean_inp_max = np.mean(inp_maxes)
            mean_cpr_max = np.mean(cpr_maxes)
            ratio = mean_cpr_std / (mean_inp_std + 1e-8)
            print(f"{category:<15} {mean_corr:>8.4f} {mean_inp_std:>10.6f} {mean_cpr_std:>10.6f} {mean_inp_max:>10.6f} {mean_cpr_max:>10.6f} {ratio:>10.4f}")


if __name__ == '__main__':
    main()
