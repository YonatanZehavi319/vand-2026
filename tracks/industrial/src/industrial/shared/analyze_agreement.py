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
    parser.add_argument('--top_pct', type=float, default=5.0, help='Top percentile for overlap analysis')
    args = parser.parse_args()

    categories = sorted(os.listdir(args.inp_dir))
    top_pct = args.top_pct

    # Collect all stats, print table at the end
    results = []

    for category in categories:
        inp_good = os.path.join(args.inp_dir, category, 'good')
        cpr_good = os.path.join(args.cpr_dir, category, 'good')

        if not os.path.isdir(inp_good) or not os.path.isdir(cpr_good):
            continue

        inp_files = sorted(glob(os.path.join(inp_good, '*_heatmap_raw.npy')))

        correlations = []
        inp_stds = []
        cpr_stds = []
        inp_snrs = []
        cpr_snrs = []
        top_overlaps = []
        inp_entropies = []
        cpr_entropies = []
        peak_corrs = []

        for npy_path in inp_files:
            fname = os.path.basename(npy_path).replace('_heatmap_raw.npy', '')
            cpr_npy = os.path.join(cpr_good, f'{fname}_heatmap_raw.npy')

            if not os.path.exists(cpr_npy):
                continue

            inp_map = np.load(npy_path).astype(np.float32)
            cpr_map = np.load(cpr_npy).astype(np.float32)

            inp_map = cv2.resize(inp_map, (args.save_size, args.save_size)).flatten()
            cpr_map = cv2.resize(cpr_map, (args.save_size, args.save_size)).flatten()

            # Normalize to 0-1
            inp_norm = (inp_map - inp_map.min()) / (inp_map.max() - inp_map.min() + 1e-8)
            cpr_norm = (cpr_map - cpr_map.min()) / (cpr_map.max() - cpr_map.min() + 1e-8)

            # Full correlation
            corr = np.corrcoef(inp_norm, cpr_norm)[0, 1]
            correlations.append(corr)

            # Std
            inp_stds.append(inp_map.std())
            cpr_stds.append(cpr_map.std())

            # SNR: max / mean (how peaky)
            inp_snrs.append(inp_map.max() / (inp_map.mean() + 1e-8))
            cpr_snrs.append(cpr_map.max() / (cpr_map.mean() + 1e-8))

            # Top-k overlap: what fraction of INP's top pixels are also in CPR's top pixels
            n_top = max(1, int(len(inp_norm) * top_pct / 100))
            inp_top = set(np.argsort(inp_norm)[-n_top:])
            cpr_top = set(np.argsort(cpr_norm)[-n_top:])
            overlap = len(inp_top & cpr_top) / n_top
            top_overlaps.append(overlap)

            # Peak correlation: correlation only among top pixels
            all_top = sorted(inp_top | cpr_top)
            if len(all_top) > 10:
                pc = np.corrcoef(inp_norm[all_top], cpr_norm[all_top])[0, 1]
                if not np.isnan(pc):
                    peak_corrs.append(pc)

            # Entropy of normalized heatmap (discretize to 256 bins)
            def entropy(arr):
                hist, _ = np.histogram(arr, bins=256, range=(0, 1))
                hist = hist / hist.sum()
                hist = hist[hist > 0]
                return -np.sum(hist * np.log2(hist))

            inp_entropies.append(entropy(inp_norm))
            cpr_entropies.append(entropy(cpr_norm))

        if correlations:
            ratio = np.mean(cpr_stds) / (np.mean(inp_stds) + 1e-8)
            mean_corr = np.mean(correlations)
            results.append({
                'category': category,
                'corr': mean_corr,
                'ratio_std': ratio,
                'corr_ratio': mean_corr / ratio,
                'inp_snr': np.mean(inp_snrs),
                'cpr_snr': np.mean(cpr_snrs),
                'top_overlap': np.mean(top_overlaps),
                'peak_corr': np.mean(peak_corrs) if peak_corrs else 0,
                'inp_entropy': np.mean(inp_entropies),
                'cpr_entropy': np.mean(cpr_entropies),
                'n_images': len(correlations),
            })

    # Print results
    print(f"\n{'Category':<15} {'Corr':>6} {'Ratio':>6} {'C/R':>6} {'TopOvl':>7} {'PkCorr':>7} {'INP_SNR':>8} {'CPR_SNR':>8} {'INP_Ent':>8} {'CPR_Ent':>8}")
    print("-" * 95)
    for r in results:
        print(f"{r['category']:<15} {r['corr']:>6.3f} {r['ratio_std']:>6.2f} {r['corr_ratio']:>6.3f} "
              f"{r['top_overlap']:>7.3f} {r['peak_corr']:>7.3f} "
              f"{r['inp_snr']:>8.2f} {r['cpr_snr']:>8.2f} "
              f"{r['inp_entropy']:>8.3f} {r['cpr_entropy']:>8.3f}")


if __name__ == '__main__':
    main()
