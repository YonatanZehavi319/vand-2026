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


def mutual_information(x, y, bins=64):
    """Compute mutual information between two arrays."""
    hist_2d, _, _ = np.histogram2d(x, y, bins=bins)
    pxy = hist_2d / hist_2d.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    px_py = px[:, None] * py[None, :]
    nonzero = pxy > 0
    return np.sum(pxy[nonzero] * np.log2(pxy[nonzero] / (px_py[nonzero] + 1e-12)))


def gradient_correlation(map1, map2, size):
    """Correlation between gradient magnitudes (edge agreement)."""
    m1 = map1.reshape(size, size)
    m2 = map2.reshape(size, size)
    gx1 = cv2.Sobel(m1, cv2.CV_32F, 1, 0, ksize=3)
    gy1 = cv2.Sobel(m1, cv2.CV_32F, 0, 1, ksize=3)
    gx2 = cv2.Sobel(m2, cv2.CV_32F, 1, 0, ksize=3)
    gy2 = cv2.Sobel(m2, cv2.CV_32F, 0, 1, ksize=3)
    grad1 = np.sqrt(gx1**2 + gy1**2).flatten()
    grad2 = np.sqrt(gx2**2 + gy2**2).flatten()
    return np.corrcoef(grad1, grad2)[0, 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp_dir', type=str, required=True)
    parser.add_argument('--cpr_dir', type=str, required=True)
    parser.add_argument('--save_size', type=int, default=512)
    parser.add_argument('--top_pct', type=float, default=5.0)
    args = parser.parse_args()

    categories = sorted(os.listdir(args.inp_dir))
    top_pct = args.top_pct
    sz = args.save_size

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
        peak_corrs = []
        inp_entropies = []
        cpr_entropies = []
        mut_infos = []
        edge_corrs = []
        inp_sparsities = []
        cpr_sparsities = []
        score_stabilities_inp = []
        score_stabilities_cpr = []
        cpr_confidences = []

        for npy_path in inp_files:
            fname = os.path.basename(npy_path).replace('_heatmap_raw.npy', '')
            cpr_npy = os.path.join(cpr_good, f'{fname}_heatmap_raw.npy')

            if not os.path.exists(cpr_npy):
                continue

            inp_map = np.load(npy_path).astype(np.float32)
            cpr_map = np.load(cpr_npy).astype(np.float32)

            inp_map = cv2.resize(inp_map, (sz, sz)).flatten()
            cpr_map = cv2.resize(cpr_map, (sz, sz)).flatten()

            inp_norm = (inp_map - inp_map.min()) / (inp_map.max() - inp_map.min() + 1e-8)
            cpr_norm = (cpr_map - cpr_map.min()) / (cpr_map.max() - cpr_map.min() + 1e-8)

            # Correlation
            corr = np.corrcoef(inp_norm, cpr_norm)[0, 1]
            correlations.append(corr)

            # Std
            inp_stds.append(inp_map.std())
            cpr_stds.append(cpr_map.std())

            # SNR
            inp_snrs.append(inp_map.max() / (inp_map.mean() + 1e-8))
            cpr_snrs.append(cpr_map.max() / (cpr_map.mean() + 1e-8))

            # Top overlap
            n_top = max(1, int(len(inp_norm) * top_pct / 100))
            inp_top = set(np.argsort(inp_norm)[-n_top:])
            cpr_top = set(np.argsort(cpr_norm)[-n_top:])
            overlap = len(inp_top & cpr_top) / n_top
            top_overlaps.append(overlap)

            # Peak correlation
            all_top = sorted(inp_top | cpr_top)
            if len(all_top) > 10:
                pc = np.corrcoef(inp_norm[all_top], cpr_norm[all_top])[0, 1]
                if not np.isnan(pc):
                    peak_corrs.append(pc)

            # Entropy
            def entropy(arr):
                hist, _ = np.histogram(arr, bins=256, range=(0, 1))
                hist = hist / hist.sum()
                hist = hist[hist > 0]
                return -np.sum(hist * np.log2(hist))

            inp_entropies.append(entropy(inp_norm))
            cpr_entropies.append(entropy(cpr_norm))

            # Mutual information
            mi = mutual_information(inp_norm, cpr_norm)
            mut_infos.append(mi)

            # Edge/gradient agreement
            gc = gradient_correlation(inp_norm, cpr_norm, sz)
            if not np.isnan(gc):
                edge_corrs.append(gc)

            # Sparsity: % of pixels above 90th percentile (how focused)
            inp_sparsities.append((inp_norm > np.percentile(inp_norm, 90)).mean())
            cpr_sparsities.append((cpr_norm > np.percentile(cpr_norm, 90)).mean())

            # Score stability: track max scores for std computation
            score_stabilities_inp.append(inp_map.max())
            score_stabilities_cpr.append(cpr_map.max())

            # CPR confidence: mean of top 5% scores
            cpr_confidences.append(np.mean(np.sort(cpr_map)[-n_top:]))

        if correlations:
            ratio = np.mean(cpr_stds) / (np.mean(inp_stds) + 1e-8)
            mean_corr = np.mean(correlations)
            inp_stability = np.std(score_stabilities_inp) / (np.mean(score_stabilities_inp) + 1e-8)
            cpr_stability = np.std(score_stabilities_cpr) / (np.mean(score_stabilities_cpr) + 1e-8)

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
                'mut_info': np.mean(mut_infos),
                'edge_corr': np.mean(edge_corrs) if edge_corrs else 0,
                'inp_stability': inp_stability,
                'cpr_stability': cpr_stability,
                'cpr_confidence': np.mean(cpr_confidences),
                'n_images': len(correlations),
            })

    # Print original metrics
    print(f"\n{'Category':<15} {'Corr':>6} {'Ratio':>6} {'C/R':>6} {'TopOvl':>7} {'PkCorr':>7} {'INP_SNR':>8} {'CPR_SNR':>8} {'INP_Ent':>8} {'CPR_Ent':>8}")
    print("-" * 95)
    for r in results:
        print(f"{r['category']:<15} {r['corr']:>6.3f} {r['ratio_std']:>6.2f} {r['corr_ratio']:>6.3f} "
              f"{r['top_overlap']:>7.3f} {r['peak_corr']:>7.3f} "
              f"{r['inp_snr']:>8.2f} {r['cpr_snr']:>8.2f} "
              f"{r['inp_entropy']:>8.3f} {r['cpr_entropy']:>8.3f}")

    # Print new metrics
    print(f"\n{'Category':<15} {'MutInfo':>8} {'EdgeCorr':>9} {'INP_Stab':>9} {'CPR_Stab':>9} {'CPR_Conf':>9}")
    print("-" * 60)
    for r in results:
        print(f"{r['category']:<15} {r['mut_info']:>8.4f} {r['edge_corr']:>9.4f} "
              f"{r['inp_stability']:>9.4f} {r['cpr_stability']:>9.4f} {r['cpr_confidence']:>9.4f}")


if __name__ == '__main__':
    main()
