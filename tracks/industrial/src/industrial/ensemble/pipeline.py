"""End-to-end ensemble pipeline: load heatmaps -> combine -> threshold -> save.

Outputs in competition submission format:
  anomaly_images/{category}/{split}/{idx}_{suffix}.tiff         (float16)
  anomaly_images_thresholded/{category}/{split}/{idx}_{suffix}.png  (binary {0,255})
"""

import argparse
import os
import sys
from glob import glob

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from industrial.ensemble.combine import (
    combine_heatmaps,
    compute_global_stats,
    fit_evt_from_validation,
    evt_threshold,
    absolute_threshold,
    compute_val_max,
)


def run_ensemble(args):
    """Run the full ensemble pipeline."""
    categories = sorted(os.listdir(args.inp_dir))
    if args.item:
        categories = [c for c in categories if c == args.item]

    save_size = args.save_size

    # Compute global stats and fit threshold from validation
    global_stats = None
    thresholds_per_cat = {}

    if args.inp_val_dir and args.cpr_val_dir:
        norm_mode = 'zscore' if args.zscore else 'minmax'
        print(f"Computing global normalization stats from validation ({norm_mode})...")
        global_stats = compute_global_stats(args.inp_val_dir, args.cpr_val_dir, categories, save_size, mode=norm_mode)

        if args.threshold_method == 'evt':
            print("Fitting EVT from validation heatmaps...")
            evt_params_per_cat = {}
            for category in categories:
                params = fit_evt_from_validation(
                    args.inp_val_dir, args.cpr_val_dir, category,
                    save_size, args.inp_weight, args.cpr_weight, global_stats=global_stats)
                if params is not None:
                    evt_params_per_cat[category] = params
            thresholds_per_cat = {'method': 'evt', 'params': evt_params_per_cat, 'fdr': args.evt_fdr}

        elif args.threshold_method == 'val_max':
            print("Computing validation max thresholds...")
            val_maxes = {}
            for category in categories:
                val_max = compute_val_max(
                    args.inp_val_dir, args.cpr_val_dir, category,
                    save_size, args.inp_weight, args.cpr_weight,
                    global_stats=global_stats, percentile=args.val_percentile)
                if val_max is not None:
                    val_maxes[category] = val_max
            thresholds_per_cat = {'method': 'val_max', 'thresholds': val_maxes}

    # Process test images
    for category in categories:
        inp_cat_dir = os.path.join(args.inp_dir, category)
        cpr_cat_dir = os.path.join(args.cpr_dir, category)

        if not os.path.isdir(inp_cat_dir):
            print(f"  Skipping {category}: INP dir not found")
            continue

        sub_dirs = sorted(os.listdir(inp_cat_dir))
        n_combined = 0

        for sub_dir in sub_dirs:
            inp_sub = os.path.join(inp_cat_dir, sub_dir)
            cpr_sub = os.path.join(cpr_cat_dir, sub_dir)

            # Output dirs
            heatmap_out = os.path.join(args.out_dir, 'heatmaps', category, sub_dir)
            anomaly_out = os.path.join(args.out_dir, 'anomaly_images', category, sub_dir)
            thresh_out = os.path.join(args.out_dir, 'anomaly_images_thresholded', category, sub_dir)
            os.makedirs(heatmap_out, exist_ok=True)
            os.makedirs(anomaly_out, exist_ok=True)
            os.makedirs(thresh_out, exist_ok=True)

            inp_heatmaps = sorted(glob(os.path.join(inp_sub, '*_heatmap_raw.npy')))

            for npy_path in inp_heatmaps:
                fname = os.path.basename(npy_path).replace('_heatmap_raw.npy', '')

                combined, was_combined = combine_heatmaps(
                    inp_sub, cpr_sub, fname, save_size, args.inp_weight, args.cpr_weight,
                    global_stats=global_stats)
                if combined is None:
                    continue
                if was_combined:
                    n_combined += 1

                # Save heatmap visualization
                plt.imsave(os.path.join(heatmap_out, f'{fname}_heatmap.png'), combined, cmap='jet')

                # Save float16 TIFF (submission format)
                combined_f16 = combined.astype(np.float16)
                try:
                    import tifffile
                    tifffile.imwrite(os.path.join(anomaly_out, f'{fname}.tiff'), combined_f16)
                except ImportError:
                    # Fallback: save as npy
                    np.save(os.path.join(anomaly_out, f'{fname}.npy'), combined_f16)

                # Binary mask
                if thresholds_per_cat:
                    if thresholds_per_cat['method'] == 'evt':
                        params = thresholds_per_cat['params'].get(category)
                        if params is not None:
                            pred_mask = evt_threshold(combined, params, fdr=thresholds_per_cat['fdr'])
                        else:
                            pred_mask = _otsu_fallback(combined)
                    elif thresholds_per_cat['method'] == 'val_max':
                        val_max = thresholds_per_cat['thresholds'].get(category)
                        if val_max is not None:
                            pred_mask = absolute_threshold(combined, val_max)
                        else:
                            pred_mask = _otsu_fallback(combined)
                else:
                    pred_mask = _otsu_fallback(combined)

                # Save binary PNG (submission format)
                cv2.imwrite(os.path.join(thresh_out, f'{fname}.png'), pred_mask)

                # Also save in heatmaps dir for evaluation
                plt.imsave(os.path.join(heatmap_out, f'{fname}_binary.png'), pred_mask, cmap='gray')

                # Copy GT if exists (check AD 2 then AD 1 layout)
                if args.data_dir:
                    gt_path_ad2 = os.path.join(args.data_dir, category, 'test_public', 'ground_truth', sub_dir, f'{fname}_mask.png')
                    gt_path_ad1 = os.path.join(args.data_dir, category, 'ground_truth', sub_dir, f'{fname}_mask.png')
                    gt_path = gt_path_ad2 if os.path.exists(gt_path_ad2) else gt_path_ad1
                    if os.path.exists(gt_path):
                        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                        gt_resized = cv2.resize(gt, (save_size, save_size), interpolation=cv2.INTER_NEAREST)
                        plt.imsave(os.path.join(heatmap_out, f'{fname}_gt.png'), gt_resized, cmap='gray')

                plt.close('all')

        print(f"  {category}: {n_combined} images combined")

    print(f"\nEnsemble results saved to {args.out_dir}/")


def _otsu_fallback(combined):
    combined_uint8 = (combined * 255).astype(np.uint8)
    _, pred_mask = cv2.threshold(combined_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return pred_mask


def main():
    parser = argparse.ArgumentParser(description='Ensemble pipeline: INP-Former + CPR')
    parser.add_argument('--inp_dir', type=str, required=True, help='INP-Former test heatmaps dir')
    parser.add_argument('--cpr_dir', type=str, required=True, help='CPR test heatmaps dir')
    parser.add_argument('--data_dir', type=str, default=None, help='Dataset root (for GT masks)')
    parser.add_argument('--out_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--item', type=str, default=None, help='Single category')
    parser.add_argument('--save_size', type=int, default=512, help='Output size')
    parser.add_argument('--inp_weight', type=float, default=1.0)
    parser.add_argument('--cpr_weight', type=float, default=1.0)
    parser.add_argument('--zscore', action='store_true', help='Use z-score normalization')
    # Validation dirs for threshold fitting
    parser.add_argument('--inp_val_dir', type=str, default=None)
    parser.add_argument('--cpr_val_dir', type=str, default=None)
    # Threshold method
    parser.add_argument('--threshold_method', type=str, default='evt', choices=['evt', 'val_max', 'otsu'])
    parser.add_argument('--evt_fdr', type=float, default=0.01)
    parser.add_argument('--val_percentile', type=float, default=99.9, help='Percentile for val_max threshold')

    args = parser.parse_args()
    run_ensemble(args)


if __name__ == '__main__':
    main()
