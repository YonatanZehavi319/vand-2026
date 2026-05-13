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
    compute_auto_cpr_weights,
    compute_global_stats,
    fit_evt_from_validation,
    evt_threshold,
    post_process_heatmap,
    compute_mean_std_from_validation,
    mean_std_threshold,
    absolute_threshold,
    compute_val_max,
)


def run_ensemble(args):
    """Run the full ensemble pipeline."""
    categories = sorted(os.listdir(args.inp_dir))
    if args.item:
        categories = [c for c in categories if c == args.item]

    save_size = args.save_size
    combine_mode = getattr(args, 'combine_mode', 'average')
    median_sub = getattr(args, 'median_sub', False)
    cpr_power = getattr(args, 'cpr_power', 1.0)

    # Build post-processing args dict (shared between validation and test)
    pp_args = {
        'bilateral': getattr(args, 'bilateral', False),
        'bilateral_d': getattr(args, 'bilateral_d', 9),
        'bilateral_sc': getattr(args, 'bilateral_sc', 75),
        'bilateral_ss': getattr(args, 'bilateral_ss', 75),
        'guided': getattr(args, 'guided', False),
        'guided_r': getattr(args, 'guided_r', 8),
        'guided_eps': getattr(args, 'guided_eps', 0.01),
        'median_sub': median_sub,
    }

    # Where to find original validation images for guided filtering
    val_image_dir = getattr(args, 'val_image_dir', None)

    # Compute per-category CPR weights if auto mode
    auto_cpr = getattr(args, 'auto_cpr_weight', False)
    cpr_weights_per_cat = {}
    if auto_cpr and args.inp_val_dir:
        print("Computing auto CPR weights from INP SNR...")
        cpr_weights_per_cat = compute_auto_cpr_weights(args.inp_val_dir, categories, save_size)

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
                cat_cpr_weight = cpr_weights_per_cat.get(category, args.cpr_weight)
                params = fit_evt_from_validation(
                    args.inp_val_dir, args.cpr_val_dir, category,
                    save_size, args.inp_weight, cat_cpr_weight, global_stats=global_stats,
                    combine_mode=combine_mode, post_process_args=pp_args, val_image_dir=val_image_dir, cpr_power=cpr_power)
                if params is not None:
                    evt_params_per_cat[category] = params
            # Compute per-category FDR
            evt_fdr_per_cat = {}
            adaptive_fdr = getattr(args, 'adaptive_fdr', False)
            if adaptive_fdr and evt_params_per_cat:
                alpha = getattr(args, 'adaptive_strength', 0.3)
                abs_shapes = [abs(p[0]) for p in evt_params_per_cat.values()]
                median_shape = np.median(abs_shapes)
                print(f"\nAdaptive FDR (base={args.evt_fdr}, strength={alpha}, median_|shape|={median_shape:.4f}):")
                for cat, params in evt_params_per_cat.items():
                    cat_abs_shape = abs(params[0])
                    raw_fdr = args.evt_fdr * (cat_abs_shape / median_shape)
                    cat_fdr = alpha * raw_fdr + (1 - alpha) * args.evt_fdr
                    cat_fdr = np.clip(cat_fdr, 0.001, 0.5)
                    evt_fdr_per_cat[cat] = cat_fdr
                    print(f"  {cat}: |shape|={cat_abs_shape:.4f}, fdr={cat_fdr:.4f}")
            else:
                for cat in evt_params_per_cat:
                    evt_fdr_per_cat[cat] = args.evt_fdr
            thresholds_per_cat = {'method': 'evt', 'params': evt_params_per_cat, 'fdr_per_cat': evt_fdr_per_cat}

        elif args.threshold_method == 'val_max':
            print("Computing validation max thresholds...")
            val_maxes = {}
            for category in categories:
                cat_cpr_weight = cpr_weights_per_cat.get(category, args.cpr_weight)
                val_max = compute_val_max(
                    args.inp_val_dir, args.cpr_val_dir, category,
                    save_size, args.inp_weight, cat_cpr_weight,
                    global_stats=global_stats, percentile=args.val_percentile,
                    combine_mode=combine_mode, post_process_args=pp_args, val_image_dir=val_image_dir, cpr_power=cpr_power)
                if val_max is not None:
                    val_maxes[category] = val_max
            thresholds_per_cat = {'method': 'val_max', 'thresholds': val_maxes}

        elif args.threshold_method == 'mean_std':
            k = getattr(args, 'mean_std_k', 3.0)
            print(f"Computing mean+{k}*std thresholds from validation...")
            mean_std_per_cat = {}
            for category in categories:
                cat_cpr_weight = cpr_weights_per_cat.get(category, args.cpr_weight)
                params = compute_mean_std_from_validation(
                    args.inp_val_dir, args.cpr_val_dir, category,
                    save_size, args.inp_weight, cat_cpr_weight,
                    global_stats=global_stats, combine_mode=combine_mode,
                    post_process_args=pp_args, val_image_dir=val_image_dir, cpr_power=cpr_power)
                if params is not None:
                    mean_std_per_cat[category] = params
            thresholds_per_cat = {'method': 'mean_std', 'params': mean_std_per_cat, 'k': k}

    # Process test images
    for category in categories:
        inp_cat_dir = os.path.join(args.inp_dir, category)
        cpr_cat_dir = os.path.join(args.cpr_dir, category)

        if not os.path.isdir(inp_cat_dir):
            print(f"  Skipping {category}: INP dir not found")
            continue

        sub_dirs = sorted(os.listdir(inp_cat_dir))
        n_combined = 0
        cat_cpr_weight = cpr_weights_per_cat.get(category, args.cpr_weight)

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
                    inp_sub, cpr_sub, fname, save_size, args.inp_weight, cat_cpr_weight,
                    global_stats=global_stats, combine_mode=combine_mode, cpr_power=cpr_power)
                if combined is None:
                    continue
                if was_combined:
                    n_combined += 1

                # Apply post-processing (same pipeline as validation)
                guide_img = None
                if pp_args.get('guided') and args.data_dir:
                    for split_name in ['test_public', 'test']:
                        for ext in ['.png', '.JPG', '.bmp']:
                            p = os.path.join(args.data_dir, category, split_name, sub_dir, fname + ext)
                            if os.path.exists(p):
                                guide_img = cv2.imread(p)
                                guide_img = cv2.resize(guide_img, (save_size, save_size))
                                break
                        if guide_img is not None:
                            break
                combined = post_process_heatmap(combined, guide_img=guide_img, **pp_args)

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
                            cat_fdr = thresholds_per_cat['fdr_per_cat'].get(category, args.evt_fdr)
                            pred_mask = evt_threshold(combined, params, fdr=cat_fdr)
                        else:
                            pred_mask = _otsu_fallback(combined)
                    elif thresholds_per_cat['method'] == 'val_max':
                        val_max = thresholds_per_cat['thresholds'].get(category)
                        if val_max is not None:
                            pred_mask = absolute_threshold(combined, val_max)
                        else:
                            pred_mask = _otsu_fallback(combined)
                    elif thresholds_per_cat['method'] == 'mean_std':
                        ms_params = thresholds_per_cat['params'].get(category)
                        if ms_params is not None:
                            pred_mask = mean_std_threshold(combined, ms_params, k=thresholds_per_cat['k'])
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
    parser.add_argument('--threshold_method', type=str, default='evt', choices=['evt', 'val_max', 'otsu', 'mean_std'])
    parser.add_argument('--evt_fdr', type=float, default=0.01)
    parser.add_argument('--val_percentile', type=float, default=99.9, help='Percentile for val_max threshold')
    parser.add_argument('--mean_std_k', type=float, default=3.0, help='k for mean+k*std threshold (default 3.0)')
    parser.add_argument('--combine_mode', type=str, default='average', choices=['average', 'boost'],
                        help='How to combine heatmaps: average (weighted avg) or boost (CPR boosts INP)')
    # Smoothing options
    parser.add_argument('--bilateral', action='store_true', help='Apply bilateral filter to combined heatmap')
    parser.add_argument('--bilateral_d', type=int, default=9, help='Bilateral filter diameter (default 9)')
    parser.add_argument('--bilateral_sc', type=float, default=75, help='Bilateral filter sigmaColor (default 75)')
    parser.add_argument('--bilateral_ss', type=float, default=75, help='Bilateral filter sigmaSpace (default 75)')
    parser.add_argument('--guided', action='store_true', help='Apply guided filter using original image')
    parser.add_argument('--guided_r', type=int, default=8, help='Guided filter radius (default 8)')
    parser.add_argument('--guided_eps', type=float, default=0.01, help='Guided filter eps (default 0.01)')
    # Adaptive FDR
    parser.add_argument('--adaptive_fdr', action='store_true', help='Scale FDR per category based on GEV shape')
    parser.add_argument('--adaptive_strength', type=float, default=0.3, help='Blend: 0=uniform, 1=full adaptive (default 0.3)')
    # Per-image normalization
    parser.add_argument('--median_sub', action='store_true', help='Subtract per-image median before thresholding')
    parser.add_argument('--val_image_dir', type=str, default=None, help='Validation images dir (for guided filter during EVT fitting)')
    # Auto CPR weight
    parser.add_argument('--auto_cpr_weight', action='store_true', help='Auto-compute per-category CPR weight from INP SNR')
    parser.add_argument('--cpr_power', type=float, default=1.0, help='Power applied to CPR signal in boost mode (default 1.0, >1 sharpens)')

    args = parser.parse_args()
    run_ensemble(args)


if __name__ == '__main__':
    main()
