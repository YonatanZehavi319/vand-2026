"""Ensemble combination and thresholding for INP-Former + CPR heatmaps.

Core functions extracted from ensemble_cpr.py.
"""

import numpy as np
import cv2
import os
from glob import glob
from scipy.stats import genextreme


def normalize_map(amap):
    return (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)


def load_heatmap(path):
    """Load a saved heatmap PNG as grayscale."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return img.astype(np.float32) / 255.0


def load_heatmap_npy(path):
    """Load raw heatmap from .npy file if available."""
    if os.path.exists(path):
        return np.load(path).astype(np.float32)
    return None


def _gradient_correlation(map1, map2, size):
    """Correlation between gradient magnitudes of two maps."""
    m1 = map1.reshape(size, size)
    m2 = map2.reshape(size, size)
    gx1 = cv2.Sobel(m1, cv2.CV_32F, 1, 0, ksize=3)
    gy1 = cv2.Sobel(m1, cv2.CV_32F, 0, 1, ksize=3)
    gx2 = cv2.Sobel(m2, cv2.CV_32F, 1, 0, ksize=3)
    gy2 = cv2.Sobel(m2, cv2.CV_32F, 0, 1, ksize=3)
    grad1 = np.sqrt(gx1**2 + gy1**2).flatten()
    grad2 = np.sqrt(gx2**2 + gy2**2).flatten()
    corr = np.corrcoef(grad1, grad2)[0, 1]
    return corr if not np.isnan(corr) else 0.0


def compute_auto_cpr_weights(inp_val_dir, cpr_val_dir, categories, save_size):
    """Compute per-category CPR weight based on edge correlation: max(0, EdgeCorr - 0.1) * 1.15."""
    weights = {}
    for category in categories:
        inp_good = os.path.join(inp_val_dir, category, 'good')
        cpr_good = os.path.join(cpr_val_dir, category, 'good')
        if not os.path.isdir(inp_good):
            weights[category] = 0.0
            continue

        edge_corrs = []
        for npy_path in sorted(glob(os.path.join(inp_good, '*_heatmap_raw.npy'))):
            fname = os.path.basename(npy_path).replace('_heatmap_raw.npy', '')
            cpr_npy = os.path.join(cpr_good, f'{fname}_heatmap_raw.npy')
            if not os.path.exists(cpr_npy):
                continue

            inp_map = np.load(npy_path).astype(np.float32)
            cpr_map = np.load(cpr_npy).astype(np.float32)
            inp_map = cv2.resize(inp_map, (save_size, save_size))
            cpr_map = cv2.resize(cpr_map, (save_size, save_size))

            inp_norm = (inp_map - inp_map.min()) / (inp_map.max() - inp_map.min() + 1e-8)
            cpr_norm = (cpr_map - cpr_map.min()) / (cpr_map.max() - cpr_map.min() + 1e-8)

            ec = _gradient_correlation(inp_norm, cpr_norm, save_size)
            edge_corrs.append(ec)

        if edge_corrs:
            mean_ec = np.mean(edge_corrs)
            w = max(0.0, mean_ec)
            weights[category] = w
            print(f"  {category}: EdgeCorr={mean_ec:.4f}, cpr_weight={w:.3f}")
        else:
            weights[category] = 0.0

    return weights


def compute_spatial_prior(inp_val_dir, cpr_val_dir, categories, save_size, inp_weight, cpr_weight_map,
                          global_stats=None, combine_mode='average', cpr_power=1.0, grid_size=4, suppress_floor=0.3):
    """Build per-category spatial suppression maps from validation heatmaps.

    For each category, divides the heatmap into a grid and computes the mean
    anomaly score per cell across all validation images. High-scoring cells
    are areas prone to false positives. Returns a dict of suppression maps
    (lower values = more suppression)."""
    priors = {}
    cell_h = save_size // grid_size
    cell_w = save_size // grid_size

    for category in categories:
        inp_good = os.path.join(inp_val_dir, category, 'good')
        cpr_good = os.path.join(cpr_val_dir, category, 'good')
        if not os.path.isdir(inp_good):
            continue

        cat_cpr_weight = cpr_weight_map.get(category, 1.0)
        grid_sums = np.zeros((grid_size, grid_size), dtype=np.float64)
        n_images = 0

        for npy_path in sorted(glob(os.path.join(inp_good, '*_heatmap_raw.npy'))):
            fname = os.path.basename(npy_path).replace('_heatmap_raw.npy', '')
            combined, _ = combine_heatmaps(inp_good, cpr_good, fname, save_size,
                                           inp_weight, cat_cpr_weight,
                                           global_stats=global_stats,
                                           combine_mode=combine_mode, cpr_power=cpr_power)
            if combined is None:
                continue
            for r in range(grid_size):
                for c in range(grid_size):
                    cell = combined[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
                    grid_sums[r, c] += cell.mean()
            n_images += 1

        if n_images == 0:
            continue

        grid_means = grid_sums / n_images
        # Invert: high mean → low weight (suppress FP areas)
        # Scale so min cell gets weight 1.0, max cell gets lower weight
        gmax = grid_means.max()
        gmin = grid_means.min()
        if gmax - gmin > 1e-8:
            # Normalize to [0, 1] where 0 = highest FP area, 1 = lowest
            grid_weights = 1.0 - (grid_means - gmin) / (gmax - gmin)
            # Scale to [suppress_floor, 1.0] so we don't fully zero out any region
            grid_weights = suppress_floor + grid_weights * (1.0 - suppress_floor)
        else:
            grid_weights = np.ones((grid_size, grid_size))

        # Upscale to full resolution with smooth interpolation
        prior_map = cv2.resize(grid_weights.astype(np.float32), (save_size, save_size),
                               interpolation=cv2.INTER_LINEAR)
        priors[category] = prior_map
        print(f"  {category}: spatial prior grid (min={grid_weights.min():.3f}, max={grid_weights.max():.3f})")

    return priors


def combine_heatmaps(inp_dir, cpr_dir, fname, save_size, inp_weight, cpr_weight, global_stats=None, combine_mode='average', cpr_power=1.0):
    """Load and combine INP-Former + CPR heatmaps for a single image.

    Args:
        combine_mode: 'average' (weighted average), 'boost' (CPR boosts INP),
                      or 'gated_boost' (CPR boosts INP only where CPR is confident)

    Returns the combined heatmap, or None if INP heatmap not found.
    """
    inp_npy = os.path.join(inp_dir, f'{fname}_heatmap_raw.npy')
    inp_map = load_heatmap_npy(inp_npy)
    if inp_map is None:
        inp_path = os.path.join(inp_dir, f'{fname}_heatmap.png')
        inp_map = load_heatmap(inp_path)
    if inp_map is None:
        return None, False

    inp_resized = cv2.resize(inp_map, (save_size, save_size))
    if global_stats is not None and global_stats['mode'] == 'zscore':
        inp_val = (inp_resized - global_stats['inp_mean']) / (global_stats['inp_std'] + 1e-8)
    elif global_stats is not None:
        inp_val = (inp_resized - global_stats['inp_min']) / (global_stats['inp_max'] - global_stats['inp_min'] + 1e-8)
    else:
        inp_val = normalize_map(inp_resized)

    cpr_npy = os.path.join(cpr_dir, f'{fname}_heatmap_raw.npy')
    cpr_map = load_heatmap_npy(cpr_npy)
    if cpr_map is None:
        cpr_path = os.path.join(cpr_dir, f'{fname}_heatmap.png')
        cpr_map = load_heatmap(cpr_path)

    if cpr_map is not None:
        cpr_resized = cv2.resize(cpr_map, (save_size, save_size))
        if global_stats is not None and global_stats['mode'] == 'zscore':
            cpr_val = (cpr_resized - global_stats['cpr_mean']) / (global_stats['cpr_std'] + 1e-8)
        elif global_stats is not None:
            cpr_val = (cpr_resized - global_stats['cpr_min']) / (global_stats['cpr_max'] - global_stats['cpr_min'] + 1e-8)
        else:
            cpr_val = normalize_map(cpr_resized)

        if combine_mode == 'boost':
            combined = inp_val * (1 + cpr_weight * cpr_val ** cpr_power)
        elif combine_mode == 'gated_boost':
            gate = (cpr_val > np.percentile(cpr_val, 95)).astype(np.float32)
            combined = inp_val * (1 + cpr_weight * cpr_val ** cpr_power * gate)
        else:
            combined = (inp_weight * inp_val + cpr_weight * cpr_val) / (inp_weight + cpr_weight)
        return combined, True
    else:
        return inp_val, False


def compute_global_stats(inp_val_dir, cpr_val_dir, categories, save_size, mode='minmax'):
    """Compute global normalization stats per model across all validation heatmaps."""
    inp_pixels = []
    cpr_pixels = []

    for category in categories:
        inp_good = os.path.join(inp_val_dir, category, 'good')
        cpr_good = os.path.join(cpr_val_dir, category, 'good')

        for npy_path in sorted(glob(os.path.join(inp_good, '*_heatmap_raw.npy'))):
            amap = np.load(npy_path).astype(np.float32)
            amap = cv2.resize(amap, (save_size, save_size))
            inp_pixels.append(amap.flatten())

        for npy_path in sorted(glob(os.path.join(cpr_good, '*_heatmap_raw.npy'))):
            amap = np.load(npy_path).astype(np.float32)
            amap = cv2.resize(amap, (save_size, save_size))
            cpr_pixels.append(amap.flatten())

    inp_all = np.concatenate(inp_pixels).astype(np.float32)
    cpr_all = np.concatenate(cpr_pixels).astype(np.float32)

    if mode == 'zscore':
        stats = {
            'mode': 'zscore',
            'inp_mean': inp_all.mean(), 'inp_std': inp_all.std(),
            'cpr_mean': cpr_all.mean(), 'cpr_std': cpr_all.std(),
        }
    else:
        stats = {
            'mode': 'minmax',
            'inp_min': inp_all.min(), 'inp_max': inp_all.max(),
            'cpr_min': cpr_all.min(), 'cpr_max': cpr_all.max(),
        }

    return stats


def post_process_heatmap(combined, guide_img=None, bilateral=False, bilateral_d=9, bilateral_sc=75, bilateral_ss=75,
                         guided=False, guided_r=8, guided_eps=0.01, median_sub=False):
    """Apply post-processing to a combined heatmap. Same pipeline for validation and test."""
    if bilateral:
        combined = cv2.bilateralFilter(combined.astype(np.float32), d=bilateral_d,
                                        sigmaColor=bilateral_sc, sigmaSpace=bilateral_ss)
    if guided and guide_img is not None:
        guide_gray = cv2.cvtColor(guide_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        combined = cv2.ximgproc.guidedFilter(guide_gray, combined.astype(np.float32),
                                              radius=guided_r, eps=guided_eps)
    if median_sub:
        combined = combined - np.median(combined)
    return combined


def _find_val_image(val_dir, category, fname):
    """Find the original validation image for guided filtering."""
    for ext in ['.png', '.JPG', '.bmp']:
        p = os.path.join(val_dir, category, 'validation', 'good', fname + ext)
        if os.path.exists(p):
            return p
    return None


def fit_evt_from_validation(inp_val_dir, cpr_val_dir, category, save_size, inp_weight, cpr_weight,
                            global_stats=None, combine_mode='average', post_process_args=None, val_image_dir=None, cpr_power=1.0):
    """Fit GEV distribution on combined validation/good heatmaps for a category."""
    inp_val_good = os.path.join(inp_val_dir, category, 'good')
    cpr_val_good = os.path.join(cpr_val_dir, category, 'good')

    if not os.path.isdir(inp_val_good):
        print(f"  WARNING: No INP validation heatmaps for {category}")
        return None

    inp_npy_files = sorted(glob(os.path.join(inp_val_good, '*_heatmap_raw.npy')))
    pp = post_process_args or {}

    all_pixel_scores = []
    for npy_path in inp_npy_files:
        fname = os.path.basename(npy_path).replace('_heatmap_raw.npy', '')
        combined, _ = combine_heatmaps(inp_val_good, cpr_val_good, fname, save_size, inp_weight, cpr_weight, global_stats=global_stats, combine_mode=combine_mode, cpr_power=cpr_power)
        if combined is not None:
            # Apply same post-processing as test
            guide_img = None
            if pp.get('guided') and val_image_dir:
                img_path = _find_val_image(val_image_dir, category, fname)
                if img_path is not None:
                    guide_img = cv2.imread(img_path)
                    guide_img = cv2.resize(guide_img, (save_size, save_size))
            combined = post_process_heatmap(combined, guide_img=guide_img, **pp)
            all_pixel_scores.append(combined.flatten())

    if not all_pixel_scores:
        return None

    all_pixel_scores = np.concatenate(all_pixel_scores)
    tail_threshold = np.percentile(all_pixel_scores, 95)
    tail_scores = all_pixel_scores[all_pixel_scores >= tail_threshold]
    if len(tail_scores) > 500000:
        tail_scores = np.random.choice(tail_scores, 500000, replace=False)
    print(f'  {category}: fitting GEV on {len(tail_scores)} tail samples...')
    shape, loc, scale = genextreme.fit(tail_scores)
    print(f'  {category}: EVT fit: shape={shape:.4f}, loc={loc:.6f}, scale={scale:.6f}')
    return shape, loc, scale


def evt_threshold(combined_map, evt_params, fdr=0.01):
    """Apply EVT-based thresholding. Pixels with p-value < fdr are anomalous."""
    shape, loc, scale = evt_params
    p_values = 1 - genextreme.cdf(combined_map, shape, loc=loc, scale=scale)
    return ((p_values < fdr) * 255).astype(np.uint8)


def absolute_threshold(combined_map, val_max):
    """Threshold by validation max. Pixels above val_max are anomalous."""
    return ((combined_map > val_max) * 255).astype(np.uint8)


def compute_mean_std_from_validation(inp_val_dir, cpr_val_dir, category, save_size, inp_weight, cpr_weight,
                                     global_stats=None, combine_mode='average', post_process_args=None, val_image_dir=None, cpr_power=1.0):
    """Compute mean and std of combined validation scores for a category."""
    inp_val_good = os.path.join(inp_val_dir, category, 'good')
    cpr_val_good = os.path.join(cpr_val_dir, category, 'good')

    if not os.path.isdir(inp_val_good):
        print(f"  WARNING: No INP validation heatmaps for {category}")
        return None

    inp_npy_files = sorted(glob(os.path.join(inp_val_good, '*_heatmap_raw.npy')))
    pp = post_process_args or {}

    all_pixel_scores = []
    for npy_path in inp_npy_files:
        fname = os.path.basename(npy_path).replace('_heatmap_raw.npy', '')
        combined, _ = combine_heatmaps(inp_val_good, cpr_val_good, fname, save_size, inp_weight, cpr_weight, global_stats=global_stats, combine_mode=combine_mode, cpr_power=cpr_power)
        if combined is not None:
            guide_img = None
            if pp.get('guided') and val_image_dir:
                img_path = _find_val_image(val_image_dir, category, fname)
                if img_path is not None:
                    guide_img = cv2.imread(img_path)
                    guide_img = cv2.resize(guide_img, (save_size, save_size))
            combined = post_process_heatmap(combined, guide_img=guide_img, **pp)
            all_pixel_scores.append(combined.flatten())

    if not all_pixel_scores:
        return None

    all_pixel_scores = np.concatenate(all_pixel_scores)
    mean = float(all_pixel_scores.mean())
    std = float(all_pixel_scores.std())
    print(f'  {category}: mean={mean:.6f}, std={std:.6f}')
    return mean, std


def mean_std_threshold(combined_map, mean_std_params, k=3.0):
    """Threshold at mean + k*std. Pixels above threshold are anomalous."""
    mean, std = mean_std_params
    threshold = mean + k * std
    return ((combined_map > threshold) * 255).astype(np.uint8)


def compute_val_max(inp_val_dir, cpr_val_dir, category, save_size, inp_weight, cpr_weight,
                    global_stats=None, percentile=99.9, combine_mode='average', post_process_args=None, val_image_dir=None, cpr_power=1.0):
    """Compute the near-max of validation combined scores for a category."""
    inp_val_good = os.path.join(inp_val_dir, category, 'good')
    cpr_val_good = os.path.join(cpr_val_dir, category, 'good')

    inp_npy_files = sorted(glob(os.path.join(inp_val_good, '*_heatmap_raw.npy')))
    pp = post_process_args or {}

    all_pixel_scores = []
    for npy_path in inp_npy_files:
        fname = os.path.basename(npy_path).replace('_heatmap_raw.npy', '')
        combined, _ = combine_heatmaps(inp_val_good, cpr_val_good, fname, save_size, inp_weight, cpr_weight, global_stats=global_stats, combine_mode=combine_mode, cpr_power=cpr_power)
        if combined is not None:
            guide_img = None
            if pp.get('guided') and val_image_dir:
                img_path = _find_val_image(val_image_dir, category, fname)
                if img_path is not None:
                    guide_img = cv2.imread(img_path)
                    guide_img = cv2.resize(guide_img, (save_size, save_size))
            combined = post_process_heatmap(combined, guide_img=guide_img, **pp)
            all_pixel_scores.append(combined.flatten())

    if not all_pixel_scores:
        return None

    all_pixel_scores = np.concatenate(all_pixel_scores)
    threshold = np.percentile(all_pixel_scores, percentile)
    print(f'  {category}: val {percentile}th percentile threshold = {threshold:.6f}')
    return threshold
