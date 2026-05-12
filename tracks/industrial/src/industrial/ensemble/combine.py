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


def combine_heatmaps(inp_dir, cpr_dir, fname, save_size, inp_weight, cpr_weight, global_stats=None, combine_mode='average'):
    """Load and combine INP-Former + CPR heatmaps for a single image.

    Args:
        combine_mode: 'average' (weighted average) or 'boost' (CPR boosts INP signal)

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
            combined = inp_val * (1 + cpr_weight * cpr_val)
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


def fit_evt_from_validation(inp_val_dir, cpr_val_dir, category, save_size, inp_weight, cpr_weight, global_stats=None, combine_mode='average'):
    """Fit GEV distribution on combined validation/good heatmaps for a category."""
    inp_val_good = os.path.join(inp_val_dir, category, 'good')
    cpr_val_good = os.path.join(cpr_val_dir, category, 'good')

    if not os.path.isdir(inp_val_good):
        print(f"  WARNING: No INP validation heatmaps for {category}")
        return None

    inp_npy_files = sorted(glob(os.path.join(inp_val_good, '*_heatmap_raw.npy')))

    all_pixel_scores = []
    for npy_path in inp_npy_files:
        fname = os.path.basename(npy_path).replace('_heatmap_raw.npy', '')
        combined, _ = combine_heatmaps(inp_val_good, cpr_val_good, fname, save_size, inp_weight, cpr_weight, global_stats=global_stats, combine_mode=combine_mode)
        if combined is not None:
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


def compute_val_max(inp_val_dir, cpr_val_dir, category, save_size, inp_weight, cpr_weight, global_stats=None, percentile=99.9, combine_mode='average'):
    """Compute the near-max of validation combined scores for a category."""
    inp_val_good = os.path.join(inp_val_dir, category, 'good')
    cpr_val_good = os.path.join(cpr_val_dir, category, 'good')

    inp_npy_files = sorted(glob(os.path.join(inp_val_good, '*_heatmap_raw.npy')))

    all_pixel_scores = []
    for npy_path in inp_npy_files:
        fname = os.path.basename(npy_path).replace('_heatmap_raw.npy', '')
        combined, _ = combine_heatmaps(inp_val_good, cpr_val_good, fname, save_size, inp_weight, cpr_weight, global_stats=global_stats, combine_mode=combine_mode)
        if combined is not None:
            all_pixel_scores.append(combined.flatten())

    if not all_pixel_scores:
        return None

    all_pixel_scores = np.concatenate(all_pixel_scores)
    threshold = np.percentile(all_pixel_scores, percentile)
    print(f'  {category}: val {percentile}th percentile threshold = {threshold:.6f}')
    return threshold
