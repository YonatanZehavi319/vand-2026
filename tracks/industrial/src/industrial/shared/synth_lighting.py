"""Synthetic lighting augmentation.

Core functions for all lighting transforms (RGB numpy arrays).
Used by both the training pipeline (dataset.py) and standalone CLI.

Usage:
    python synth_lighting.py --image path/to/image.png --mode left
    python synth_lighting.py --image path/to/image.png --mode overexpose --intensity 0.7
    python synth_lighting.py --image path/to/image.png --mode all
"""

import numpy as np
import cv2
import argparse
import os
import random


DIRECTIONS = ['left', 'right', 'top', 'bottom']
MODES = DIRECTIONS + ['overexpose', 'underexpose', 'tint', 'warmth']


def make_light_gradient(h, w, direction):
    """Create a gradient map (0 to 1) for a given light direction."""
    if direction == 'left':
        grad = np.linspace(1, 0, w)[None, :].repeat(h, axis=0)
    elif direction == 'right':
        grad = np.linspace(0, 1, w)[None, :].repeat(h, axis=0)
    elif direction == 'top':
        grad = np.linspace(1, 0, h)[:, None].repeat(w, axis=1)
    elif direction == 'bottom':
        grad = np.linspace(0, 1, h)[:, None].repeat(w, axis=1)
    else:
        raise ValueError(f"Unknown direction: {direction}")
    return grad.astype(np.float32)


def apply_directional_lighting(image, direction='left', intensity=0.5, ambient=0.6):
    """Apply directional lighting gradient.

    Args:
        image: numpy array (H, W, 3), uint8, RGB
        direction: 'left', 'right', 'top', 'bottom', or 'random'
        intensity: strength of effect (0-1)
        ambient: min brightness (0-1)
    """
    if direction == 'random':
        direction = random.choice(DIRECTIONS)

    h, w = image.shape[:2]
    grad = make_light_gradient(h, w, direction)
    light_map = ambient + grad * (1.0 - ambient)
    light_map = 1.0 - intensity * (1.0 - light_map)

    result = image.astype(np.float32) * light_map[:, :, None]
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_overexposure(image, intensity=0.5):
    """Brighten using gamma correction (preserves contrast)."""
    gamma = 1.0 - intensity * 0.5
    result = 255.0 * (image.astype(np.float32) / 255.0) ** gamma
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_underexposure(image, intensity=0.5):
    """Darken using gamma correction."""
    gamma = 1.0 + intensity * 0.8
    result = 255.0 * (image.astype(np.float32) / 255.0) ** gamma
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_tint(image, intensity=0.5):
    """Apply a random color tint (RGB)."""
    tint = np.array([random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]) * intensity * 20
    result = image.astype(np.float32) + tint[None, None, :]
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_warmth(image, intensity=0.5):
    """Shift color temperature. Expects RGB (channel 0=R, 2=B)."""
    warm = intensity * 12
    sign = random.choice([-1, 1])
    result = image.astype(np.float32)
    result[:, :, 0] += sign * warm       # R
    result[:, :, 2] -= sign * warm * 0.5  # B
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_lighting(image, mode, intensity=0.5, ambient=0.6):
    """Apply a lighting effect.

    Args:
        image: numpy array (H, W, 3), uint8, RGB
        mode: 'left', 'right', 'top', 'bottom', 'overexpose', 'underexpose',
              'tint', 'warmth', 'random'
        intensity: strength of effect (0-1)
        ambient: min brightness for directional modes (0-1)

    Returns:
        (result, mode_used): numpy array (H, W, 3) uint8, and the mode string
    """
    if mode == 'random':
        mode = random.choice(MODES)

    if mode in DIRECTIONS:
        return apply_directional_lighting(image, direction=mode, intensity=intensity, ambient=ambient), mode
    elif mode == 'overexpose':
        return apply_overexposure(image, intensity=intensity), mode
    elif mode == 'underexpose':
        return apply_underexposure(image, intensity=intensity), mode
    elif mode == 'tint':
        return apply_tint(image, intensity=intensity), mode
    elif mode == 'warmth':
        return apply_warmth(image, intensity=intensity), mode
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--mode', type=str, nargs='+', default=['all'],
                        help='One or more modes: left right top bottom overexpose underexpose tint warmth random all')
    parser.add_argument('--intensity', type=float, default=0.5, help='Effect strength (0-1)')
    parser.add_argument('--ambient', type=float, default=0.6, help='Min brightness for directional (0-1)')
    parser.add_argument('--output_dir', type=str, default='./lighting_demo')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    # OpenCV reads BGR, convert to RGB for processing
    image_bgr = cv2.imread(args.image)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    basename = os.path.splitext(os.path.basename(args.image))[0]

    cv2.imwrite(os.path.join(args.output_dir, f'{basename}_original.png'), image_bgr)

    modes = MODES if 'all' in args.mode else args.mode
    for m in modes:
        result, used_mode = apply_lighting(image, mode=m, intensity=args.intensity, ambient=args.ambient)
        # Convert back to BGR for saving
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(args.output_dir, f'{basename}_{used_mode}.png'), result_bgr)
        print(f"  {basename}_{used_mode}.png")

    print(f"Saved to {args.output_dir}/")
