"""Generate lighting-augmented copies of validation images.

Creates augmented validation images on disk so both INP-Former and CPR
can run inference on the same augmented set for threshold fitting.

Usage:
    python -m industrial.shared.augment_validation \
        --data_dir /workspace/mvtec \
        --out_dir /workspace/mvtec_val_aug \
        --n_augmented 4 \
        --intensity 0.3
"""

import argparse
import os
import random
from glob import glob

import cv2
import numpy as np

from industrial.shared.synth_lighting import apply_lighting, MODES


def augment_category(src_dir, dst_dir, n_augmented, intensity, max_augs):
    """Generate augmented copies for one category's validation/good images."""
    good_dir = os.path.join(src_dir, 'validation', 'good')
    if not os.path.isdir(good_dir):
        print(f"  Skipping: {good_dir} not found")
        return 0

    out_good = os.path.join(dst_dir, 'validation', 'good')
    os.makedirs(out_good, exist_ok=True)

    images = sorted(
        glob(os.path.join(good_dir, '*.png')) +
        glob(os.path.join(good_dir, '*.JPG')) +
        glob(os.path.join(good_dir, '*.bmp'))
    )

    count = 0
    for img_path in images:
        fname = os.path.splitext(os.path.basename(img_path))[0]
        ext = os.path.splitext(img_path)[1]

        # Copy original
        img_bgr = cv2.imread(img_path)
        cv2.imwrite(os.path.join(out_good, f'{fname}{ext}'), img_bgr)
        count += 1

        # Convert to RGB for augmentation
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Generate augmented copies
        for aug_idx in range(n_augmented):
            augmented = img_rgb.copy()
            n_augs = random.randint(1, max_augs)
            chosen = random.sample(MODES, n_augs)
            for mode in chosen:
                aug_intensity = random.uniform(intensity * 0.5, intensity * 1.5)
                augmented, _ = apply_lighting(augmented, mode=mode, intensity=aug_intensity)
            # Convert back to BGR for saving
            aug_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(out_good, f'{fname}_aug{aug_idx}{ext}'), aug_bgr)
            count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description='Generate augmented validation images')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset root (e.g., /workspace/mvtec)')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory for augmented dataset')
    parser.add_argument('--n_augmented', type=int, default=4, help='Number of augmented copies per image (default 4)')
    parser.add_argument('--intensity', type=float, default=0.3, help='Base augmentation intensity (default 0.3)')
    parser.add_argument('--max_augs', type=int, default=2, help='Max augmentations to stack per image (default 2)')
    parser.add_argument('--item', type=str, default=None, help='Single category to process')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default 42)')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    categories = sorted(os.listdir(args.data_dir))
    categories = [c for c in categories if os.path.isdir(os.path.join(args.data_dir, c, 'validation'))]
    if args.item:
        categories = [c for c in categories if c == args.item]

    print(f"Augmenting validation images: {len(categories)} categories, {args.n_augmented} augmented copies each")

    for category in categories:
        count = augment_category(
            os.path.join(args.data_dir, category),
            os.path.join(args.out_dir, category),
            args.n_augmented,
            args.intensity,
            args.max_augs,
        )
        print(f"  {category}: {count} images saved")

    print(f"\nAugmented validation images saved to {args.out_dir}/")


if __name__ == '__main__':
    main()
