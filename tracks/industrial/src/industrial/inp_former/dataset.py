import random

from torchvision import transforms
from PIL import Image
import os
import torch
import glob
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageFolder
import numpy as np
import torch.multiprocessing
import json

# import imgaug.augmenters as iaa
# from perlin import rand_perlin_2d_np

torch.multiprocessing.set_sharing_strategy('file_system')


class RandomLightingAugmentation:
    """Randomly apply lighting augmentations per image.
    - Directional lighting (left/right/top/bottom)
    - Over/underexposure
    - Color tint and warmth
    Applied with a given probability; otherwise image is unchanged.
    Core transforms imported from synth_lighting.py."""

    def __init__(self, p=0.5, intensity_range=(0.08, 0.2), max_augs=2):
        self.p = p
        self.intensity_range = intensity_range
        self.max_augs = max_augs

    def __call__(self, img):
        if random.random() > self.p:
            return img
        from industrial.shared.synth_lighting import apply_lighting, MODES
        img_np = np.array(img).astype(np.uint8)
        n_augs = random.randint(1, self.max_augs)
        chosen = random.sample(MODES, n_augs)
        for aug in chosen:
            intensity = random.uniform(*self.intensity_range)
            img_np, _ = apply_lighting(img_np, mode=aug, intensity=intensity)
        return Image.fromarray(img_np)


def get_data_transforms(size, isize, mean_train=None, std_train=None, lighting_aug=False, lighting_intensity=(0.08, 0.2), lighting_prob=0.5, lighting_max_augs=2, tiling=False):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    if tiling:
        # Tiling mode: resize directly to crop_size, no CenterCrop.
        # Tile margins already provide border context.
        train_transforms_list = [transforms.Resize((isize, isize))]
        if lighting_aug:
            train_transforms_list.append(RandomLightingAugmentation(p=lighting_prob, intensity_range=lighting_intensity, max_augs=lighting_max_augs))
        train_transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_train, std=std_train)])
        data_transforms = transforms.Compose(train_transforms_list)
        gt_transforms = transforms.Compose([
            transforms.Resize((isize, isize)),
            transforms.ToTensor()])
    else:
        train_transforms_list = [transforms.Resize((size, size))]
        if lighting_aug:
            train_transforms_list.append(RandomLightingAugmentation(p=lighting_prob, intensity_range=lighting_intensity, max_augs=lighting_max_augs))
        train_transforms_list.extend([
            transforms.ToTensor(),
            transforms.CenterCrop(isize),
            transforms.Normalize(mean=mean_train, std=std_train)])
        data_transforms = transforms.Compose(train_transforms_list)
        gt_transforms = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.CenterCrop(isize),
            transforms.ToTensor()])
    return data_transforms, gt_transforms

class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase, split=None):
        """MVTec dataset loader supporting both AD 1 and AD 2 layouts.

        Args:
            root: Category root dir (e.g., /data/mvtec/can)
            phase: 'train' or 'test'
            split: For AD 2, which split to use: 'test_public', 'test_private',
                   'test_private_mixed'. If None, auto-detects layout.
        """
        self.phase = phase
        self.transform = transform
        self.gt_transform = gt_transform

        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            # Auto-detect AD 2 vs AD 1 layout
            if split is not None:
                self.img_path = os.path.join(root, split)
            elif os.path.isdir(os.path.join(root, 'test_public')):
                self.img_path = os.path.join(root, 'test_public')
            else:
                self.img_path = os.path.join(root, 'test')

            # GT location: AD 2 nests it under the split dir
            ad2_gt = os.path.join(self.img_path, 'ground_truth')
            ad1_gt = os.path.join(root, 'ground_truth')
            if os.path.isdir(ad2_gt):
                self.gt_path = ad2_gt
            else:
                self.gt_path = ad1_gt

        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()
        self.cls_idx = 0

    def _glob_images(self, directory):
        return glob.glob(os.path.join(directory, '*.png')) + \
               glob.glob(os.path.join(directory, '*.JPG')) + \
               glob.glob(os.path.join(directory, '*.bmp'))

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        if self.phase == 'train':
            # Train: only good images, may be in train/good/ or flat in train/
            good_dir = os.path.join(self.img_path, 'good')
            if os.path.isdir(good_dir):
                img_paths = sorted(self._glob_images(good_dir))
            else:
                img_paths = sorted(self._glob_images(self.img_path))
            img_tot_paths.extend(img_paths)
            gt_tot_paths.extend([0] * len(img_paths))
            tot_labels.extend([0] * len(img_paths))
            tot_types.extend(['good'] * len(img_paths))
        else:
            # Check if this is a flat split (test_private, test_private_mixed)
            subdirs = [d for d in os.listdir(self.img_path)
                       if os.path.isdir(os.path.join(self.img_path, d)) and d != 'ground_truth']

            if not subdirs:
                # Flat directory — no labels available (private test)
                img_paths = sorted(self._glob_images(self.img_path))
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([-1] * len(img_paths))
                tot_types.extend(['unknown'] * len(img_paths))
            else:
                for defect_type in sorted(subdirs):
                    img_paths = sorted(self._glob_images(os.path.join(self.img_path, defect_type)))
                    if defect_type == 'good':
                        img_tot_paths.extend(img_paths)
                        gt_tot_paths.extend([0] * len(img_paths))
                        tot_labels.extend([0] * len(img_paths))
                        tot_types.extend(['good'] * len(img_paths))
                    else:
                        # defect_type is 'bad' in AD 2, or named types in AD 1
                        gt_paths = sorted(glob.glob(os.path.join(self.gt_path, defect_type, '*.png')))
                        img_tot_paths.extend(img_paths)
                        if len(gt_paths) == len(img_paths):
                            gt_tot_paths.extend(gt_paths)
                        else:
                            # Match GTs by name pattern: {stem}_mask.png
                            for ip in img_paths:
                                stem = os.path.splitext(os.path.basename(ip))[0]
                                gt_file = os.path.join(self.gt_path, defect_type, f'{stem}_mask.png')
                                gt_tot_paths.append(gt_file if os.path.exists(gt_file) else 0)
                        tot_labels.extend([1] * len(img_paths))
                        tot_types.extend([defect_type] * len(img_paths))

        return np.array(img_tot_paths), np.array(gt_tot_paths), np.array(tot_labels), np.array(tot_types)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path


def compute_tile_info(w, h, overlap=0.2, target_tile=1000, margin_ratio=0.1):
    """Compute tile grid info without touching pixel data. Lightweight for init."""
    cols = max(2, round(w / target_tile))
    rows = max(2, round(h / target_tile))
    if h < target_tile * 0.75:
        rows = 1
    if w < target_tile * 0.75:
        cols = 1

    tile_w = int(w / (cols - overlap * (cols - 1))) if cols > 1 else w
    tile_h = int(h / (rows - overlap * (rows - 1))) if rows > 1 else h

    margin_x = int(tile_w * margin_ratio)
    margin_y = int(tile_h * margin_ratio)

    stride_x = int((w - tile_w) / max(1, cols - 1)) if cols > 1 else 0
    stride_y = int((h - tile_h) / max(1, rows - 1)) if rows > 1 else 0

    positions = []
    for r in range(rows):
        for c in range(cols):
            y = min(r * stride_y, h - tile_h)
            x = min(c * stride_x, w - tile_w)
            positions.append((y, x))

    return {'h': h, 'w': w, 'tile_h': tile_h, 'tile_w': tile_w,
            'rows': rows, 'cols': cols, 'positions': positions,
            'n_tiles': rows * cols, 'margin_x': margin_x, 'margin_y': margin_y}


def extract_tiles(img, overlap=0.2, target_tile=1000, margin_ratio=0.1):
    """Extract adaptive grid of overlapping tiles from a PIL Image.
    Grid size adapts to aspect ratio. Mirror-pads edges by margin_ratio of tile size.
    Returns: (list of PIL crops, tile_info dict)."""
    w, h = img.size
    tile_info = compute_tile_info(w, h, overlap, target_tile, margin_ratio)
    tile_h, tile_w = tile_info['tile_h'], tile_info['tile_w']
    margin_x, margin_y = tile_info['margin_x'], tile_info['margin_y']

    # Mirror-pad the image so edge tiles have context
    img_np = np.array(img)
    if img_np.ndim == 2:
        padded = np.pad(img_np, ((margin_y, margin_y), (margin_x, margin_x)), mode='reflect')
    else:
        padded = np.pad(img_np, ((margin_y, margin_y), (margin_x, margin_x), (0, 0)), mode='reflect')
    padded_img = Image.fromarray(padded)

    tiles = []
    for y, x in tile_info['positions']:
        tile = padded_img.crop((x, y, x + tile_w + 2 * margin_x, y + tile_h + 2 * margin_y))
        tiles.append(tile)

    return tiles, tile_info


class TiledImageFolder(torch.utils.data.Dataset):
    """Training dataset that yields individual tiles from each image."""
    def __init__(self, root, transform, overlap=0.5, target_tile=1000):
        self.transform = transform
        self.overlap = overlap
        self.target_tile = target_tile
        # Pre-compute tile counts per image
        self.samples = []
        self.tile_index = []  # (img_idx, tile_idx)
        raw_paths = []
        for class_dir in sorted(os.listdir(root)):
            class_path = os.path.join(root, class_dir)
            if not os.path.isdir(class_path):
                continue
            for ext in ('*.png', '*.JPG', '*.bmp'):
                for img_path in sorted(glob.glob(os.path.join(class_path, ext))):
                    raw_paths.append(img_path)
        # Build index by reading image size to get tile count (no pixel loading)
        for i, path in enumerate(raw_paths):
            img = Image.open(path)
            w, h = img.size
            info = compute_tile_info(w, h, self.overlap, target_tile=self.target_tile)
            self.samples.append(path)
            for t in range(info['n_tiles']):
                self.tile_index.append((i, t))

    def __len__(self):
        return len(self.tile_index)

    def __getitem__(self, idx):
        img_idx, tile_idx = self.tile_index[idx]
        img = Image.open(self.samples[img_idx]).convert('RGB')
        tiles, _ = extract_tiles(img, self.overlap, target_tile=self.target_tile)
        return self.transform(tiles[tile_idx]), 0


class TiledMVTecDataset(torch.utils.data.Dataset):
    """Test dataset that yields tiles with GT tiles and metadata for stitching.
    Supports both AD 1 and AD 2 layouts."""
    def __init__(self, root, transform, gt_transform, phase, overlap=0.5, target_tile=1000, split=None):
        # Auto-detect AD 2 vs AD 1
        if split is not None:
            self.img_path_dir = os.path.join(root, split)
        elif os.path.isdir(os.path.join(root, 'test_public')):
            self.img_path_dir = os.path.join(root, 'test_public')
        else:
            self.img_path_dir = os.path.join(root, 'test')

        ad2_gt = os.path.join(self.img_path_dir, 'ground_truth')
        ad1_gt = os.path.join(root, 'ground_truth')
        self.gt_path_dir = ad2_gt if os.path.isdir(ad2_gt) else ad1_gt

        self.transform = transform
        self.gt_transform = gt_transform
        self.overlap = overlap
        self.target_tile = target_tile
        self.img_paths, self.gt_paths, self.labels, self.types = self._load()
        self.tile_index = []
        for i, path in enumerate(self.img_paths):
            img = Image.open(path)
            w, h = img.size
            info = compute_tile_info(w, h, self.overlap, target_tile=self.target_tile)
            for t in range(info['n_tiles']):
                self.tile_index.append((i, t))

    def _glob_images(self, directory):
        return glob.glob(os.path.join(directory, '*.png')) + \
               glob.glob(os.path.join(directory, '*.JPG')) + \
               glob.glob(os.path.join(directory, '*.bmp'))

    def _load(self):
        img_tot, gt_tot, labels, types = [], [], [], []
        subdirs = [d for d in sorted(os.listdir(self.img_path_dir))
                   if os.path.isdir(os.path.join(self.img_path_dir, d)) and d != 'ground_truth']

        if not subdirs:
            # Flat directory (private test splits)
            imgs = sorted(self._glob_images(self.img_path_dir))
            img_tot.extend(imgs)
            gt_tot.extend([0] * len(imgs))
            labels.extend([-1] * len(imgs))
            types.extend(['unknown'] * len(imgs))
        else:
            for defect_type in subdirs:
                imgs = sorted(self._glob_images(os.path.join(self.img_path_dir, defect_type)))
                if defect_type == 'good':
                    img_tot.extend(imgs)
                    gt_tot.extend([0] * len(imgs))
                    labels.extend([0] * len(imgs))
                    types.extend(['good'] * len(imgs))
                else:
                    gt_dir = os.path.join(self.gt_path_dir, defect_type)
                    for ip in imgs:
                        stem = os.path.splitext(os.path.basename(ip))[0]
                        gt_file = os.path.join(gt_dir, f'{stem}_mask.png')
                        gt_tot.append(gt_file if os.path.exists(gt_file) else 0)
                    img_tot.extend(imgs)
                    labels.extend([1] * len(imgs))
                    types.extend([defect_type] * len(imgs))
        return np.array(img_tot), np.array(gt_tot), np.array(labels), np.array(types)

    def __len__(self):
        return len(self.tile_index)

    def __getitem__(self, idx):
        img_idx, tile_idx = self.tile_index[idx]
        img_path = self.img_paths[img_idx]
        label = self.labels[img_idx]

        img = Image.open(img_path).convert('RGB')
        tiles, tile_info = extract_tiles(img, self.overlap, target_tile=self.target_tile)
        tile_img = self.transform(tiles[tile_idx])

        if label == 0:
            tile_gt = torch.zeros([1, tile_img.size(-2), tile_img.size(-1)])
        else:
            gt = Image.open(self.gt_paths[img_idx]).convert('L')
            gt_tiles, _ = extract_tiles(gt, self.overlap, target_tile=self.target_tile)
            tile_gt = self.gt_transform(gt_tiles[tile_idx])

        # Pack tile_info as individual values for dataloader compatibility
        return (tile_img, tile_gt, label, img_path, tile_idx,
                tile_info['h'], tile_info['w'], tile_info['tile_h'], tile_info['tile_w'],
                tile_info['n_tiles'], str(tile_info['positions']),
                tile_info['margin_x'], tile_info['margin_y'])


class RealIADDataset(torch.utils.data.Dataset):
    def __init__(self, root, category, transform, gt_transform, phase):
        self.img_path = os.path.join(root, 'realiad_1024', category)
        self.transform = transform
        self.gt_transform = gt_transform
        self.phase = phase

        json_path = os.path.join(root, 'realiad_jsons', 'realiad_jsons', category + '.json')
        with open(json_path) as file:
            class_json = file.read()
        class_json = json.loads(class_json)

        self.img_paths, self.gt_paths, self.labels, self.types = [], [], [], []

        data_set = class_json[phase]
        for sample in data_set:
            self.img_paths.append(os.path.join(root, 'realiad_1024', category, sample['image_path']))
            label = sample['anomaly_class'] != 'OK'
            if label:
                self.gt_paths.append(os.path.join(root, 'realiad_1024', category, sample['mask_path']))
            else:
                self.gt_paths.append(None)
            self.labels.append(label)
            self.types.append(sample['anomaly_class'])

        self.img_paths = np.array(self.img_paths)
        self.gt_paths = np.array(self.gt_paths)
        self.labels = np.array(self.labels)
        self.types = np.array(self.types)
        self.cls_idx = 0

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        if self.phase == 'train':
            return img, label

        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path



