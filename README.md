# VAND 2026 Industrial Track — Gated Boost Ensemble

Anomaly detection and segmentation on MVTec AD 2, combining **INP-Former** and **CPR** with a gated boost ensemble strategy.

**Team:** Yonatan Zehavi (Hebrew University of Jerusalem), Niv Cohen (New York University)

## Method Summary

- **INP-Former**: Reconstruction-based anomaly detection with DINOv2 encoder and Intrinsic Normal Prototypes, using overlapping tiling for high-resolution images.
- **CPR**: Cascade Patch Retrieval with DenseNet201 backbone for fine-grained patch-level anomaly localization.
- **Ensemble**: Gated boost — CPR selectively amplifies INP-Former's signal only where CPR is confident (top 5% pixels). Per-category CPR weight is automatically derived from edge correlation between models on validation data.
- **Threshold**: Extreme Value Theory (GEV) fitted on combined validation heatmaps.
- **Post-processing**: Guided filter using original image as reference for edge-aware smoothing.

## Setup

```bash
# Clone and install
git clone https://github.com/YonatanZehavi319/vand-2026.git
cd vand-2026
pip install -e tracks/industrial/ -e utils/

# Additional dependencies
pip install matplotlib numpy==1.26.4 opencv-contrib-python
```

## Download Weights

Download from [GitHub Releases](https://github.com/YonatanZehavi319/vand-2026/releases/tag/v1.0):

```bash
# INP-Former weights (split into 3 parts)
cat inp_weights_25ep_part_aa inp_weights_25ep_part_ab inp_weights_25ep_part_ac > inp_weights.tar.gz
mkdir -p weights/INP-Former-Multi-Class_dataset=MVTec-AD_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6
tar xzf inp_weights.tar.gz -C weights/INP-Former-Multi-Class_dataset=MVTec-AD_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6/

# CPR weights
tar xzf cpr_weights.tar.gz -C log/mvtec_train_v2/
```

## Dataset

**MVTec AD 2**: Download from [mvtec.com](https://www.mvtec.com/company/research/datasets/mvtec-ad-2) and extract to `/workspace/mvtec/`. The directory should contain category folders (`can/`, `fabric/`, etc.) each with `train/`, `validation/`, `test_public/`, `test_private/`, `test_private_mixed/`.

**DTD (Describable Textures Dataset)**: Required for CPR synthetic data generation. Download from [robots.ox.ac.uk/~vgg/data/dtd](https://www.robots.ox.ac.uk/~vgg/data/dtd/) and extract to `/workspace/dtd/`.

CPR preprocessing generates these from the above datasets:
- Foreground masks: `log/foreground/`
- Retrieval index: `log/retrieval_mvtec_DenseNet_features.denseblock1_320/`
- Synthetic data: `log/synthetic_mvtec_320_6000_True_jpg/` (uses DTD textures + Perlin noise)

These are generated during CPR preprocessing (see Training section).

## Reproduce Submission

### 1. Run Validation (for threshold fitting)

```bash
# INP-Former validation (non-augmented)
python -m industrial.train --model inp --phase validation \
    --data_path /workspace/mvtec --save_dir weights \
    --tiling --target_tile 600 --batch_size 4

# CPR validation
python -m industrial.train --model cpr --phase validation \
    --data-root /workspace/mvtec --save-dir ./output/cpr \
    --checkpoints "log/mvtec_train_v2/{category}/03000.pth"
```

### 2. Run Inference on Private Test Splits

All best settings are defaults, so the commands are simple:

```bash
python -m industrial.test --data_dir /workspace/mvtec --out_dir ./output_submit --split test_private
python -m industrial.test --data_dir /workspace/mvtec --out_dir ./output_submit --split test_private_mixed
```

<details>
<summary>Full explicit command (equivalent)</summary>

```bash
python -m industrial.test \
    --data_dir /workspace/mvtec \
    --out_dir ./output_submit \
    --split test_private \
    --inp_save_dir weights \
    --tiling --target_tile 600 --batch_size 4 \
    --cpr_checkpoints "log/mvtec_train_v2/{category}/03000.pth" \
    --combine_mode gated_boost \
    --auto_cpr_weight \
    --guided --guided_eps 0.001 \
    --evt_fdr 0.05 --cpr_power 1.5 \
    --inp_val_dir "weights/INP-Former-Multi-Class_dataset=MVTec-AD_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6/val_heatmaps" \
    --cpr_val_dir output/cpr/val_heatmaps
```
</details>
```

### 3. Package Submission

```bash
tar czf submission.tar.gz -C output_submit anomaly_images anomaly_images_thresholded
```

Upload `submission.tar.gz` to [benchmark.mvtec.com](https://benchmark.mvtec.com/).

### 4. Evaluate on test_public (optional)

```bash
python -m industrial.shared.seg_f1 ./output /workspace/mvtec
```

## Training from Scratch

### INP-Former

```bash
python -m industrial.train --model inp --phase train \
    --data_path /workspace/mvtec --save_dir weights \
    --tiling --target_tile 600 --total_epochs 25 \
    --lighting_aug --lighting_min 0.02 --lighting_max 0.12
```

### CPR

```bash
# 1. Preprocessing (foreground, retrieval, synthetic data)
python -m industrial.cpr.tools.generate_foreground -fd log/foreground --data-root /workspace/mvtec
python -m industrial.cpr.tools.generate_retrieval --data-root /workspace/mvtec
python -m industrial.cpr.tools.generate_synthetic_data -fd log/foreground --data-root /workspace/mvtec --dtd-dir /workspace/dtd/images

# 2. Training
python -m industrial.train --model cpr --phase train \
    -lp log/mvtec_train_v2 \
    --data-dir log/synthetic_mvtec_320_6000_True_jpg \
    -fd log/foreground \
    -rd log/retrieval_mvtec_DenseNet_features.denseblock1_320 \
    --data-root /workspace/mvtec \
    --steps 3000 -tps 1000 \
    --lighting-prob 0.7 --lighting-min 0.02 --lighting-max 0.12
```

## Ensemble Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `combine_mode` | `gated_boost` | CPR boosts INP only where CPR is confident |
| `auto_cpr_weight` | enabled | Per-category weight from edge correlation |
| `guided_eps` | 0.001 | Guided filter epsilon |
| `evt_fdr` | 0.05 | EVT false discovery rate |
| `cpr_power` | 1.5 | Power applied to CPR signal |
| `tiling` | 600 | Target tile size for INP-Former |

## Results (test_public)

| Category | SegF1 | Precision | Recall |
|----------|-------|-----------|--------|
| can | 0.003 | 0.001 | 0.101 |
| fabric | 0.237 | 0.339 | 0.182 |
| fruit_jelly | 0.628 | 0.812 | 0.511 |
| rice | 0.662 | 0.646 | 0.679 |
| sheet_metal | 0.452 | 0.654 | 0.345 |
| vial | 0.386 | 0.609 | 0.283 |
| wallplugs | 0.142 | 0.101 | 0.237 |
| walnuts | 0.665 | 0.639 | 0.692 |
| **Mean** | **0.397** | **0.475** | **0.379** |

## License

CC-BY-NC-4.0
