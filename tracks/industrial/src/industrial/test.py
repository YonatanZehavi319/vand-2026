"""Inference entrypoint that writes submission-ready industrial predictions.

Runs both models, then the ensemble pipeline to produce final outputs.
"""

import argparse


def main() -> None:
    """Generate predictions for selected categories."""
    parser = argparse.ArgumentParser(description='Run inference and ensemble')
    parser.add_argument('--model', type=str, default='both', choices=['inp', 'cpr', 'both', 'ensemble_only'],
                        help='Which model(s) to run (default: both)')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--out_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--item', type=str, default=None, help='Single category')
    parser.add_argument('--split', type=str, default=None, help='Dataset split: test_public, test_private, test_private_mixed')
    # Ensemble args
    parser.add_argument('--save_size', type=int, default=512)
    parser.add_argument('--inp_weight', type=float, default=1.0)
    parser.add_argument('--cpr_weight', type=float, default=1.0)
    parser.add_argument('--zscore', action='store_true')
    parser.add_argument('--threshold_method', type=str, default='evt', choices=['evt', 'val_max', 'otsu', 'mean_std'])
    parser.add_argument('--evt_fdr', type=float, default=0.01)
    parser.add_argument('--val_percentile', type=float, default=99.9)
    parser.add_argument('--mean_std_k', type=float, default=3.0, help='k for mean+k*std threshold')
    parser.add_argument('--combine_mode', type=str, default='average', choices=['average', 'boost'],
                        help='How to combine heatmaps: average or boost (CPR boosts INP)')
    # INP-Former args
    parser.add_argument('--inp_save_dir', type=str, default=None, help='INP-Former weights dir (default: {out_dir}/inp_former)')
    parser.add_argument('--tiling', action='store_true', help='Use tiling for INP-Former')
    parser.add_argument('--target_tile', type=int, default=1000, help='Target tile size')
    parser.add_argument('--batch_size', type=int, default=16)
    # CPR args
    parser.add_argument('--cpr_save_dir', type=str, default=None, help='CPR output dir (default: {out_dir}/cpr)')
    parser.add_argument('--cpr_checkpoints', type=str, default=None, help='CPR checkpoint pattern (e.g., log/mvtec_train_v2/{category}/03000.pth)')
    # Validation dirs (override auto-detected paths)
    parser.add_argument('--inp_val_dir', type=str, default=None, help='INP-Former validation heatmaps dir')
    parser.add_argument('--cpr_val_dir', type=str, default=None, help='CPR validation heatmaps dir')
    # Smoothing options
    parser.add_argument('--bilateral', action='store_true', help='Apply bilateral filter')
    parser.add_argument('--bilateral_d', type=int, default=9)
    parser.add_argument('--bilateral_sc', type=float, default=75)
    parser.add_argument('--bilateral_ss', type=float, default=75)
    parser.add_argument('--guided', action='store_true', help='Apply guided filter')
    parser.add_argument('--guided_r', type=int, default=8)
    parser.add_argument('--guided_eps', type=float, default=0.01)
    # Adaptive FDR
    parser.add_argument('--adaptive_fdr', action='store_true')
    parser.add_argument('--adaptive_strength', type=float, default=0.3)
    # Per-image normalization
    parser.add_argument('--median_sub', action='store_true', help='Subtract per-image median')
    parser.add_argument('--val_image_dir', type=str, default=None, help='Validation images dir (for guided filter during EVT fitting)')
    # Auto CPR weight
    parser.add_argument('--auto_cpr_weight', action='store_true', help='Auto-compute per-category CPR weight from INP SNR')
    parser.add_argument('--cpr_power', type=float, default=1.0, help='Power for CPR signal in boost mode')
    parser.add_argument('--spatial_prior', action='store_true', help='Apply spatial FP suppression')
    parser.add_argument('--grid_size', type=int, default=4, help='Grid size for spatial prior')
    parser.add_argument('--suppress_floor', type=float, default=0.3, help='Min suppression weight')

    args = parser.parse_args()

    inp_save_dir = args.inp_save_dir or f'{args.out_dir}/inp_former'
    cpr_save_dir = args.cpr_save_dir or f'{args.out_dir}/cpr'

    # Heatmap output goes to out_dir regardless of where weights are
    inp_heatmap_dir = f'{args.out_dir}/inp_former/heatmaps'
    cpr_heatmap_dir = f'{args.out_dir}/cpr/heatmaps'

    # Validation dirs: use explicit args, or look next to weights
    inp_val_dir = args.inp_val_dir or f'{inp_save_dir}/INP-Former-Multi-Class_dataset=MVTec-AD_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6/val_heatmaps'
    cpr_val_dir = args.cpr_val_dir or f'{cpr_save_dir}/val_heatmaps'

    if args.model in ('inp', 'both'):
        print(f"\n{'='*20} INP-Former test {'='*20}")
        from industrial.inp_former.runner import parser as inp_parser, _setup_and_run as inp_run
        inp_argv = [
            '--data_path', args.data_dir,
            '--phase', 'test',
            '--save_maps',
            '--save_dir', inp_save_dir,
            '--batch_size', str(args.batch_size),
        ]
        if args.tiling:
            inp_argv += ['--tiling', '--target_tile', str(args.target_tile)]
        if args.item:
            inp_argv += ['--item', args.item]
        if args.split:
            inp_argv += ['--split', args.split]
        inp_test_args = inp_parser.parse_args(inp_argv)
        inp_run(inp_test_args)

        # Move heatmaps to out_dir if saved elsewhere
        import shutil, os
        src = os.path.join(inp_save_dir, 'INP-Former-Multi-Class_dataset=MVTec-AD_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6', 'heatmaps')
        if os.path.isdir(src) and src != inp_heatmap_dir:
            os.makedirs(inp_heatmap_dir, exist_ok=True)
            for cat in os.listdir(src):
                dst_cat = os.path.join(inp_heatmap_dir, cat)
                if os.path.exists(dst_cat):
                    shutil.rmtree(dst_cat)
                shutil.move(os.path.join(src, cat), dst_cat)

    if args.model in ('cpr', 'both'):
        print(f"\n{'='*20} CPR test {'='*20}")
        from industrial.cpr.runner import run_test as cpr_test, get_test_args_parser
        cpr_argv = [
            '--save-maps',
            '--save-dir', cpr_save_dir,
            '--data-root', args.data_dir,
        ]
        if args.cpr_checkpoints:
            cpr_argv += ['--checkpoints', args.cpr_checkpoints]
        if args.item:
            cpr_argv += ['--sub-categories', args.item]
        if args.split:
            cpr_argv += ['--split', args.split]
        cpr_test_args = get_test_args_parser().parse_args(cpr_argv)
        cpr_test(cpr_test_args)

        # Move heatmaps to out_dir if saved elsewhere
        import os
        src = os.path.join(cpr_save_dir, 'heatmaps')
        if os.path.isdir(src) and src != cpr_heatmap_dir:
            os.makedirs(cpr_heatmap_dir, exist_ok=True)
            for cat in os.listdir(src):
                dst_cat = os.path.join(cpr_heatmap_dir, cat)
                if os.path.exists(dst_cat):
                    shutil.rmtree(dst_cat)
                shutil.move(os.path.join(src, cat), dst_cat)

    # Run ensemble
    if args.model in ('both', 'ensemble_only'):
        print(f"\n{'='*20} Ensemble {'='*20}")
        from industrial.ensemble.pipeline import run_ensemble

        class EnsembleArgs:
            pass

        ens = EnsembleArgs()
        ens.inp_dir = inp_heatmap_dir
        ens.cpr_dir = cpr_heatmap_dir
        ens.data_dir = args.data_dir
        ens.out_dir = args.out_dir
        ens.item = args.item
        ens.save_size = args.save_size
        ens.inp_weight = args.inp_weight
        ens.cpr_weight = args.cpr_weight
        ens.zscore = args.zscore
        ens.inp_val_dir = inp_val_dir
        ens.cpr_val_dir = cpr_val_dir
        ens.threshold_method = args.threshold_method
        ens.evt_fdr = args.evt_fdr
        ens.val_percentile = args.val_percentile
        ens.mean_std_k = args.mean_std_k
        ens.combine_mode = args.combine_mode
        ens.bilateral = args.bilateral
        ens.bilateral_d = args.bilateral_d
        ens.bilateral_sc = args.bilateral_sc
        ens.bilateral_ss = args.bilateral_ss
        ens.guided = args.guided
        ens.guided_r = args.guided_r
        ens.guided_eps = args.guided_eps
        ens.adaptive_fdr = args.adaptive_fdr
        ens.adaptive_strength = args.adaptive_strength
        ens.median_sub = args.median_sub
        ens.val_image_dir = args.val_image_dir
        ens.auto_cpr_weight = args.auto_cpr_weight
        ens.cpr_power = args.cpr_power
        ens.spatial_prior = args.spatial_prior
        ens.grid_size = args.grid_size
        ens.suppress_floor = args.suppress_floor

        run_ensemble(ens)

    print(f"\nResults saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
