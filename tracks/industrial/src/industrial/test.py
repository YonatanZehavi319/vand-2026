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
    # Ensemble args
    parser.add_argument('--save_size', type=int, default=512)
    parser.add_argument('--inp_weight', type=float, default=1.0)
    parser.add_argument('--cpr_weight', type=float, default=1.0)
    parser.add_argument('--zscore', action='store_true')
    parser.add_argument('--threshold_method', type=str, default='evt', choices=['evt', 'val_max', 'otsu'])
    parser.add_argument('--evt_fdr', type=float, default=0.01)
    parser.add_argument('--val_percentile', type=float, default=99.9)

    args = parser.parse_args()

    inp_heatmap_dir = f'{args.out_dir}/inp_former/heatmaps'
    cpr_heatmap_dir = f'{args.out_dir}/cpr/heatmaps'
    inp_val_dir = f'{args.out_dir}/inp_former/val_heatmaps'
    cpr_val_dir = f'{args.out_dir}/cpr/val_heatmaps'

    if args.model in ('inp', 'both'):
        print(f"\n{'='*20} INP-Former test {'='*20}")
        from industrial.inp_former.runner import parser as inp_parser, _setup_and_run as inp_run
        inp_argv = [
            '--data_path', args.data_dir,
            '--phase', 'test',
            '--save_maps',
            '--save_dir', f'{args.out_dir}/inp_former',
        ]
        if args.item:
            inp_argv += ['--item', args.item]
        inp_test_args = inp_parser.parse_args(inp_argv)
        inp_run(inp_test_args)

    if args.model in ('cpr', 'both'):
        print(f"\n{'='*20} CPR test {'='*20}")
        from industrial.cpr.runner import run_test as cpr_test, get_test_args_parser
        cpr_argv = [
            '--save-maps',
            '--save-dir', f'{args.out_dir}/cpr',
        ]
        if args.item:
            cpr_argv += ['--sub-categories', args.item]
        cpr_test_args = get_test_args_parser().parse_args(cpr_argv)
        cpr_test(cpr_test_args)

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

        run_ensemble(ens)

    print(f"\nResults saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
