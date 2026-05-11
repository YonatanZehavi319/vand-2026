"""Training entrypoint for the industrial track.

Dispatches to INP-Former and/or CPR training based on --model flag.
"""

import argparse
import sys


def main() -> None:
    """Train one or both models per selected categories."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--model', type=str, default='both', choices=['inp', 'cpr', 'both'],
                            help='Which model to train: inp, cpr, or both (default: both)')
    pre_parser.add_argument('--phase', type=str, default='train', choices=['train', 'validation'],
                            help='Phase: train or validation (default: train)')
    pre_args, remaining = pre_parser.parse_known_args()

    if pre_args.model in ('inp', 'both'):
        print(f"\n{'='*20} INP-Former ({pre_args.phase}) {'='*20}")
        from industrial.inp_former.runner import parser as inp_parser, _setup_and_run as inp_run
        inp_args = inp_parser.parse_args(remaining + ['--phase', pre_args.phase])
        inp_run(inp_args)

    if pre_args.model in ('cpr', 'both'):
        print(f"\n{'='*20} CPR ({pre_args.phase}) {'='*20}")
        if pre_args.phase == 'train':
            from industrial.cpr.runner import train as cpr_train, get_train_args_parser
            cpr_args = get_train_args_parser().parse_args(remaining)
            cpr_train(cpr_args)
        elif pre_args.phase == 'validation':
            from industrial.cpr.runner import run_test as cpr_test, get_test_args_parser
            cpr_args = get_test_args_parser().parse_args(remaining + ['--validation'])
            cpr_test(cpr_args)


if __name__ == "__main__":
    main()
