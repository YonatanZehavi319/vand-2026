"""CPR runner — thin wrapper exposing train/test/validate as callable functions.

The actual logic lives in train_orig.py and test_orig.py (copied from the CPR repo).
This module provides a clean interface for the unified entry points.
"""

from industrial.cpr.train_orig import main as _train_main, get_args_parser as _train_args_parser, ContrastiveLoss, train_one_step
from industrial.cpr.test_orig import main as _test_main, get_args_parser as _test_args_parser, test, validate


def get_train_args_parser():
    return _train_args_parser()


def get_test_args_parser():
    return _test_args_parser()


def train(args):
    """Run CPR training."""
    _train_main(args)


def run_test(args):
    """Run CPR testing."""
    _test_main(args)
