"""Submission packaging entrypoint for industrial predictions."""

import argparse


def main() -> None:
    """Validate predictions and create a tar.gz submission archive."""
    parser = argparse.ArgumentParser(description='Package submission')
    parser.add_argument('--submission_dir', type=str, required=True, help='Root directory with predictions')
    parser.add_argument('--output', type=str, default=None, help='Output archive path (default: {submission_dir}.tar.gz)')
    args = parser.parse_args()

    from industrial.submission import validate_submission, prepare_submission

    print("Validating submission...")
    try:
        validate_submission(args.submission_dir)
        print("Validation passed!")
    except (FileNotFoundError, ValueError) as e:
        print(f"Validation failed: {e}")
        return

    archive = prepare_submission(args.submission_dir, args.output)
    print(f"Submission archive created: {archive}")


if __name__ == "__main__":
    main()
