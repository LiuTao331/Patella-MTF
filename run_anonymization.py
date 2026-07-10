#!/usr/bin/env python3
"""Command‑line interface for batch DICOM anonymization of knee CT data."""

import argparse
import sys
from pathlib import Path

from anonymizer import ResearchCTAnonymizer


def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Batch anonymize knee CT DICOM images."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Root directory containing original DICOM files."
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to save anonymized DICOM files."
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="Directory for log files (default: ./logs)."
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="",
        help="Subdirectory under output_dir (default: root of output_dir)."
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Disable recursive search in subdirectories."
    )
    parser.add_argument(
        "--verify-samples",
        type=int,
        default=3,
        help="Number of files to verify after anonymization (default: 3)."
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point for DICOM anonymization.

    Returns
    -------
    int
        Exit code (0 for success).
    """
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir)

    if not input_dir.is_dir():
        print(f"ERROR: Input directory does not exist: {input_dir}", file=sys.stderr)
        return 1

    anonymizer = ResearchCTAnonymizer(
        output_dir=str(output_dir),
        log_dir=str(log_dir)
    )

    success_count = anonymizer.batch_anonymize(
        input_dir=str(input_dir),
        output_subdir=args.output_subdir,
        recursive=not args.no_recursive
    )

    print(f"\nAnonymization completed. {success_count} files successfully processed.")
    print(f"Output: {anonymizer.output_dir}")
    print(f"Logs:   {anonymizer.log_dir}")

    # Verification
    all_anonymized = list(Path(anonymizer.output_dir).rglob("*.dcm"))
    if not all_anonymized:
        print("No anonymized files found to verify.")
        return 0

    sample_files = all_anonymized[: args.verify_samples]
    print(f"\nVerification of {len(sample_files)} randomly selected file(s):")
    for fpath in sample_files:
        ver = anonymizer.verify_complete_anonymization(fpath)
        status = "PASS" if ver["completely_anonymized"] else "FAIL"
        print(f"  [{status}] {fpath.name}")
        if not ver["completely_anonymized"]:
            if ver.get("remaining_demographics"):
                print(f"    Remaining demographics: {ver['remaining_demographics']}")
            if ver.get("remaining_identifiers"):
                print(f"    Remaining identifiers: {ver['remaining_identifiers']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())