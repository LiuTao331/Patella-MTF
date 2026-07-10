#!/usr/bin/env python3
"""Preprocessing pipeline for knee CT data.

Converts DICOM series to NIfTI, resamples to isotropic spacing, and applies
a bone window with intensity normalisation to [0, 1].

Requirements: SimpleITK, numpy, tqdm
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


def convert_dicom_to_nifti(
    input_dir: Path,
    output_dir: Path
) -> int:
    """Convert DICOM series folders to compressed NIfTI files.

    Each subdirectory under `input_dir` is treated as a separate DICOM series.
    Output files are named after the subdirectory.

    Parameters
    ----------
    input_dir : Path
        Directory containing per‑patient/series subfolders with DICOM files.
    output_dir : Path
        Directory where the resulting .nii.gz files will be saved.

    Returns
    -------
    int
        Number of series successfully converted.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    series_folders = sorted(
        [d for d in input_dir.iterdir() if d.is_dir()]
    )

    converted = 0
    for folder in tqdm(series_folders, desc="DICOM to NIfTI"):
        try:
            reader = sitk.ImageSeriesReader()
            dicom_files = reader.GetGDCMSeriesFileNames(str(folder))
            if not dicom_files:
                tqdm.write(f"No DICOM files in {folder}")
                continue
            reader.SetFileNames(dicom_files)
            image = reader.Execute()

            out_path = output_dir / f"{folder.name}.nii.gz"
            sitk.WriteImage(image, str(out_path))
            converted += 1
        except Exception as exc:
            tqdm.write(f"Error converting {folder}: {exc}")

    print(f"Converted {converted} series to NIfTI.")
    return converted


def resample_isotropic(
    image: sitk.Image,
    new_spacing: Tuple[float, float, float] = (0.5, 0.5, 0.5)
) -> sitk.Image:
    """Resample a CT image to isotropic spacing.

    Uses linear interpolation and sets the default background value to
    -1000 HU (air) which is appropriate for CT.

    Parameters
    ----------
    image : sitk.Image
        Input image.
    new_spacing : tuple of float, optional
        Desired voxel spacing in mm, by default (0.5, 0.5, 0.5).

    Returns
    -------
    sitk.Image
        Resampled image.
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [
        int(round(original_size[i] * original_spacing[i] / new_spacing[i]))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1000)  # air HU for CT background

    return resampler.Execute(image)


def apply_bone_window(
    image: sitk.Image,
    center: float = 400,
    width: float = 1500
) -> sitk.Image:
    """Apply a bone window and normalise to [0, 1].

    Parameters
    ----------
    image : sitk.Image
        Input CT image (in Hounsfield Units).
    center : float, optional
        Window center in HU, by default 400.
    width : float, optional
        Window width in HU, by default 1500.

    Returns
    -------
    sitk.Image
        Windowed and normalised image with intensity range [0, 1].
    """
    arr = sitk.GetArrayFromImage(image)
    window_min = center - width / 2
    window_max = center + width / 2
    arr = np.clip(arr, window_min, window_max)
    arr = (arr - window_min) / (window_max - window_min)
    # Values are already in [0, 1] after the rescaling; explicit clip is redundant
    # but kept as a safety measure.
    arr = np.clip(arr, 0.0, 1.0)

    out_img = sitk.GetImageFromArray(arr)
    out_img.CopyInformation(image)
    return out_img


def preprocess_batch(
    input_dir: Path,
    output_dir: Path,
    new_spacing: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    bone_window: bool = True,
    center: float = 400,
    width: float = 1500
) -> int:
    """Resample and optionally apply a bone window to all NIfTI files.

    Parameters
    ----------
    input_dir : Path
        Directory containing .nii or .nii.gz files.
    output_dir : Path
        Directory where processed files will be saved.
    new_spacing : tuple of float, optional
        Target isotropic spacing in mm, by default (0.5, 0.5, 0.5).
    bone_window : bool, optional
        If True, apply the bone window and normalise, by default True.
    center : float, optional
        Bone window center (HU), by default 400.
    width : float, optional
        Bone window width (HU), by default 1500.

    Returns
    -------
    int
        Number of files successfully processed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all NIfTI files (compressed or uncompressed)
    nifti_files = sorted(
        list(input_dir.glob("*.nii")) + list(input_dir.glob("*.nii.gz"))
    )
    # Ensure no duplicates if both .nii and .nii.gz of the same stem exist
    unique_stems = {}
    for fpath in nifti_files:
        # Use the name without .nii or .nii.gz as key, prefer .nii.gz
        stem = fpath.name.replace(".nii.gz", "").replace(".nii", "")
        if stem not in unique_stems or fpath.suffix == ".gz":
            unique_stems[stem] = fpath
    nifti_files = sorted(unique_stems.values())

    processed = 0
    for fpath in tqdm(nifti_files, desc="Preprocessing"):
        try:
            img = sitk.ReadImage(str(fpath))
            img = resample_isotropic(img, new_spacing)
            if bone_window:
                img = apply_bone_window(img, center, width)
            out_path = output_dir / fpath.name
            sitk.WriteImage(img, str(out_path))
            processed += 1
        except Exception as exc:
            tqdm.write(f"Error processing {fpath}: {exc}")

    print(f"Preprocessed {processed}/{len(nifti_files)} files.")
    return processed


def main() -> int:
    """Main entry point for the preprocessing pipeline.

    Returns
    -------
    int
        Exit code (0 on success).
    """
    parser = argparse.ArgumentParser(
        description="Convert DICOM to preprocessed NIfTI for patella CT."
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True,
        help="Available subcommands"
    )

    # Subcommand: dicom2nifti
    parser_conv = subparsers.add_parser(
        "dicom2nifti", help="Convert DICOM series to NIfTI"
    )
    parser_conv.add_argument(
        "input_dir", type=Path,
        help="Directory containing DICOM series subfolders"
    )
    parser_conv.add_argument(
        "output_dir", type=Path,
        help="Directory to save NIfTI (.nii.gz) files"
    )

    # Subcommand: preprocess
    parser_prep = subparsers.add_parser(
        "preprocess", help="Resample and apply bone window"
    )
    parser_prep.add_argument(
        "input_dir", type=Path,
        help="Directory containing NIfTI (.nii/.nii.gz) files"
    )
    parser_prep.add_argument(
        "output_dir", type=Path,
        help="Directory to save preprocessed NIfTI files"
    )
    parser_prep.add_argument(
        "--spacing", nargs=3, type=float, default=[0.5, 0.5, 0.5],
        help="Isotropic spacing in mm (default: 0.5 0.5 0.5)"
    )
    parser_prep.add_argument(
        "--no-bone-window", action="store_true",
        help="Disable bone windowing"
    )
    parser_prep.add_argument(
        "--center", type=float, default=400,
        help="Bone window center (HU), default 400"
    )
    parser_prep.add_argument(
        "--width", type=float, default=1500,
        help="Bone window width (HU), default 1500"
    )

    args = parser.parse_args()

    if args.command == "dicom2nifti":
        convert_dicom_to_nifti(args.input_dir, args.output_dir)
    elif args.command == "preprocess":
        preprocess_batch(
            args.input_dir, args.output_dir,
            new_spacing=tuple(args.spacing),
            bone_window=not args.no_bone_window,
            center=args.center, width=args.width
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())