#!/usr/bin/env python3
"""
Figure 4h – Cross‑modal Average Absolute Spearman Correlation Heatmap.

Computes the average absolute Spearman correlation between pairs of the four
feature sets (Conventional, Radiomics, 2D CNN, 3D CNN) and draws a heatmap
showing the low redundancy among modalities.

For efficiency, correlation is estimated on random subsets of samples
(max_samples = 500) and features (max_dims = 50). Results are reproducible
when a fixed random seed is provided.

Input data (expected in --data_dir):
    X_cli_for_r.csv  – conventional morphometric features
    X_rad_for_r.csv  – radiomic features
    X_2d_for_r.csv   – 2D CNN features (e.g. PCA‑reduced)
    X_3d_for_r.csv   – 3D CNN features (e.g. PCA‑reduced)

Usage:
    python fig4_cross_modal_correlation.py \
        --data_dir /path/to/data \
        --output_dir /path/to/output \
        --seed 42

Requirements: numpy, pandas, matplotlib, seaborn, scipy
"""

import argparse
import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global style (Nature Communications compatible)
# ---------------------------------------------------------------------------
NC_RED = '#E64B35'
NC_BLUE = '#4DBBD5'
NC_GREEN = '#00A087'

TICK_SIZE = 2.835  # points

def set_style():
    """Apply consistent matplotlib style. Uses Arial if available, else sans-serif."""
    font_family = 'sans-serif'
    try:
        if any('Arial' in f.name for f in plt.matplotlib.font_manager.fontManager.ttflist):
            font_family = 'Arial'
    except Exception:
        pass

    rcParams.update({
        'font.family': font_family,
        'font.size': 14,
        'axes.labelsize': 14,
        'axes.titlesize': 15,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 12,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
        'axes.linewidth': 1.0,
        'axes.edgecolor': 'black',
        'axes.spines.top': True,
        'axes.spines.right': True,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.major.size': TICK_SIZE,
        'ytick.major.size': TICK_SIZE,
        'axes.grid': False,
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
    })


# ---------------------------------------------------------------------------
# Data loading with encoding auto‑detection
# ---------------------------------------------------------------------------
def read_csv_auto(filepath):
    """Read CSV with fallback encodings."""
    encodings = ['utf-8-sig', 'gbk', 'gb2312', 'latin1', 'utf-8']
    for enc in encodings:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            logger.info(f"Read {filepath} with encoding {enc}")
            return df
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"Unable to read {filepath} with any encoding: {encodings}")


# ---------------------------------------------------------------------------
# Correlation computation (reproducible with a given seed)
# ---------------------------------------------------------------------------
def avg_abs_correlation(A, B, max_samples=500, max_dims=50, seed=42):
    """
    Compute the average absolute Spearman correlation between two feature sets.
    Uses random subsets if dimensions/samples exceed thresholds for speed,
    controlled by a fixed seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    n = min(max_samples, A.shape[0])
    idx = rng.choice(A.shape[0], n, replace=False)

    if A.shape[1] > max_dims:
        idx_a = rng.choice(A.shape[1], max_dims, replace=False)
        A_sub = A[np.ix_(idx, idx_a)]
    else:
        A_sub = A[idx]
    if B.shape[1] > max_dims:
        idx_b = rng.choice(B.shape[1], max_dims, replace=False)
        B_sub = B[np.ix_(idx, idx_b)]
    else:
        B_sub = B[idx]

    # Check for NaN values in the subset
    if np.any(np.isnan(A_sub)) or np.any(np.isnan(B_sub)):
        logger.warning("NaN values detected in feature subsets; they will be skipped during correlation.")

    corr_sum = 0.0
    count = 0
    for i in range(A_sub.shape[1]):
        for j in range(B_sub.shape[1]):
            corr, _ = spearmanr(A_sub[:, i], B_sub[:, j])
            if not np.isnan(corr):
                corr_sum += abs(corr)
                count += 1
    return corr_sum / count if count > 0 else 0.0


def plot_cross_modal_correlation(X_cli, X_rad, X_2d, X_3d, output_dir,
                                 seed, save_base='figure_cross_modal_correlation'):
    """Generate and save the correlation heatmap."""
    corr_cli_rad = avg_abs_correlation(X_cli, X_rad, seed=seed)
    corr_cli_2d  = avg_abs_correlation(X_cli, X_2d, seed=seed)
    corr_cli_3d  = avg_abs_correlation(X_cli, X_3d, seed=seed)
    corr_rad_2d  = avg_abs_correlation(X_rad, X_2d, seed=seed)
    corr_rad_3d  = avg_abs_correlation(X_rad, X_3d, seed=seed)
    corr_2d_3d   = avg_abs_correlation(X_2d, X_3d, seed=seed)

    modalities = ['Conventional', 'Radiomics', '2D CNN', '3D CNN']
    corr_mat = pd.DataFrame([
        [1.0, corr_cli_rad, corr_cli_2d, corr_cli_3d],
        [corr_cli_rad, 1.0, corr_rad_2d, corr_rad_3d],
        [corr_cli_2d, corr_rad_2d, 1.0, corr_2d_3d],
        [corr_cli_3d, corr_rad_3d, corr_2d_3d, 1.0]
    ], index=modalities, columns=modalities)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr_mat, annot=True, fmt='.3f', cmap='RdBu_r', vmin=0.0, vmax=1.0,
                square=True, linewidths=0.5, cbar_kws={'label': '|Spearman r|', 'shrink': 0.8},
                ax=ax)
    ax.set_title('Cross-Modal Average Absolute Correlation', fontweight='bold', pad=15)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('|Spearman r|', fontsize=12)

    for ext in ['pdf', 'png']:
        path = os.path.join(output_dir, f'{save_base}.{ext}')
        fig.savefig(path, format=ext, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"Saved {save_base} (PDF + PNG)")
    return corr_mat


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross‑modal average absolute Spearman correlation heatmap."
    )
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing X_cli_for_r.csv, X_rad_for_r.csv, X_2d_for_r.csv, X_3d_for_r.csv')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the figure')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    set_style()

    # -------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------
    data_dir = args.data_dir
    required_files = ['X_cli_for_r.csv', 'X_rad_for_r.csv', 'X_2d_for_r.csv', 'X_3d_for_r.csv']
    for fname in required_files:
        if not os.path.isfile(os.path.join(data_dir, fname)):
            logger.error(f"Missing file: {fname} in {data_dir}")
            sys.exit(1)

    try:
        X_cli = read_csv_auto(os.path.join(data_dir, 'X_cli_for_r.csv')).values.astype(np.float64)
        X_rad = read_csv_auto(os.path.join(data_dir, 'X_rad_for_r.csv')).values.astype(np.float64)
        X_2d  = read_csv_auto(os.path.join(data_dir, 'X_2d_for_r.csv')).values.astype(np.float64)
        X_3d  = read_csv_auto(os.path.join(data_dir, 'X_3d_for_r.csv')).values.astype(np.float64)
    except Exception as e:
        logger.error(f"Failed to read CSV files: {e}")
        sys.exit(1)

    # Check that all feature matrices have the same number of rows
    n_samples = X_cli.shape[0]
    if not (X_rad.shape[0] == n_samples and X_2d.shape[0] == n_samples and X_3d.shape[0] == n_samples):
        logger.error("Number of samples inconsistent across feature files.")
        sys.exit(1)

    # Check for NaN values in each modality
    for name, mat in zip(['Conventional', 'Radiomics', '2D CNN', '3D CNN'],
                         [X_cli, X_rad, X_2d, X_3d]):
        nan_count = np.isnan(mat).sum()
        if nan_count > 0:
            logger.warning(f"{name} contains {nan_count} NaN values. Correlation computation may skip those entries.")

    logger.info(f"Loaded {n_samples} samples.")
    logger.info(f"Conventional: {X_cli.shape[1]} features")
    logger.info(f"Radiomics:    {X_rad.shape[1]} features")
    logger.info(f"2D CNN:       {X_2d.shape[1]} features")
    logger.info(f"3D CNN:       {X_3d.shape[1]} features")

    # -------------------------------------------------------------------
    # Compute and plot
    # -------------------------------------------------------------------
    logger.info("Computing cross‑modal correlations (seed=%d)...", args.seed)
    corr_matrix = plot_cross_modal_correlation(X_cli, X_rad, X_2d, X_3d,
                                               args.output_dir, args.seed,
                                               save_base='figure_cross_modal_correlation')

    # Optional textual summary
    print("\nCross‑Modal Average Absolute Spearman Correlation Matrix:")
    print(corr_matrix)
    off_diag = corr_matrix.values[np.triu_indices(4, k=1)]
    mean_off_diag = np.mean(off_diag)
    logger.info(f"Mean off‑diagonal correlation: {mean_off_diag:.4f}")
    if mean_off_diag < 0.3:
        logger.info("Low redundancy → complementary information across modalities.")
    else:
        logger.info("Moderate redundancy → fusion may still be beneficial.")

    logger.info(f"Figure saved to {args.output_dir}")


if __name__ == "__main__":
    main()