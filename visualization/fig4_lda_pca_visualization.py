#!/usr/bin/env python3
"""
Figure 4e–g – LDA + PCA Projections of Radiomic, 2D CNN, and 3D CNN Features.

This script computes the 2‑dimensional LDA + orthogonal PCA embedding for each
feature set, calculates sex‑separation metrics (Silhouette, Davies‑Bouldin index,
Cohen's d), and produces both individual and combined scatter plots.

Input data (expected in --data_dir):
    X_rad_train_selected.npy   – radiomic features (n_samples × n_features)
    X_2d_train_norm.npy        – 2D CNN features
    X_3d_train_norm.npy        – 3D CNN features
    y_train.npy                – integer labels (0 = Female, 1 = Male)

Usage:
    python fig4_lda_pca_visualization.py \
        --data_dir /path/to/data \
        --output_dir /path/to/output \
        [--seed 42]

Requirements: numpy, matplotlib, scikit-learn
"""

import argparse
import sys
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import silhouette_score, davies_bouldin_score

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global style (Nature Communications compatible)
# ---------------------------------------------------------------------------
NC_RED = '#E64B35'
NC_BLUE = '#4DBBD5'

TICK_SIZE = 2.835  # points

def set_style():
    """Apply consistent matplotlib style. Uses Arial if available, else sans-serif."""
    font_family = 'sans-serif'
    try:
        # Simple check for Arial font availability
        if any('Arial' in f.name for f in plt.matplotlib.font_manager.fontManager.ttflist):
            font_family = 'Arial'
    except Exception:
        pass

    plt.rcParams.update({
        'font.family': font_family,
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
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
# LDA + PCA computation (corrected Cohen's d)
# ---------------------------------------------------------------------------
def compute_lda_pca(X, y):
    """
    Returns (embedding, metrics_dict)
    embedding: (n_samples, 2)  (LDA axis, orthogonal PCA axis)
    metrics : silhouette, davies_bouldin, lda_mean_diff, cohens_d,
              male_mean, female_mean
    """
    lda = LDA(n_components=1)
    X_lda = lda.fit_transform(X, y)

    # Orthogonal subspace via residual after projecting onto LDA direction
    w = lda.coef_[0]
    proj = np.dot(X, w) / np.dot(w, w)
    X_orth = X - np.outer(proj, w)

    pca = PCA(n_components=1)
    X_orth_pca = pca.fit_transform(X_orth)

    emb = np.hstack([X_lda, X_orth_pca])

    sil = silhouette_score(emb, y)
    db = davies_bouldin_score(emb, y)

    lda_scores = X_lda.ravel()
    male = lda_scores[y == 1]
    female = lda_scores[y == 0]

    # Cohen's d with proper pooled standard deviation
    n_m, n_f = len(male), len(female)
    var_m = np.var(male, ddof=1)
    var_f = np.var(female, ddof=1)
    pooled_std = np.sqrt(((n_m - 1) * var_m + (n_f - 1) * var_f) / (n_m + n_f - 2))

    mean_m = np.mean(male)
    mean_f = np.mean(female)
    mean_diff = mean_m - mean_f
    cohens_d = mean_diff / pooled_std if pooled_std != 0 else np.nan

    return emb, {
        'silhouette': sil,
        'davies_bouldin': db,
        'lda_mean_diff': mean_diff,
        'cohens_d': cohens_d,
        'lda_male_mean': mean_m,
        'lda_female_mean': mean_f,
    }


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------
def plot_lda_pca(ax, emb, y, title, sil=None, db=None, sub_label=None):
    """Draw LDA+PCA scatter on a given axes."""
    ax.scatter(emb[y == 0, 0], emb[y == 0, 1],
               c=NC_RED, label='Female', alpha=0.6, s=20, edgecolors='none')
    ax.scatter(emb[y == 1, 0], emb[y == 1, 1],
               c=NC_BLUE, label='Male', alpha=0.6, s=20, edgecolors='none')
    ax.set_title(title, fontweight='bold', pad=12)
    ax.set_xlabel('LDA Discriminant Axis')
    ax.set_ylabel('Orthogonal PCA Axis')
    ax.legend(frameon=False, loc='upper right', fontsize=10)

    if sub_label:
        ax.text(-0.12, 1.02, sub_label, transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='bottom', ha='right')
    if sil is not None and db is not None:
        ax.text(0.05, 0.95, f'Sil: {sil:.3f}\nDBI: {db:.3f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)


def save_figure(fig, base_name, output_dir):
    """Save figure as PDF and PNG."""
    for ext in ['pdf', 'png']:
        path = os.path.join(output_dir, f'{base_name}.{ext}')
        fig.savefig(path, format=ext, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"Saved {base_name} (PDF + PNG)")


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="LDA + PCA projection of radiomic, 2D CNN and 3D CNN features."
    )
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing .npy files (X_rad_train_selected.npy, ...)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save figures')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (not strictly needed here)')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    set_style()

    # -------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------
    data_dir = args.data_dir
    required_files = [
        'X_rad_train_selected.npy',
        'X_2d_train_norm.npy',
        'X_3d_train_norm.npy',
        'y_train.npy',
    ]
    for fname in required_files:
        if not os.path.isfile(os.path.join(data_dir, fname)):
            logger.error(f"Missing file: {fname} in {data_dir}")
            sys.exit(1)

    try:
        X_rad = np.load(os.path.join(data_dir, 'X_rad_train_selected.npy'))
        X_2d = np.load(os.path.join(data_dir, 'X_2d_train_norm.npy'))
        X_3d = np.load(os.path.join(data_dir, 'X_3d_train_norm.npy'))
        y = np.load(os.path.join(data_dir, 'y_train.npy')).ravel()
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)

    # Quick validation
    n_samples = len(y)
    if X_rad.shape[0] != n_samples or X_2d.shape[0] != n_samples or X_3d.shape[0] != n_samples:
        logger.error("Mismatch in number of samples between features and labels.")
        sys.exit(1)

    logger.info(f"Loaded {n_samples} samples.")
    logger.info(f"Radiomics: {X_rad.shape[1]} features, 2D: {X_2d.shape[1]}, 3D: {X_3d.shape[1]}")

    # -------------------------------------------------------------------
    # Standardise and compute LDA+PCA
    # -------------------------------------------------------------------
    scaler = StandardScaler()
    feature_dict = {
        'Radiomics': X_rad,
        '2D CNN': X_2d,
        '3D CNN': X_3d,
    }
    features_scaled = {k: scaler.fit_transform(v) for k, v in feature_dict.items()}

    results = {}
    for name in ['Radiomics', '2D CNN', '3D CNN']:
        emb, metrics = compute_lda_pca(features_scaled[name], y)
        results[name] = {'emb': emb, **metrics}

    # -------------------------------------------------------------------
    # Print summary statistics
    # -------------------------------------------------------------------
    header = f"{'Feature':<15} {'Silhouette':>10} {'DBI':>10} {'LDA ΔMean':>10} {'Cohen d':>10}"
    print("\n" + "=" * 60)
    print("LDA + PCA Separation Metrics")
    print("=" * 60)
    print(header)
    print("-" * len(header))
    for name in ['Radiomics', '2D CNN', '3D CNN']:
        m = results[name]
        print(f"{name:<15} {m['silhouette']:>10.4f} {m['davies_bouldin']:>10.4f} "
              f"{m['lda_mean_diff']:>10.4f} {m['cohens_d']:>10.4f}")

    print("\nLDA Discriminant Axis Means:")
    for name in ['Radiomics', '2D CNN', '3D CNN']:
        m = results[name]
        print(f"  {name}: Female = {m['lda_female_mean']:.4f}, Male = {m['lda_male_mean']:.4f}")

    # -------------------------------------------------------------------
    # Generate individual plots
    # -------------------------------------------------------------------
    print("\n>>> Generating individual plots...")
    for name in ['Radiomics', '2D CNN', '3D CNN']:
        fig, ax = plt.subplots(figsize=(6, 5))
        m = results[name]
        plot_lda_pca(ax, m['emb'], y, name,
                     sil=m['silhouette'], db=m['davies_bouldin'])
        plt.tight_layout()
        safe_name = name.replace(' ', '_')
        save_figure(fig, f'{safe_name}_LDA_PCA', args.output_dir)

    # -------------------------------------------------------------------
    # Combined figure
    # -------------------------------------------------------------------
    print(">>> Generating combined figure...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, name, sub_lbl in zip(axes, ['Radiomics', '2D CNN', '3D CNN'], ['A', 'B', 'C']):
        m = results[name]
        plot_lda_pca(ax, m['emb'], y, name,
                     sil=m['silhouette'], db=m['davies_bouldin'],
                     sub_label=sub_lbl)
    plt.suptitle('Linear Discriminant Analysis + Orthogonal PCA Projection',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, 'ThreeFeatures_LDA_PCA_Comparison', args.output_dir)

    logger.info(f"All figures saved to {args.output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()