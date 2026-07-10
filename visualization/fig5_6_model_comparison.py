#!/usr/bin/env python3
"""
Figure 5 & 6 – Multi‑model Performance Comparison (Internal and External Test Sets).

This script loads predicted probabilities for seven models (Conventional,
Radiomics, 2D CNN, 3D CNN, Weighted Ensemble, FeatureConcat MLP, Transformer
Fusion) and generates comprehensive evaluation plots:
  - ROC curves with 95 % bootstrap confidence intervals
  - Calibration curves
  - Performance bar plots (AUC, Accuracy, Sensitivity, Specificity)
  - Confusion matrices (row percentages)
  - Decision Curve Analysis (DCA)
  - Performance heatmaps

Input can be provided as CSV files with columns:
  number, sex, prob_ModelName1, prob_ModelName2, ...

Usage:
    python fig5_6_model_comparison.py \
        --internal_csv <internal_probs.csv> \
        --external_csv <external_probs.csv> \
        --output_dir ./comparison_figures \
        [--seed 42] [--prob_prefix prob_]

Requirements: numpy, pandas, matplotlib, seaborn, scikit‑learn, scipy
"""

import argparse
import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix, roc_curve,
    f1_score, precision_score, recall_score
)
from sklearn.calibration import calibration_curve

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default Nature Communications style
# ---------------------------------------------------------------------------
NATURE_COLORS = [
    '#E64B35', '#4DBBD5', '#00A087', '#3C5488',
    '#F39B7F', '#8491B4', '#91D1C2', '#7E6148'
]

TICK_SIZE = 2.835  # points


def set_style(nc_colors=None):
    """Apply NC‑compatible matplotlib style. Tries Arial, falls back to sans‑serif."""
    if nc_colors is not None:
        global NATURE_COLORS
        NATURE_COLORS = list(nc_colors)

    font_family = 'sans-serif'
    try:
        if any('Arial' in f.name for f in fm.fontManager.ttflist):
            font_family = 'Arial'
    except Exception:
        pass

    rcParams.update({
        'font.family': font_family,
        'font.size': 14,
        'axes.labelsize': 14,
        'axes.titlesize': 15,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
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
# Helper functions
# ---------------------------------------------------------------------------
def read_csv_auto(filepath, encodings=None):
    """Read CSV with fallback encodings."""
    if encodings is None:
        encodings = ['utf-8-sig', 'gbk', 'gb2312', 'latin1', 'utf-8']
    for enc in encodings:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            logger.info(f"Read {filepath} with encoding {enc}")
            return df
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"Unable to read {filepath} with any encoding.")


def get_best_threshold(y_true, y_prob):
    """Find threshold that maximises Youden's J statistic."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return thresholds[np.argmax(tpr - fpr)]


def bootstrap_auc_ci(y_true, y_prob, n_bootstrap=1000, alpha=0.05, seed=42):
    """Bootstrap 95 % confidence interval for AUC."""
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    if not aucs:
        return np.nan, np.nan, np.nan
    return np.mean(aucs), np.percentile(aucs, 100 * alpha / 2), np.percentile(aucs, 100 * (1 - alpha / 2))


def net_benefit(y_true, prob, thresh):
    """Compute net benefit at a given threshold."""
    pred = (prob >= thresh).astype(int)
    tp = np.sum((pred == 1) & (y_true == 1))
    fp = np.sum((pred == 1) & (y_true == 0))
    n = len(y_true)
    return tp / n - (fp / n) * (thresh / (1 - thresh))


def save_figure(fig, base_name, output_dir):
    """Save figure as PDF and PNG."""
    for ext in ['pdf', 'png']:
        path = os.path.join(output_dir, f'{base_name}.{ext}')
        fig.savefig(path, format=ext, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f'Saved {base_name} (PDF + PNG)')


# ---------------------------------------------------------------------------
# Core evaluation and visualisation
# ---------------------------------------------------------------------------
def evaluate_models(y_true, prob_dict, output_dir, dataset_name, seed=42):
    """
    Compute metrics and generate all comparison plots for a single dataset.

    Parameters
    ----------
    y_true : ndarray
        Ground truth labels (0/1).
    prob_dict : dict of {model_name: probability_array}
    output_dir : str
    dataset_name : str
    seed : int, used for bootstrap confidence intervals.
    """
    os.makedirs(output_dir, exist_ok=True)
    thresholds = {name: get_best_threshold(y_true, prob) for name, prob in prob_dict.items()}

    metrics = {}
    for name, prob in prob_dict.items():
        thr = thresholds[name]
        pred = (prob >= thr).astype(int)
        auc = roc_auc_score(y_true, prob)
        _, ci_low, ci_high = bootstrap_auc_ci(y_true, prob, seed=seed)
        acc = accuracy_score(y_true, pred)
        sens = recall_score(y_true, pred)
        spec = recall_score(y_true, pred, pos_label=0)
        prec = precision_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        metrics[name] = {
            'AUC': auc, 'AUC_CI': (ci_low, ci_high),
            'Accuracy': acc, 'Sensitivity': sens,
            'Specificity': spec, 'Precision': prec, 'F1': f1
        }

    # Print summary table
    print(f'\n{"="*60}')
    print(f'Performance Summary – {dataset_name}')
    print(f'{"="*60}')
    df_metrics = pd.DataFrame(metrics).T
    df_metrics['AUC (95% CI)'] = df_metrics.apply(
        lambda r: f"{r['AUC']:.3f} ({r['AUC_CI'][0]:.3f}–{r['AUC_CI'][1]:.3f})", axis=1
    )
    print(df_metrics[['Accuracy', 'Sensitivity', 'Specificity', 'AUC (95% CI)']].to_string())

    model_names = list(prob_dict.keys())
    n_models = len(model_names)

    # ---------- ROC Curve ----------
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, name in enumerate(model_names):
        prob = prob_dict[name]
        fpr, tpr, _ = roc_curve(y_true, prob)
        auc_val = metrics[name]['AUC']
        ci = metrics[name]['AUC_CI']
        ax.plot(fpr, tpr, color=NATURE_COLORS[i % len(NATURE_COLORS)], lw=2.5,
                label=f'{name} (AUC={auc_val:.3f} [{ci[0]:.3f}–{ci[1]:.3f}])')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Chance')
    ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate', fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontweight='bold')
    ax.set_title(f'ROC Curves – {dataset_name}', fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    sns.despine()
    save_figure(fig, f'{dataset_name}_ROC', output_dir)

    # ---------- Calibration Curves ----------
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, name in enumerate(model_names):
        frac, mpred = calibration_curve(y_true, prob_dict[name], n_bins=10, strategy='uniform')
        ax.plot(mpred, frac, 'o-', color=NATURE_COLORS[i % len(NATURE_COLORS)], markersize=6, lw=2, label=name)
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfect Calibration')
    ax.set_xlabel('Mean Predicted Probability', fontweight='bold')
    ax.set_ylabel('Fraction of Positives', fontweight='bold')
    ax.set_title(f'Calibration Curves – {dataset_name}', fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    sns.despine()
    save_figure(fig, f'{dataset_name}_Calibration', output_dir)

    # ---------- Bar Plots ----------
    bar_metrics = ['AUC', 'Accuracy', 'Sensitivity', 'Specificity']
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, metric in zip(axes.flat, bar_metrics):
        values = [metrics[n][metric] for n in model_names]
        colors = [NATURE_COLORS[i % len(NATURE_COLORS)] for i in range(n_models)]
        if metric == 'AUC':
            err = [(metrics[n]['AUC'] - metrics[n]['AUC_CI'][0],
                    metrics[n]['AUC_CI'][1] - metrics[n]['AUC']) for n in model_names]
            err = np.array(err).T
            ax.bar(model_names, values, color=colors, yerr=err, capsize=5)
        else:
            ax.bar(model_names, values, color=colors)
        ax.set_ylabel(metric, fontweight='bold')
        ax.set_ylim(0.5, 1.0)
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
        sns.despine(ax=ax)
    fig.suptitle(f'Performance Metrics – {dataset_name}', fontweight='bold')
    plt.tight_layout()
    save_figure(fig, f'{dataset_name}_Barplot', output_dir)

    # ---------- Confusion Matrices (row percentages) ----------
    for name in model_names:
        pred = (prob_dict[name] >= thresholds[name]).astype(int)
        cm = confusion_matrix(y_true, pred)
        acc = accuracy_score(y_true, pred)
        sens = recall_score(y_true, pred)
        spec = recall_score(y_true, pred, pos_label=0)

        # Row‑wise percentage annotation
        row_sum = cm.sum(axis=1, keepdims=True)
        row_perc = np.where(row_sum > 0, cm.astype(float) / row_sum * 100, 0.0)
        annot = np.array([[f'{cm[i,j]}\n({row_perc[i,j]:.1f}%)' for j in range(2)] for i in range(2)])

        fig, ax = plt.subplots(figsize=(4.5, 5))
        sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', cbar=False,
                    xticklabels=['Pred Female', 'Pred Male'],
                    yticklabels=['True Female', 'True Male'],
                    annot_kws={'size': 10}, linewidths=1.5, linecolor='white',
                    vmin=0, vmax=cm.max(), ax=ax)
        ax.set_title(f'{name} (N={cm.sum()})', fontweight='bold')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        fig.text(0.5, 0.01, f'Acc={acc:.3f}  Sens={sens:.3f}  Spec={spec:.3f}',
                 ha='center', fontsize=10)
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        safe_name = name.replace(' ', '_')
        save_figure(fig, f'{dataset_name}_CM_{safe_name}', output_dir)

    # ---------- Decision Curve Analysis ----------
    thresholds_dca = np.linspace(0.01, 0.99, 200)
    treat_all = [net_benefit(y_true, np.ones_like(y_true), t) for t in thresholds_dca]
    y_max = max(treat_all) * 1.1
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, name in enumerate(model_names):
        nb = [net_benefit(y_true, prob_dict[name], t) for t in thresholds_dca]
        ax.plot(thresholds_dca, nb, color=NATURE_COLORS[i % len(NATURE_COLORS)], lw=2.5, label=name)
    ax.plot(thresholds_dca, treat_all, 'k--', lw=1.5, label='Treat All')
    ax.plot(thresholds_dca, [net_benefit(y_true, np.zeros_like(y_true), t) for t in thresholds_dca],
            'k:', lw=1.5, label='Treat None')
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.set_xlim([0, 0.8]); ax.set_ylim([-0.05, y_max])
    ax.set_xlabel('Threshold Probability', fontweight='bold')
    ax.set_ylabel('Net Benefit', fontweight='bold')
    ax.set_title(f'Decision Curve Analysis – {dataset_name}', fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, frameon=False)
    sns.despine()
    plt.tight_layout()
    save_figure(fig, f'{dataset_name}_DCA', output_dir)

    # ---------- Heatmap ----------
    heat_data = df_metrics[['AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'F1']].astype(float)
    vmin = heat_data.min().min() - 0.02
    fig, ax = plt.subplots(figsize=(max(8, n_models * 0.8), 8))
    sns.heatmap(heat_data, annot=True, fmt='.3f', cmap='RdYlBu_r', linewidths=0.5,
                cbar_kws={'label': ''}, annot_kws={'size': 11}, vmin=vmin, vmax=1.0, ax=ax)
    ax.set_title(f'Model Performance Heatmap – {dataset_name}', fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()
    save_figure(fig, f'{dataset_name}_Heatmap', output_dir)

    # ---------- Table ----------
    fig, ax = plt.subplots(figsize=(14, max(4, n_models * 0.4)))
    ax.axis('off')
    table_data = [[name,
                   f"{metrics[name]['Accuracy']:.3f}",
                   f"{metrics[name]['Sensitivity']:.3f}",
                   f"{metrics[name]['Specificity']:.3f}",
                   f"{metrics[name]['AUC']:.3f} ({metrics[name]['AUC_CI'][0]:.3f}–{metrics[name]['AUC_CI'][1]:.3f})"]
                  for name in model_names]
    columns = ['Model', 'Accuracy', 'Sensitivity', 'Specificity', 'AUC (95% CI)']
    table = ax.table(cellText=table_data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax.set_title(f'Performance Summary – {dataset_name}', fontweight='bold', pad=20)
    plt.tight_layout()
    save_figure(fig, f'{dataset_name}_Table', output_dir)

    return metrics


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate comparison figures for multiple sex estimation models.'
    )
    parser.add_argument('--internal_csv', type=str, required=True,
                        help='CSV file with internal test set probabilities. '
                             'Columns: "number", "sex", and one column per model (e.g. prob_Conventional, etc.)')
    parser.add_argument('--external_csv', type=str, required=True,
                        help='CSV file with external test set probabilities (same format).')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save all figures.')
    parser.add_argument('--prob_prefix', type=str, default='prob_',
                        help='Prefix for probability columns (default: "prob_").')
    parser.add_argument('--nc_colors', type=str, nargs='+', default=None,
                        help='Custom hex color list for the models.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for bootstrap CI (default: 42).')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    set_style(args.nc_colors)

    # Load data
    logger.info('Loading internal CSV...')
    df_int = read_csv_auto(args.internal_csv)
    logger.info('Loading external CSV...')
    df_ext = read_csv_auto(args.external_csv)

    for df, name in [(df_int, 'internal'), (df_ext, 'external')]:
        if 'sex' not in df.columns or 'number' not in df.columns:
            logger.error(f'{name} CSV must contain "number" and "sex" columns.')
            sys.exit(1)
        if not set(df['sex'].dropna().unique()).issubset({0, 1}):
            logger.warning(f'{name} CSV sex column contains values other than 0/1; ensure labels are correctly encoded.')

    # Identify probability columns
    prob_cols_int = [c for c in df_int.columns if c.startswith(args.prob_prefix)]
    prob_cols_ext = [c for c in df_ext.columns if c.startswith(args.prob_prefix)]

    if not prob_cols_int or not prob_cols_ext:
        logger.error('No probability columns found. Please check the --prob_prefix option.')
        sys.exit(1)

    common_models = sorted(set(c.replace(args.prob_prefix, '') for c in prob_cols_int) &
                           set(c.replace(args.prob_prefix, '') for c in prob_cols_ext))
    if not common_models:
        logger.error('No overlapping model names between internal and external CSVs.')
        sys.exit(1)

    logger.info(f'Models to compare: {common_models}')

    y_int = df_int['sex'].values
    y_ext = df_ext['sex'].values

    prob_dict_int = {m: df_int[f'{args.prob_prefix}{m}'].values for m in common_models}
    prob_dict_ext = {m: df_ext[f'{args.prob_prefix}{m}'].values for m in common_models}

    # Generate figures
    logger.info('Evaluating internal test set...')
    evaluate_models(y_int, prob_dict_int, os.path.join(args.output_dir, 'internal'), 'Internal', seed=args.seed)

    logger.info('Evaluating external test set...')
    evaluate_models(y_ext, prob_dict_ext, os.path.join(args.output_dir, 'external'), 'External', seed=args.seed)

    logger.info(f'All figures saved to {args.output_dir}')
    print('Done.')


if __name__ == '__main__':
    main()