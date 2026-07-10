#!/usr/bin/env python3
"""
Train and evaluate a conventional morphometric logistic regression model
for sex estimation from patellar CT measurements.

The pipeline implements the "Conventional" baseline described in the paper:
- Six automated morphometric parameters: patellar length, width, thickness,
  volume, surface area, and coronal perimeter.
- Standardisation (zero mean, unit variance).
- L2-regularised logistic regression.
- Hyperparameter C tuned via 5‑fold cross‑validation over [0.01, 0.1, 1, 10, 100]
  with AUC as the optimisation objective.

Requirements: pandas, numpy, scikit-learn, scipy, matplotlib, joblib
"""

import argparse
import json
import logging
import sys
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Configure basic logging (also keep console output via print for clarity)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def read_csv_with_encoding(filepath: Union[str, Path]) -> pd.DataFrame:
    """Read a CSV file by trying several common encodings.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError
        If no encoding works.
    """
    filepath = Path(filepath)
    for enc in ["utf-8", "gbk", "gb2312", "gb18030", "utf-8-sig", "latin1"]:
        try:
            df = pd.read_csv(str(filepath), encoding=enc)
            logger.info("Loaded %s with encoding '%s'", filepath.name, enc)
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot read {filepath} with any tried encoding.")


# ---------------------------------------------------------------------------
# Column matching
# ---------------------------------------------------------------------------
def fuzzy_match_columns(
    expected: List[str],
    actual_columns: List[str],
    cutoff: float = 0.6,
) -> Dict[str, str]:
    """Fuzzy‑match expected column names to actual columns.

    Parameters
    ----------
    expected : list of str
        Desired display names (e.g. "Patellar length").
    actual_columns : list of str
        Columns present in the DataFrame.
    cutoff : float
        Minimum similarity ratio (0–1) for a match.

    Returns
    -------
    dict
        Mapping ``{expected_name: actual_column_name}``.  Raises ``KeyError``
        if any expected name cannot be matched.
    """
    mapping = {}
    for exp in expected:
        matches = get_close_matches(exp, actual_columns, n=1, cutoff=cutoff)
        if not matches:
            raise KeyError(
                f"No matching column found for '{exp}'. "
                f"Available columns: {actual_columns}"
            )
        mapping[exp] = matches[0]
        logger.info("Matched '%s' -> '%s'", exp, matches[0])
    return mapping


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------
def t_test_report(
    df: pd.DataFrame,
    sex_col: str,
    var_mapping: Dict[str, str],
    male_label: int = 1,
    female_label: int = 0,
) -> pd.DataFrame:
    """Perform independent t‑tests for each variable between sexes.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset containing both feature and sex columns.
    sex_col : str
        Name of the sex column.
    var_mapping : dict
        Mapping {display_name: actual_column_name}.
    male_label : int
        Value coding for male (default 1).
    female_label : int
        Value coding for female (default 0).

    Returns
    -------
    pd.DataFrame
        Table with means, t‑statistic, p‑value and Cohen's d.
    """
    male = df[df[sex_col] == male_label]
    female = df[df[sex_col] == female_label]
    rows = []
    for display_name, act in var_mapping.items():
        m_arr = male[act].dropna()
        f_arr = female[act].dropna()

        # Check minimum sample size for statistical tests
        if len(m_arr) < 2 or len(f_arr) < 2:
            rows.append(
                {
                    "Variable": display_name,
                    "Male_mean": round(m_arr.mean(), 2) if len(m_arr) > 0 else np.nan,
                    "Female_mean": round(f_arr.mean(), 2) if len(f_arr) > 0 else np.nan,
                    "Mean_diff": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "Cohens_d": np.nan,
                }
            )
            continue

        t_stat, p_val = stats.ttest_ind(m_arr, f_arr)
        mean_m, mean_f = m_arr.mean(), f_arr.mean()
        diff = mean_m - mean_f
        # pooled standard deviation
        n_m, n_f = len(m_arr), len(f_arr)
        var_m = m_arr.var(ddof=1)
        var_f = f_arr.var(ddof=1)
        pooled_std = np.sqrt(
            ((n_m - 1) * var_m + (n_f - 1) * var_f) / (n_m + n_f - 2)
        )
        cohens_d = diff / pooled_std if pooled_std != 0 else np.nan
        rows.append(
            {
                "Variable": display_name,
                "Male_mean": round(mean_m, 2),
                "Female_mean": round(mean_f, 2),
                "Mean_diff": round(diff, 2),
                "t_stat": round(t_stat, 4),
                "p_value": f"{p_val:.6f}",
                "Cohens_d": round(cohens_d, 3),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5,
    param_grid: Optional[Dict] = None,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Tuple[Pipeline, GridSearchCV]:
    """Train a standardised Logistic Regression model with hyperparameter search.

    Parameters
    ----------
    X_train, y_train : array‑like
        Training data.
    cv_folds : int
        Number of cross‑validation folds (default 5).
    param_grid : dict, optional
        Grid for ``logreg__C``.  Default: [0.01, 0.1, 1, 10, 100].
    random_state : int
        Random seed used for both the model and the cross‑validation splits.
    n_jobs : int
        Number of parallel jobs for GridSearchCV.

    Returns
    -------
    best_model : Pipeline
        Fitted pipeline (StandardScaler + LogisticRegression).
    grid : GridSearchCV
        Fitted grid search object.
    """
    if param_grid is None:
        param_grid = {"logreg__C": [0.01, 0.1, 1, 10, 100]}

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    penalty="l2",
                    max_iter=2000,
                    random_state=random_state,
                ),
            ),
        ]
    )

    # Explicit stratified k‑fold for reproducibility
    cv = StratifiedKFold(
        n_splits=cv_folds, shuffle=True, random_state=random_state
    )

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=n_jobs,
        verbose=1,
    )
    grid.fit(X_train, y_train)
    logger.info("Best hyperparameters: %s", grid.best_params_)
    logger.info("Best CV AUC: %.4f", grid.best_score_)

    # Show all combinations
    cv_df = pd.DataFrame(grid.cv_results_)
    # Robust extraction of parameter columns (they all start with 'param_')
    param_cols = [col for col in cv_df.columns if col.startswith("param_")]
    logger.info(
        "CV results:\n%s",
        cv_df[param_cols + ["mean_test_score", "std_test_score"]]
        .round(4)
        .to_string(),
    )

    return grid.best_estimator_, grid


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(
    model: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
) -> Dict[str, Any]:
    """Compute accuracy, AUC and other metrics on test set.

    Parameters
    ----------
    model : Pipeline
        Trained model.
    X_test, y_test : array‑like
        Test data.
    class_names : list of str
        Names of the classes (e.g. ['Female', 'Male']).

    Returns
    -------
    dict
        A dictionary containing:
        - 'accuracy' (float)
        - 'auc' (float)
        - 'y_proba' (ndarray): predicted probabilities for positive class.
        - 'y_true' (ndarray): true binary labels.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc_val = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print("\nTest set performance:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  AUC:       {auc_val:.4f}")
    print(f"  Confusion matrix:\n{cm}")
    print(classification_report(y_test, y_pred, target_names=class_names))

    return {
        "accuracy": float(acc),
        "auc": float(auc_val),
        "y_proba": y_proba,
        "y_true": y_test,
    }


# ---------------------------------------------------------------------------
# Discriminant formula (raw scale)
# ---------------------------------------------------------------------------
def extract_discriminant_formula(
    model: Pipeline,
    variable_names: List[str],
) -> Tuple[float, np.ndarray]:
    """Convert standardised coefficients back to the original measurement scale.

    Parameters
    ----------
    model : Pipeline
        Trained model (must contain 'scaler' and 'logreg' steps).
    variable_names : list of str
        Names of the input variables (display names).

    Returns
    -------
    intercept : float
        Intercept on the original scale.
    coef_orig : ndarray
        Coefficients on the original scale.
    """
    scaler = model.named_steps["scaler"]
    logreg = model.named_steps["logreg"]

    means = scaler.mean_
    stds = scaler.scale_
    coef_std = logreg.coef_[0]
    intercept_std = logreg.intercept_[0]

    coef_orig = coef_std / stds
    intercept_orig = intercept_std - np.sum(coef_std * means / stds)

    print("\nDiscriminant formula (original scale):")
    print(f"  z = {intercept_orig:.4f}")
    for name, c in zip(variable_names, coef_orig):
        print(f"      + ({c:.4f}) x {name}")

    # Variable importance (standardised |coefficient|)
    importance = pd.DataFrame(
        {
            "Variable": variable_names,
            "Std_coef": coef_std,
        }
    ).sort_values("Std_coef", key=abs, ascending=False)
    print("\nVariable importance (standardised coefficients):")
    print(importance.to_string(index=False))

    return intercept_orig, coef_orig


# ---------------------------------------------------------------------------
# Plot and save
# ---------------------------------------------------------------------------
def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "ROC Curve (Test set)",
) -> None:
    """Plot ROC curve and optionally save the figure.

    Parameters
    ----------
    y_true : ndarray
        True binary labels.
    y_proba : ndarray
        Predicted probabilities for the positive class.
    save_path : Path, optional
        If provided, save the figure to this path.
    title : str
        Plot title.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.3f})"
    )
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
        logger.info("ROC curve saved to %s", save_path)
    plt.show()


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments.

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description=(
            "Train a conventional logistic regression model for patellar "
            "sex estimation.  Implements the 'Conventional' baseline from "
            "the Patella-MTF paper."
        )
    )
    parser.add_argument(
        "train_csv", type=str, help="Path to training CSV file"
    )
    parser.add_argument(
        "test_csv", type=str, help="Path to test CSV file"
    )
    parser.add_argument(
        "output_dir", type=str, help="Directory to save outputs"
    )
    parser.add_argument(
        "--sex-col",
        type=str,
        default="sex",
        help="Name of the sex column (default: 'sex')",
    )
    parser.add_argument(
        "--male-label",
        type=int,
        default=1,
        help="Value for male in sex column (default: 1)",
    )
    parser.add_argument(
        "--female-label",
        type=int,
        default=0,
        help="Value for female in sex column (default: 0)",
    )
    parser.add_argument(
        "--measurement-columns",
        nargs="+",
        default=[
            "Patellar length",
            "Patellar width",
            "Patellar thickness",
            "Patellar volume",
            "Patellar surface area",
            "Patellar coronal perimeter",
        ],
        help=(
            "Expected measurement column names (English recommended). "
            "Fuzzy matching will be applied."
        ),
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=0.6,
        help="Fuzzy matching cutoff ratio (default: 0.6)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    parser.add_argument(
        "--skip-stats",
        action="store_true",
        help="Skip descriptive statistics and t-tests",
    )
    parser.add_argument(
        "--no-roc-plot",
        action="store_true",
        help="Do not display or save ROC plot",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point.

    Returns
    -------
    int
        Exit code (0 on success, 1 on known errors).
    """
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # -------------------------------------------------------------------
        # Load data
        # -------------------------------------------------------------------
        train_df = read_csv_with_encoding(args.train_csv)
        test_df = read_csv_with_encoding(args.test_csv)

        # Match measurement columns (fuzzy) using training set columns
        actual_cols = [col.strip() for col in train_df.columns]
        col_map = fuzzy_match_columns(
            args.measurement_columns, actual_cols, cutoff=args.cutoff
        )
        selected_vars = list(col_map.values())

        # Verify sex column existence
        if (
            args.sex_col not in train_df.columns
            or args.sex_col not in test_df.columns
        ):
            logger.error("Sex column '%s' not found in input data.", args.sex_col)
            return 1

        # Verify test set contains all required columns
        missing_test_cols = set(selected_vars) - set(test_df.columns)
        if missing_test_cols:
            logger.error(
                "Test set is missing the following columns: %s",
                missing_test_cols,
            )
            return 1

        # -------------------------------------------------------------------
        # Extract features and labels
        # -------------------------------------------------------------------
        X_train = train_df[selected_vars]
        y_train = train_df[args.sex_col]
        X_test = test_df[selected_vars]
        y_test = test_df[args.sex_col]

        # Check for missing values
        if X_train.isnull().any().any():
            logger.warning(
                "Training features contain missing values. "
                "Consider imputation before training."
            )
        if X_test.isnull().any().any():
            logger.warning(
                "Test features contain missing values. "
                "Performance may be affected."
            )

        logger.info(
            "Training samples: %d (Male: %d, Female: %d)",
            len(X_train),
            (y_train == args.male_label).sum(),
            (y_train == args.female_label).sum(),
        )
        logger.info(
            "Test samples: %d (Male: %d, Female: %d)",
            len(X_test),
            (y_test == args.male_label).sum(),
            (y_test == args.female_label).sum(),
        )

        # -------------------------------------------------------------------
        # Descriptive statistics (optional)
        # -------------------------------------------------------------------
        if not args.skip_stats:
            print("\n" + "=" * 60)
            print("Sexual dimorphism t-tests on training set")
            print("=" * 60)
            stats_df = t_test_report(
                train_df,
                args.sex_col,
                col_map,
                args.male_label,
                args.female_label,
            )
            print(stats_df.to_string(index=False))
            stats_df.to_csv(out_dir / "t_test_results.csv", index=False)

        # -------------------------------------------------------------------
        # Model training
        # -------------------------------------------------------------------
        print(
            f"\n{'=' * 60}\n"
            f"Training logistic regression model ({args.cv_folds}-fold CV)\n"
            f"{'=' * 60}"
        )
        model, grid_search = train_logistic_regression(
            X_train.values,
            y_train.values,
            cv_folds=args.cv_folds,
            random_state=args.random_state,
        )

        # -------------------------------------------------------------------
        # Evaluation on test set
        # -------------------------------------------------------------------
        metrics = evaluate_model(
            model,
            X_test.values,
            y_test.values,
            class_names=["Female", "Male"],
        )

        # -------------------------------------------------------------------
        # Discriminant formula (original scale)
        # -------------------------------------------------------------------
        intercept, coefs = extract_discriminant_formula(
            model, list(col_map.keys())
        )

        # -------------------------------------------------------------------
        # Cross-validation accuracy on training set
        # -------------------------------------------------------------------
        cv = StratifiedKFold(
            n_splits=args.cv_folds, shuffle=True, random_state=args.random_state
        )
        cv_acc = cross_val_score(
            model,
            X_train.values,
            y_train.values,
            cv=cv,
            scoring="accuracy",
        )
        logger.info(
            "%d-fold CV accuracy on training set: %.4f (+/- %.4f)",
            args.cv_folds,
            cv_acc.mean(),
            cv_acc.std(),
        )

        # -------------------------------------------------------------------
        # Save artifacts
        # -------------------------------------------------------------------
        import sklearn

        model_info = {
            "intercept": float(intercept),
            "coefficients": coefs.tolist(),
            "variable_names": list(col_map.keys()),
            "actual_columns": selected_vars,
            "accuracy": metrics["accuracy"],
            "auc": metrics["auc"],
            "best_params": grid_search.best_params_,
            "cv_mean_auc": float(grid_search.best_score_),
            "cv_accuracy_mean": float(cv_acc.mean()),
            "cv_accuracy_std": float(cv_acc.std()),
            "environment": {
                "sklearn_version": sklearn.__version__,
                "numpy_version": np.__version__,
                "pandas_version": pd.__version__,
            },
        }
        json_path = out_dir / "conventional_model_info.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        logger.info("Model info saved to %s", json_path)

        model_path = out_dir / "logistic_regression_pipeline.pkl"
        joblib.dump(model, model_path)
        logger.info("Trained pipeline saved to %s", model_path)

        # -------------------------------------------------------------------
        # ROC plot
        # -------------------------------------------------------------------
        if not args.no_roc_plot:
            roc_path = out_dir / "roc_curve.png"
            plot_roc_curve(
                metrics["y_true"],
                metrics["y_proba"],
                save_path=roc_path,
                title="Sex Estimation ROC Curve (Test set)",
            )

        print("\nDone.")
        return 0

    except Exception as e:
        logger.exception("An unexpected error occurred: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())