#!/usr/bin/env python3
"""
Train a radiomics-based logistic regression model for sex estimation from
patellar CT images.

Pipeline (mirrors the paper's Methods):
1. Extract radiomic features from preprocessed NIfTI images and masks
   using PyRadiomics (bin width = 25 HU, label = 1).
2. Clean features: remove constant features, drop features with >50% missing
   values, fill remaining missing values with per-feature mean.
3. Reduce redundancy via Spearman correlation (|r| >= 0.9).
4. Standardise the features (zero mean, unit variance).
5. Perform feature selection with LASSO (LassoCV).
6. Train an L2-regularised logistic regression classifier on the selected,
   standardised features. Hyperparameter C is tuned via 5‑fold cross‑validation
   grid search over [0.01, 0.1, 1, 10, 100] with AUC as the optimisation target
   – exactly as described in the paper.  No further standardisation is applied
   during the final model training because the features are already standardised.

Output files:
    scaler_56feat.pkl               – Fitted StandardScaler.
    lasso_selector_56feat.pkl       – Spearman keep indices, LASSO selector, feature names.
    gender_prediction_model_56feat.pkl – Trained logistic regression pipeline.

Requirements: numpy, pandas, SimpleITK, pyradiomics, scikit-learn, tqdm
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import SimpleITK as sitk
import sklearn
from radiomics import featureextractor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------
def find_image_mask_pair(
    sample_id: int,
    img_dir: Path,
    mask_dir: Path,
    rater_id: int = 2,
) -> Tuple[Optional[Path], Optional[Path]]:
    """Locate image and mask files for a given sample.

    Expected naming: ``<sample_id>-<rater_id>.nii.gz``.
    """
    img_path = img_dir / f"{sample_id}-{rater_id}.nii.gz"
    mask_path = mask_dir / f"{sample_id}-{rater_id}.nii.gz"
    if img_path.is_file() and mask_path.is_file():
        return img_path, mask_path
    return None, None


def extract_features_from_sample(
    img_path: Path,
    mask_path: Path,
    extractor: featureextractor.RadiomicsFeatureExtractor,
) -> Optional[Dict[str, float]]:
    """Extract radiomic features from a single image/mask pair."""
    try:
        image = sitk.ReadImage(str(img_path))
        mask = sitk.ReadImage(str(mask_path))
        if image.GetSize() != mask.GetSize():
            logger.warning("Size mismatch between image and mask: %s", img_path)
            return None
        result = extractor.execute(image, mask)
        feat_dict = {
            k: v for k, v in result.items() if not k.startswith("diagnostics_")
        }
        return feat_dict
    except Exception as e:
        logger.error("Feature extraction failed for %s: %s", img_path, e)
        return None


def build_feature_matrices(
    df_meta: pd.DataFrame,
    img_dir: Path,
    mask_dir: Path,
    id_col: str = "number",
    rater_id: int = 2,
) -> Tuple[pd.DataFrame, pd.Series, List[int]]:
    """Extract radiomic features for all valid samples."""
    # resampledPixelSpacing=None disables internal resampling – images are already isotropic.
    params = {
        "binWidth": 25,
        "resampledPixelSpacing": None,
        "normalize": False,
        "label": 1,
        "verbose": False,
    }
    extractor = featureextractor.RadiomicsFeatureExtractor(**params)
    extractor.enableAllFeatures()
    logger.info("Enabled feature classes: %s", extractor.enabledFeatures)

    features_list, labels_list, valid_ids = [], [], []
    for _, row in tqdm(df_meta.iterrows(), total=len(df_meta), desc="Extracting features"):
        sid = int(row[id_col])
        img_path, mask_path = find_image_mask_pair(sid, img_dir, mask_dir, rater_id)
        if img_path is None:
            logger.warning("Skipping sample %d (missing image or mask)", sid)
            continue
        feat = extract_features_from_sample(img_path, mask_path, extractor)
        if feat is not None:
            features_list.append(feat)
            labels_list.append(row["sex"])
            valid_ids.append(sid)

    if not features_list:
        raise RuntimeError("No features could be extracted. Check your data paths.")

    X = pd.DataFrame(features_list)
    y = pd.Series(labels_list, name="sex")
    logger.info("Extracted features for %d samples, %d raw features", len(X), X.shape[1])
    return X, y, valid_ids


# ---------------------------------------------------------------------------
# Feature cleaning
# ---------------------------------------------------------------------------
def clean_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Clean and align training and test feature matrices.

    Steps:
    1. Intersect columns to ensure both sets have identical features.
    2. Remove constant features (variance < 1e-10 on training set).
    3. Remove features with >50% missing values.
    4. Drop columns that are entirely NaN.
    5. Impute remaining missing values with training-set column means.
    """
    # Align columns
    common_cols = list(set(X_train.columns).intersection(set(X_test.columns)))
    logger.info("Train columns: %d, Test columns: %d, Common: %d",
                len(X_train.columns), len(X_test.columns), len(common_cols))
    X_train, X_test = X_train[common_cols], X_test[common_cols]

    # Remove constant features
    variances = X_train.var()
    const_cols = variances[variances < 1e-10].index.tolist()
    if const_cols:
        X_train.drop(columns=const_cols, inplace=True)
        X_test.drop(columns=const_cols, inplace=True)

    # Remove features with >50% missing
    missing_ratio = X_train.isnull().mean()
    high_miss = missing_ratio[missing_ratio > 0.5].index.tolist()
    if high_miss:
        X_train.drop(columns=high_miss, inplace=True)
        X_test.drop(columns=high_miss, inplace=True)

    # Drop all‑NaN columns
    all_nan_cols = X_train.columns[X_train.isnull().all()].tolist()
    if all_nan_cols:
        logger.warning("Dropping columns with all NaN values: %s", all_nan_cols)
        X_train.drop(columns=all_nan_cols, inplace=True)
        X_test.drop(columns=all_nan_cols, inplace=True)

    # Impute with training means
    train_means = X_train.mean()
    X_train.fillna(train_means, inplace=True)
    X_test.fillna(train_means, inplace=True)

    # Ensure identical column order
    X_test = X_test[X_train.columns.tolist()]
    logger.info("Cleaned features: %d remaining", X_train.shape[1])
    return X_train, X_test


# ---------------------------------------------------------------------------
# Feature selection (Spearman + LASSO)
# ---------------------------------------------------------------------------
def select_features_spearman_lasso(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    spearman_threshold: float = 0.9,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[int], SelectFromModel, StandardScaler]:
    """Spearman correlation filtering, standardisation, LASSO selection.

    Spearman correlation is computed on the original feature values (rank‑based).
    After filtering, a global StandardScaler is fitted on the training set and
    applied to both sets.  LASSO is then applied to the standardised data.

    Returns
    -------
    X_train_sel : ndarray
        Selected, standardised training features.
    X_test_sel : ndarray
        Selected, standardised test features.
    keep_idx : list of int
        Indices of features retained after Spearman filtering (w.r.t cleaned columns).
    selected_indices : list of int
        Indices of final features after LASSO (w.r.t cleaned columns).
    lasso_selector : SelectFromModel
        Fitted LASSO selector.
    scaler : StandardScaler
        Fitted StandardScaler used before LASSO; also expected to be reused
        during inference.
    """
    # Spearman correlation filtering (rank‑based, insensitive to linear scaling)
    spearman_corr = X_train.corr(method="spearman").abs()
    high_corr = np.where(spearman_corr.values >= spearman_threshold)
    remove_idx = set()
    for i, j in zip(*high_corr):
        if i < j:
            remove_idx.add(j)
    keep_idx = [i for i in range(X_train.shape[1]) if i not in remove_idx]
    X_train_corr = X_train.iloc[:, keep_idx].values.astype(float)
    X_test_corr = X_test.iloc[:, keep_idx].values.astype(float)
    logger.info("After Spearman filtering: %d features", len(keep_idx))

    # Standardise (once, globally)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_corr)
    X_test_scaled = scaler.transform(X_test_corr)

    # LASSO selection
    lasso = LassoCV(
        cv=5,
        random_state=random_state,
        max_iter=2000,
        n_jobs=-1,
        n_alphas=100,
    )
    lasso.fit(X_train_scaled, y_train)
    selector = SelectFromModel(lasso, prefit=True)
    X_train_sel = selector.transform(X_train_scaled)
    X_test_sel = selector.transform(X_test_scaled)

    lasso_support = selector.get_support(indices=True)
    selected_indices = [keep_idx[i] for i in lasso_support]
    logger.info("LASSO selected %d features", X_train_sel.shape[1])

    return X_train_sel, X_test_sel, keep_idx, selected_indices, selector, scaler


# ---------------------------------------------------------------------------
# Model training (no additional standardisation)
# ---------------------------------------------------------------------------
def train_radiomics_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5,
    param_grid: Optional[Dict] = None,
    random_state: int = 42,
) -> Tuple[Pipeline, GridSearchCV]:
    """Train an L2‑regularised logistic regression on already‑standardised features.

    The pipeline contains only the classifier because the input features have
    already been standardised during feature selection.
    """
    if param_grid is None:
        param_grid = {"logreg__C": [0.01, 0.1, 1, 10, 100]}

    pipe = Pipeline([
        ("logreg", LogisticRegression(
            penalty="l2",
            max_iter=2000,
            random_state=random_state,
        ))
    ])
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    grid = GridSearchCV(
        pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)
    logger.info("Best hyperparameters: %s", grid.best_params_)
    logger.info("Best CV AUC: %.4f", grid.best_score_)
    return grid.best_estimator_, grid


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(
    model: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str] = ["Female", "Male"],
) -> Dict[str, Any]:
    """Evaluate the model on a test set."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc_val = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print("\nTest set performance:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  AUC:      {auc_val:.4f}")
    print(f"  Confusion matrix:\n{cm}")
    print(classification_report(y_test, y_pred, target_names=class_names))
    return {
        "accuracy": float(acc),
        "auc": float(auc_val),
        "y_proba": y_proba,
        "y_true": y_test,
    }


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a radiomics-based sex estimation model for patellar CT."
    )
    parser.add_argument("--img-dir", type=Path, required=True,
                        help="Directory containing preprocessed NIfTI images.")
    parser.add_argument("--mask-dir", type=Path, required=True,
                        help="Directory containing segmentation masks (NIfTI).")
    parser.add_argument("--train-csv", type=Path, required=True,
                        help="CSV with columns 'number', 'sex' for training set.")
    parser.add_argument("--test-csv", type=Path, required=True,
                        help="CSV with columns 'number', 'sex' for test set.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory to save model and results.")
    parser.add_argument("--rater-id", type=int, default=2,
                        help="Rater suffix in filenames (default: 2).")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Number of CV folds for hyperparameter tuning (default: 5).")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--pre-extracted-train", type=Path, default=None,
                        help="Load pre-extracted training features from CSV.")
    parser.add_argument("--pre-extracted-test", type=Path, default=None,
                        help="Load pre-extracted test features from CSV.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
        )

    try:
        # ---- Load or extract features ----
        if args.pre_extracted_train and args.pre_extracted_test:
            logger.info("Loading pre-extracted features...")
            X_train_raw = pd.read_csv(args.pre_extracted_train)
            X_test_raw = pd.read_csv(args.pre_extracted_test)
            train_meta = pd.read_csv(args.train_csv)
            test_meta = pd.read_csv(args.test_csv)
            y_train = train_meta["sex"].astype(int)
            y_test = test_meta["sex"].astype(int)
            logger.info("Loaded train features: %s, test features: %s",
                        X_train_raw.shape, X_test_raw.shape)
        else:
            train_meta = pd.read_csv(args.train_csv)
            test_meta = pd.read_csv(args.test_csv)
            train_meta["number"] = pd.to_numeric(train_meta["number"], errors="coerce").astype(int)
            test_meta["number"] = pd.to_numeric(test_meta["number"], errors="coerce").astype(int)

            logger.info("Extracting features for training set...")
            X_train_raw, y_train, _ = build_feature_matrices(
                train_meta, args.img_dir, args.mask_dir, id_col="number", rater_id=args.rater_id
            )
            logger.info("Extracting features for test set...")
            X_test_raw, y_test, _ = build_feature_matrices(
                test_meta, args.img_dir, args.mask_dir, id_col="number", rater_id=args.rater_id
            )

        # ---- Clean features ----
        X_train_clean, X_test_clean = clean_features(X_train_raw.copy(), X_test_raw.copy())

        # ---- Feature selection (includes standardisation) ----
        (X_train_sel, X_test_sel, spearman_keep_idx, selected_indices,
         lasso_selector, scaler) = select_features_spearman_lasso(
            X_train_clean, X_test_clean, y_train.values, random_state=args.random_state
        )

        # ---- Train final classifier (input is already standardised) ----
        logger.info("Training radiomics classifier with %d-fold CV...", args.cv_folds)
        model, grid = train_radiomics_classifier(
            X_train_sel, y_train.values, cv_folds=args.cv_folds, random_state=args.random_state
        )

        # ---- Evaluate ----
        metrics = evaluate_model(model, X_test_sel, y_test.values)

        # ---- Save artifacts (naming matches previously trained files) ----
        # 1. StandardScaler
        joblib.dump(scaler, out_dir / "scaler_56feat.pkl")
        logger.info("Scaler saved to scaler_56feat.pkl")

        # 2. Feature selection pipeline (without scaler)
        selection_pipeline = {
            "spearman_keep_idx": spearman_keep_idx,
            "selected_feature_indices": selected_indices,
            "feature_names_after_cleaning": X_train_clean.columns.tolist(),
            "lasso_selector": lasso_selector,
        }
        joblib.dump(selection_pipeline, out_dir / "lasso_selector_56feat.pkl")
        logger.info("Feature selection pipeline saved to lasso_selector_56feat.pkl")

        # 3. Trained logistic regression model
        joblib.dump(model, out_dir / "gender_prediction_model_56feat.pkl")
        logger.info("Model saved to gender_prediction_model_56feat.pkl")

        # 4. Performance summary
        info = {
            "n_selected_features": X_train_sel.shape[1],
            "test_accuracy": metrics["accuracy"],
            "test_auc": metrics["auc"],
            "best_params": grid.best_params_,
            "cv_mean_auc": float(grid.best_score_),
            "environment": {
                "sklearn_version": sklearn.__version__,
                "numpy_version": np.__version__,
                "pandas_version": pd.__version__,
            },
        }
        with open(out_dir / "radiomics_model_info.json", "w") as f:
            json.dump(info, f, indent=2)

        logger.info("All outputs saved to %s", out_dir)
        print("\nDone.")
        return 0

    except Exception as e:
        logger.exception("Fatal error: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())