#!/usr/bin/env python3
"""
Train the weighted ensemble model described in the paper.

The ensemble combines four base models via weighted averaging of predicted
probabilities:
  1. Conventional morphometric model (logistic regression)
  2. Radiomics model (logistic regression on selected features)
  3. 2D ResNet‑50 (fine‑tuned on ImageNet)
  4. 3D ResNet‑50 (fine‑tuned on MedicalNet)

Weights are optimised on the internal training set using coarse‑to‑fine grid
search over the probability simplex, maximising AUC.  The optimal decision
threshold is determined by maximising the Youden index on the training set.
The final ensemble is evaluated on the internal test set and saved as a
single `ensemble_predictor.pkl` artefact.

Requirements: torch, torchvision, nibabel, pandas, numpy, scikit‑learn,
               scipy, tqdm, joblib, Pillow.
"""

import argparse
import itertools
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm
from PIL import Image
import nibabel as nib
from scipy.ndimage import zoom

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Image preprocessing (identical to training scripts)
# ---------------------------------------------------------------------------
def load_nii(path: Path) -> np.ndarray:
    return nib.load(str(path)).get_fdata().astype(np.float32)


def get_bbox_from_mask(mask: np.ndarray, margin: int = 5) -> Optional[Tuple[slice, ...]]:
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return None
    z_min = max(0, coords[:, 0].min() - margin)
    z_max = min(mask.shape[0], coords[:, 0].max() + margin + 1)
    y_min = max(0, coords[:, 1].min() - margin)
    y_max = min(mask.shape[1], coords[:, 1].max() + margin + 1)
    x_min = max(0, coords[:, 2].min() - margin)
    x_max = min(mask.shape[2], coords[:, 2].max() + margin + 1)
    return slice(z_min, z_max), slice(y_min, y_max), slice(x_min, x_max)


def max_intensity_projection(vol: np.ndarray, axis: int) -> np.ndarray:
    return np.max(vol, axis=axis)


def resize_to_target(volume: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
    factors = [t / s for s, t in zip(volume.shape, target_size)]
    resized = zoom(volume, factors, order=1)
    out = np.zeros(target_size, dtype=np.float32)
    crop_slices, out_slices = [], []
    for dim in range(3):
        if resized.shape[dim] >= target_size[dim]:
            start = (resized.shape[dim] - target_size[dim]) // 2
            crop_slices.append(slice(start, start + target_size[dim]))
            out_slices.append(slice(None))
        else:
            crop_slices.append(slice(None))
            pad_before = (target_size[dim] - resized.shape[dim]) // 2
            out_slices.append(slice(pad_before, pad_before + resized.shape[dim]))
    resized_cropped = resized[tuple(crop_slices)]
    out[tuple(out_slices)] = resized_cropped
    return out


# ---------------------------------------------------------------------------
# Datasets for inference (no data augmentation)
# ---------------------------------------------------------------------------
class Knee2DDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        ct_dir: Path,
        mask_dir: Path,
        seg_id: int = 2,
        target_size: Tuple[int, int] = (224, 224),
        transform: Optional[transforms.Compose] = None,
    ):
        self.df = df.copy()
        self.df["number"] = self.df["number"].astype(str).str.strip()
        self.ct_dir = ct_dir
        self.mask_dir = mask_dir
        self.seg_id = seg_id
        self.target_size = target_size
        self.transform = transform
        self.samples = []
        for _, row in self.df.iterrows():
            sid = row["number"]
            ct_path = ct_dir / f"{sid}-{seg_id}.nii.gz"
            mask_path = mask_dir / f"{sid}-{seg_id}.nii.gz"
            if ct_path.exists() and mask_path.exists():
                label = int(row["sex"])
                self.samples.append((sid, ct_path, mask_path, label))
            else:
                logger.warning("Skipping 2D sample %s: file missing", sid)
        self.df = self.df[self.df["number"].isin([s[0] for s in self.samples])].reset_index(drop=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sid, ct_path, mask_path, label = self.samples[idx]
        ct = load_nii(ct_path)
        mask = load_nii(mask_path)
        ct = np.clip(ct * 255.0, 0, 255).astype(np.uint8)
        bbox = get_bbox_from_mask(mask)
        if bbox is None:
            raise ValueError(f"Empty mask for {sid}")
        roi = ct[bbox] * (mask[bbox] > 0)
        mip_z = max_intensity_projection(roi, 0)
        mip_y = max_intensity_projection(roi, 1)
        mip_x = max_intensity_projection(roi, 2)
        resize_fn = transforms.Resize(self.target_size)
        to_pil = lambda arr: Image.fromarray(arr)
        channels = [
            np.array(resize_fn(to_pil(mip))) for mip in (mip_z, mip_y, mip_x)
        ]
        combined = np.stack(channels, axis=-1)
        img_pil = Image.fromarray(combined.astype(np.uint8))
        if self.transform:
            img_pil = self.transform(img_pil)
        return img_pil, torch.tensor(label, dtype=torch.long)


class Knee3DDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        ct_dir: Path,
        mask_dir: Path,
        target_size: Tuple[int, int, int] = (64, 64, 64),
        seg_id: int = 2,
    ):
        self.df = df.copy()
        self.df["number"] = self.df["number"].astype(str).str.strip()
        self.ct_dir = ct_dir
        self.mask_dir = mask_dir
        self.target_size = target_size
        self.seg_id = seg_id
        self.samples = []
        for _, row in self.df.iterrows():
            sid = row["number"]
            ct_path = ct_dir / f"{sid}-{seg_id}.nii.gz"
            mask_path = mask_dir / f"{sid}-{seg_id}.nii.gz"
            if ct_path.exists() and mask_path.exists():
                label = int(row["sex"])
                self.samples.append((sid, ct_path, mask_path, label))
            else:
                logger.warning("Skipping 3D sample %s: file missing", sid)
        self.df = self.df[self.df["number"].isin([s[0] for s in self.samples])].reset_index(drop=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sid, ct_path, mask_path, label = self.samples[idx]
        ct = load_nii(ct_path)
        mask = load_nii(mask_path)
        bbox = get_bbox_from_mask(mask)
        roi = ct[bbox] * (mask[bbox] > 0)
        resized = resize_to_target(roi, self.target_size)
        vol = torch.from_numpy(resized).unsqueeze(0).float()
        return vol, torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_traditional_model(pipeline_path: Path) -> object:
    return joblib.load(pipeline_path)


def load_radiomics_model(
    scaler_path: Path, lasso_path: Path, model_path: Path
) -> Tuple:
    scaler = joblib.load(scaler_path)
    lasso_pkg = joblib.load(lasso_path)
    lasso_selector = lasso_pkg["lasso_selector"]
    spearman_idx = lasso_pkg["spearman_keep_idx"]
    model = joblib.load(model_path)
    return scaler, lasso_selector, spearman_idx, model


def load_2d_model(state_path: Path, device: torch.device) -> nn.Module:
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def load_3d_model(state_path: Path, device: torch.device) -> nn.Module:
    try:
        from models import resnet as medicalnet_resnet
    except ImportError:
        raise ImportError(
            "MedicalNet not found. Clone from https://github.com/Tencent/MedicalNet"
        )
    backbone = medicalnet_resnet.resnet50(
        sample_input_W=64, sample_input_H=64, sample_input_D=64,
        num_seg_classes=1,
    )
    class MedicalNetGender(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
            self.fc = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, 1),
            )
        def forward(self, x):
            x = self.backbone.conv1(x); x = self.backbone.bn1(x); x = self.backbone.relu(x)
            x = self.backbone.maxpool(x); x = self.backbone.layer1(x); x = self.backbone.layer2(x)
            x = self.backbone.layer3(x); x = self.backbone.layer4(x)
            pooled = self.global_avg_pool(x).flatten(1)
            return self.fc(pooled)
    model = MedicalNetGender(backbone)
    state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Inference (predict probability for each sample)
# ---------------------------------------------------------------------------
def predict_traditional(pipeline, df: pd.DataFrame, feat_columns: List[str]) -> np.ndarray:
    X = df[feat_columns].values
    return pipeline.predict_proba(X)[:, 1]


def predict_radiomics(
    scaler, lasso_selector, spearman_idx, model,
    df: pd.DataFrame, omics_columns: List[str]
) -> np.ndarray:
    # Correct order: 1) Spearman filter, 2) scale, 3) LASSO selection
    X = df[omics_columns].values.astype(np.float32)
    X_spearman = X[:, spearman_idx]               # 1. select features
    X_scaled = scaler.transform(X_spearman)       # 2. standardise
    X_selected = lasso_selector.transform(X_scaled)  # 3. LASSO selection
    return model.predict_proba(X_selected)[:, 1]


def predict_2d(
    model: nn.Module, device: torch.device, df: pd.DataFrame,
    ct_dir: Path, mask_dir: Path, seg_id: int,
    target_size: Tuple[int, int], transform: transforms.Compose,
    batch_size: int = 32
) -> np.ndarray:
    ds = Knee2DDataset(df, ct_dir, mask_dir, seg_id, target_size, transform)
    if len(ds) == 0:
        logger.warning("No valid 2D samples, returning 0.5")
        return np.full(len(df), 0.5)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    probs = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            outputs = model(images)
            prob = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            probs.extend(prob)
    prob_dict = {ds.samples[i][0]: probs[i] for i in range(len(probs))}
    return np.array([prob_dict.get(str(row["number"]), 0.5) for _, row in df.iterrows()])


def predict_3d(
    model: nn.Module, device: torch.device, df: pd.DataFrame,
    ct_dir: Path, mask_dir: Path, seg_id: int,
    target_size: Tuple[int, int, int], batch_size: int = 4
) -> np.ndarray:
    ds = Knee3DDataset(df, ct_dir, mask_dir, target_size, seg_id)
    if len(ds) == 0:
        logger.warning("No valid 3D samples, returning 0.5")
        return np.full(len(df), 0.5)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    probs = []
    with torch.no_grad():
        for volumes, _ in loader:
            volumes = volumes.to(device)
            outputs = model(volumes).squeeze(1)
            prob = torch.sigmoid(outputs).cpu().numpy()
            probs.extend(prob)
    prob_dict = {ds.samples[i][0]: probs[i] for i in range(len(probs))}
    return np.array([prob_dict.get(str(row["number"]), 0.5) for _, row in df.iterrows()])


# ---------------------------------------------------------------------------
# Weight optimisation
# ---------------------------------------------------------------------------
def grid_search_weights(X: np.ndarray, y: np.ndarray, step: float = 0.05) -> Tuple[np.ndarray, float]:
    n = X.shape[1]
    best_auc, best_w = 0.0, None
    for combo in itertools.product(np.arange(0, 1+step, step), repeat=n-1):
        w = list(combo)
        last = 1.0 - sum(w)
        if last < 0 or last > 1:
            continue
        weights = np.array(w + [last])
        auc = roc_auc_score(y, X @ weights)
        if auc > best_auc:
            best_auc, best_w = auc, weights
    return best_w, best_auc


def refine_weights(X, y, initial_w, step=0.01, radius=0.1):
    n = len(initial_w)
    ranges = [np.arange(max(0, w-radius), min(1, w+radius)+step, step) for w in initial_w]
    best_auc, best_w = 0.0, initial_w.copy()
    for combo in itertools.product(*ranges[:-1]):
        w = list(combo)
        last = 1.0 - sum(w)
        if last < 0 or last > 1:
            continue
        weights = np.array(w + [last])
        auc = roc_auc_score(y, X @ weights)
        if auc > best_auc:
            best_auc, best_w = auc, weights
    return best_w, best_auc


# ---------------------------------------------------------------------------
# Ensemble predictor (serialisable with CPU models only)
# ---------------------------------------------------------------------------
class EnsemblePredictor:
    def __init__(self, weights, threshold,
                 trad_pipeline, trad_cols,
                 omics_scaler, omics_lasso, spearman_idx, omics_model, omics_cols,
                 model_2d, model_3d, pre_cfg_2d,
                 ct_dir=None, mask_dir=None, seg_id=2,
                 target_size_2d=(224, 224), target_size_3d=(64, 64, 64),
                 device="cpu"):
        self.weights = weights
        self.threshold = threshold
        self.trad_pipeline = trad_pipeline
        self.trad_cols = trad_cols
        self.omics_scaler = omics_scaler
        self.omics_lasso = omics_lasso
        self.spearman_idx = spearman_idx
        self.omics_model = omics_model
        self.omics_cols = omics_cols
        # Store only state_dicts for PyTorch models to avoid pickling issues
        self.model_2d_state = model_2d.cpu().state_dict()
        self.model_3d_state = model_3d.cpu().state_dict()
        self.pre_cfg_2d = pre_cfg_2d
        self.transform_2d = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=pre_cfg_2d["mean"], std=pre_cfg_2d["std"]),
        ])
        self.ct_dir = ct_dir
        self.mask_dir = mask_dir
        self.seg_id = seg_id
        self.target_size_2d = target_size_2d
        self.target_size_3d = target_size_3d
        self.device = torch.device(device)

        # Reconstruct models on CPU (lightweight)
        self._recreate_models()

    def _recreate_models(self):
        # 2D model
        self.model_2d = models.resnet50(weights=None)
        self.model_2d.fc = nn.Linear(self.model_2d.fc.in_features, 2)
        self.model_2d.load_state_dict(self.model_2d_state)
        self.model_2d.to(self.device).eval()
        # 3D model (requires MedicalNet)
        try:
            from models import resnet as medicalnet_resnet
        except ImportError:
            raise ImportError("MedicalNet not found.")
        backbone = medicalnet_resnet.resnet50(
            sample_input_W=64, sample_input_H=64, sample_input_D=64,
            num_seg_classes=1,
        )
        class MedicalNetGender(nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.backbone = backbone
                self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
                self.fc = nn.Sequential(
                    nn.Linear(2048, 512), nn.ReLU(inplace=True),
                    nn.Dropout(0.3), nn.Linear(512, 1))
            def forward(self, x):
                x = self.backbone.conv1(x); x = self.backbone.bn1(x); x = self.backbone.relu(x)
                x = self.backbone.maxpool(x); x = self.backbone.layer1(x); x = self.backbone.layer2(x)
                x = self.backbone.layer3(x); x = self.backbone.layer4(x)
                pooled = self.global_avg_pool(x).flatten(1)
                return self.fc(pooled)
        self.model_3d = MedicalNetGender(backbone)
        self.model_3d.load_state_dict(self.model_3d_state)
        self.model_3d.to(self.device).eval()

    def predict(self, df, ct_dir=None, mask_dir=None):
        ct_dir = Path(ct_dir) if ct_dir else Path(self.ct_dir)
        mask_dir = Path(mask_dir) if mask_dir else Path(self.mask_dir)
        prob_trad = predict_traditional(self.trad_pipeline, df, self.trad_cols)
        prob_omics = predict_radiomics(self.omics_scaler, self.omics_lasso,
                                       self.spearman_idx, self.omics_model,
                                       df, self.omics_cols)
        prob_2d = predict_2d(self.model_2d, self.device, df, ct_dir, mask_dir,
                             self.seg_id, self.target_size_2d, self.transform_2d)
        prob_3d = predict_3d(self.model_3d, self.device, df, ct_dir, mask_dir,
                             self.seg_id, self.target_size_3d)
        all_probs = np.stack([prob_trad, prob_omics, prob_2d, prob_3d], axis=1)
        ensemble_prob = all_probs @ self.weights
        pred = (ensemble_prob >= self.threshold).astype(int)
        return ensemble_prob, pred


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the weighted ensemble for patellar sex estimation."
    )
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--train-measure", type=Path, required=True)
    parser.add_argument("--test-measure", type=Path, required=True)
    parser.add_argument("--train-omics", type=Path, required=True)
    parser.add_argument("--test-omics", type=Path, required=True)
    parser.add_argument("--ct-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--trad-pipeline", type=Path, required=True)
    parser.add_argument("--omics-scaler", type=Path, required=True)
    parser.add_argument("--omics-lasso", type=Path, required=True)
    parser.add_argument("--omics-model", type=Path, required=True)
    parser.add_argument("--model-2d-state", type=Path, required=True)
    parser.add_argument("--model-2d-cfg", type=Path, required=True)
    parser.add_argument("--model-3d-state", type=Path, required=True)
    parser.add_argument("--trad-columns", nargs="+", type=str,
                        default=["Patellar length", "Patellar width",
                                 "Patellar thickness", "Patellar volume",
                                 "Patellar surface area", "Patellar coronal perimeter"])
    parser.add_argument("--omics-columns", nargs="*", default=None,
                        help="Radiomics column names (if not provided, auto‑detect from training CSV)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seg-id", type=int, default=2)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    device = torch.device(args.device)
    logger.info("Device: %s", device)

    # ---- Load base models ----
    trad_pipeline = load_traditional_model(args.trad_pipeline)
    omics_scaler, omics_lasso, spearman_idx, omics_model = load_radiomics_model(
        args.omics_scaler, args.omics_lasso, args.omics_model)
    with open(args.model_2d_cfg, "r") as f:
        pre_cfg_2d = json.load(f)
    model_2d = load_2d_model(args.model_2d_state, device)
    model_3d = load_3d_model(args.model_3d_state, device)

    # ---- Read metadata and feature tables ----
    train_meta = pd.read_csv(args.train_csv)
    test_meta = pd.read_csv(args.test_csv)
    for df in (train_meta, test_meta):
        df["number"] = df["number"].astype(str).str.strip()

    train_meas = pd.read_csv(args.train_measure)
    test_meas = pd.read_csv(args.test_measure)
    train_omics = pd.read_csv(args.train_omics, encoding="utf-8-sig")
    test_omics = pd.read_csv(args.test_omics, encoding="utf-8-sig")

    # Traditional columns (exact English names expected)
    trad_actual = []
    for col in args.trad_columns:
        if col in train_meas.columns:
            trad_actual.append(col)
        else:
            raise KeyError(f"Traditional column '{col}' not found in measurements CSV")

    # Radiomics columns: auto‑detect if not specified
    if args.omics_columns is None or len(args.omics_columns) == 0:
        omics_cols = [c for c in train_omics.columns if c not in ("number", "sex", "dataset")]
    else:
        omics_cols = args.omics_columns

    # Merge metadata with features
    train_all = train_meta.merge(train_meas[["number"] + trad_actual], on="number", how="inner")
    test_all = test_meta.merge(test_meas[["number"] + trad_actual], on="number", how="inner")
    train_all = train_all.merge(train_omics[["number"] + omics_cols], on="number", how="inner")
    test_all = test_all.merge(test_omics[["number"] + omics_cols], on="number", how="inner")

    logger.info("After merging: train set %d samples, test set %d samples",
                len(train_all), len(test_all))

    # Build transform for 2D inference
    transform_2d = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=pre_cfg_2d["mean"], std=pre_cfg_2d["std"]),
    ])

    # ---- Generate probability matrices ----
    logger.info("Inference on training set...")
    train_probs = np.column_stack([
        predict_traditional(trad_pipeline, train_all, trad_actual),
        predict_radiomics(omics_scaler, omics_lasso, spearman_idx, omics_model,
                          train_all, omics_cols),
        predict_2d(model_2d, device, train_all, args.ct_dir, args.mask_dir,
                   args.seg_id, (224, 224), transform_2d),
        predict_3d(model_3d, device, train_all, args.ct_dir, args.mask_dir,
                   args.seg_id, (64, 64, 64)),
    ])
    y_train = train_all["sex"].values

    logger.info("Inference on test set...")
    test_probs = np.column_stack([
        predict_traditional(trad_pipeline, test_all, trad_actual),
        predict_radiomics(omics_scaler, omics_lasso, spearman_idx, omics_model,
                          test_all, omics_cols),
        predict_2d(model_2d, device, test_all, args.ct_dir, args.mask_dir,
                   args.seg_id, (224, 224), transform_2d),
        predict_3d(model_3d, device, test_all, args.ct_dir, args.mask_dir,
                   args.seg_id, (64, 64, 64)),
    ])
    y_test = test_all["sex"].values

    # ---- Optimise weights ----
    logger.info("Optimising ensemble weights on training set...")
    init_weights, _ = grid_search_weights(train_probs, y_train, step=0.05)
    final_weights, train_auc = refine_weights(train_probs, y_train, init_weights,
                                              step=0.01, radius=0.1)
    logger.info("Weights: %s", dict(zip(
        ["Traditional", "Radiomics", "2D_CNN", "3D_CNN"], final_weights)))
    logger.info("Training AUC: %.4f", train_auc)

    # Optimal threshold
    ensemble_train = train_probs @ final_weights
    fpr, tpr, thresholds = roc_curve(y_train, ensemble_train)
    best_thr = thresholds[np.argmax(tpr - fpr)]
    logger.info("Optimal threshold: %.4f", best_thr)

    # ---- Evaluate on test set ----
    ensemble_test = test_probs @ final_weights
    y_pred = (ensemble_test >= best_thr).astype(int)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, ensemble_test)
    sens = recall_score(y_test, y_pred)
    spec = recall_score(y_test, y_pred, pos_label=0)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    logger.info("Ensemble test performance:")
    logger.info("  Accuracy: %.4f", acc)
    logger.info("  AUC:      %.4f", auc)
    logger.info("  Sensitivity: %.4f", sens)
    logger.info("  Specificity: %.4f", spec)
    logger.info("  Precision: %.4f", prec)
    logger.info("  F1:        %.4f", f1)

    # ---- Save EnsemblePredictor (CPU‑safe) ----
    predictor = EnsemblePredictor(
        weights=final_weights, threshold=best_thr,
        trad_pipeline=trad_pipeline, trad_cols=trad_actual,
        omics_scaler=omics_scaler, omics_lasso=omics_lasso,
        spearman_idx=spearman_idx, omics_model=omics_model,
        omics_cols=omics_cols,
        model_2d=model_2d, model_3d=model_3d,
        pre_cfg_2d=pre_cfg_2d,
        ct_dir=args.ct_dir, mask_dir=args.mask_dir,
        seg_id=args.seg_id,
        device=str(device),
    )
    joblib.dump(predictor, args.output_dir / "ensemble_predictor.pkl")
    logger.info("Saved ensemble_predictor.pkl to %s", args.output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())