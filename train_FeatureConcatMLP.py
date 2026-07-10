#!/usr/bin/env python3
"""
Train the Feature‑Concatenation MLP baseline described in the paper.

The model concatenates feature vectors from four modalities:
  1. Conventional morphometric parameters (6 features)
  2. Radiomic features (after Spearman + LASSO selection)
  3. 2048‑d 2D CNN features reduced to 256 via PCA
  4. 2048‑d 3D CNN features reduced to 256 via PCA

The concatenated vector is fed into a three‑layer MLP (128 → 64 units,
batch normalisation, ReLU, dropout 0.7) with a softmax output.
Training uses class‑balanced focal loss (γ = 2.0) under 5‑fold
cross‑validation.  The final artefact saved is
``feature_concat_mlp.pkl`` which contains the ensemble of
fold models and all preprocessing objects needed for inference.

Requirements: torch, torchvision, nibabel, pandas, numpy, scikit‑learn,
               scipy, tqdm, joblib, Pillow.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Image‑related imports (for feature extraction only, no augmentation)
import nibabel as nib
from scipy.ndimage import zoom
from torchvision import models as tv_models, transforms
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image preprocessing (identical to training scripts, no augmentation)
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
# Datasets for feature extraction only (no augmentation)
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
                logger.warning("Skipping 2D sample %s – file missing", sid)
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
        channels = [np.array(resize_fn(to_pil(mip))) for mip in (mip_z, mip_y, mip_x)]
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
                logger.warning("Skipping 3D sample %s – file missing", sid)
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
# Feature extractors (pretrained 2D/3D CNNs)
# ---------------------------------------------------------------------------
class FeatureExtractor2D(nn.Module):
    def __init__(self, weight_path: Path):
        super().__init__()
        base = tv_models.resnet50(weights=None)
        base.fc = nn.Linear(base.fc.in_features, 2)
        state = torch.load(weight_path, map_location="cpu")
        base.load_state_dict(state)
        self.conv1, self.bn1, self.relu, self.maxpool = base.conv1, base.bn1, base.relu, base.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = base.layer1, base.layer2, base.layer3, base.layer4
        self.avgpool = base.avgpool
        self.feat_dim = base.fc.in_features
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return self.avgpool(x).flatten(1)


class FeatureExtractor3D(nn.Module):
    def __init__(self, weight_path: Path):
        super().__init__()
        try:
            from models import resnet as medicalnet_resnet
        except ImportError:
            raise ImportError("MedicalNet not found. Clone from https://github.com/Tencent/MedicalNet")
        backbone = medicalnet_resnet.resnet50(
            sample_input_W=64, sample_input_H=64, sample_input_D=64, num_seg_classes=1
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
                x = self.backbone.maxpool(x)
                x = self.backbone.layer1(x); x = self.backbone.layer2(x); x = self.backbone.layer3(x); x = self.backbone.layer4(x)
                pooled = self.global_avg_pool(x).flatten(1)
                return self.fc(pooled)
        base = MedicalNetGender(backbone)
        state = torch.load(weight_path, map_location="cpu")
        base.load_state_dict(state)
        self.backbone = base.backbone
        self.global_avg_pool = base.global_avg_pool
        self.feat_dim = base.fc[0].in_features  # 2048
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(x); x = self.backbone.bn1(x); x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x); x = self.backbone.layer2(x); x = self.backbone.layer3(x); x = self.backbone.layer4(x)
        return self.global_avg_pool(x).flatten(1)


# ---------------------------------------------------------------------------
# Feature extraction utility
# ---------------------------------------------------------------------------
def extract_deep_features(
    dataset: Dataset, model: nn.Module, device: torch.device, batch_size: int = 32, desc: str = ""
) -> Tuple[np.ndarray, List[str]]:
    """Extract deep features and return (features_array, list_of_sample_ids)."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    features = []
    ids = [s[0] for s in dataset.samples]
    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc=desc):
            imgs = imgs.to(device)
            feat = model(imgs).cpu().numpy()
            features.append(feat)
    return np.vstack(features), ids


# ---------------------------------------------------------------------------
# Radiomic feature selection (corrected pipeline)
# ---------------------------------------------------------------------------
def select_features_spearman_lasso(
    X_train: np.ndarray,
    y_train: np.ndarray,
    spearman_threshold: float = 0.9,
    cv_folds: int = 5,
    random_state: int = 42,
) -> Tuple[
    np.ndarray,          # constant_feature_mask
    np.ndarray,          # spearman_keep_idx
    StandardScaler,      # scaler
    SelectFromModel,     # LASSO selector
]:
    # 1. Remove constant features
    const_mask = X_train.var(axis=0) > 1e-10
    X_train_const = X_train[:, const_mask]

    # 2. Spearman correlation
    corr = pd.DataFrame(X_train_const).corr(method="spearman").abs().values
    high_corr = np.where(corr >= spearman_threshold)
    remove = set()
    for i, j in zip(*high_corr):
        if i != j and i < j:
            remove.add(j)
    spearman_keep_idx = np.array([i for i in range(X_train_const.shape[1]) if i not in remove])
    X_train_corr = X_train_const[:, spearman_keep_idx]

    # 3. Standardisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_corr)

    # 4. LASSO selection
    lasso = LassoCV(
        cv=cv_folds,
        random_state=random_state,
        max_iter=2000,
        n_jobs=-1,
    )
    lasso.fit(X_train_scaled, y_train)
    selector = SelectFromModel(lasso, prefit=True)

    return const_mask, spearman_keep_idx, scaler, selector


def apply_radiomic_pipeline(
    X: np.ndarray,
    const_mask: np.ndarray,
    spearman_keep_idx: np.ndarray,
    scaler: StandardScaler,
    selector: SelectFromModel,
) -> np.ndarray:
    """Apply the fitted radiomic preprocessing pipeline to new data."""
    X = X[:, const_mask]
    X = X[:, spearman_keep_idx]
    X = scaler.transform(X)
    return selector.transform(X)


# ---------------------------------------------------------------------------
# MLP model
# ---------------------------------------------------------------------------
class FeatureConcatMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64], dropout: float = 0.7):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class FocalLoss(nn.Module):
    """
    Class‑balanced focal loss (γ = 2.0).
    Alpha is computed as total_samples / (num_classes * class_counts)
    to give higher weight to minority class.
    """
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss
        return focal_loss.mean()


# ---------------------------------------------------------------------------
# Training & evaluation utilities
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    clip_grad: float = 1.0,
) -> Tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for feats, labels in loader:
        feats, labels = feats.to(device), labels.to(device)
        optimizer.zero_grad()
        # Enable autocast only when scaler is active
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            out = model(feats)
            loss = criterion(out, labels)
        scaler.scale(loss).backward()
        if clip_grad > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * labels.size(0)
        pred = out.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_labels, all_probs = [], []
    for feats, labels in loader:
        feats, labels = feats.to(device), labels.to(device)
        out = model(feats)
        loss = criterion(out, labels)
        total_loss += loss.item() * labels.size(0)
        prob = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
        pred = out.argmax(dim=1).cpu().numpy()
        correct += (pred == labels.cpu().numpy()).sum()
        total += labels.size(0)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(prob)
    avg_loss = total_loss / total
    acc = correct / total
    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
    return avg_loss, acc, auc, np.array(all_labels), np.array(all_probs)


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Feature‑Concatenation MLP for patellar sex estimation."
    )
    # Data
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--train-measure", type=Path, required=True)
    parser.add_argument("--test-measure", type=Path, required=True)
    parser.add_argument("--train-omics", type=Path, required=True)
    parser.add_argument("--test-omics", type=Path, required=True)
    parser.add_argument("--ct-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    # Pre‑trained models
    parser.add_argument("--model-2d-state", type=Path, required=True)
    parser.add_argument("--model-2d-cfg", type=Path, required=True)
    parser.add_argument("--model-3d-state", type=Path, required=True)
    # Column names
    parser.add_argument("--measure-cols", nargs="+", type=str,
                        default=["Patellar length", "Patellar width", "Patellar thickness",
                                 "Patellar volume", "Patellar surface area", "Patellar coronal perimeter"])
    parser.add_argument("--omics-cols", nargs="*", default=None,
                        help="Radiomic columns (if not given, auto‑detect)")
    # Hyper‑parameters
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[128, 64])
    parser.add_argument("--dropout", type=float, default=0.7)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--spearman-threshold", type=float, default=0.9)
    parser.add_argument("--pca-dim-2d", type=int, default=256)
    parser.add_argument("--pca-dim-3d", type=int, default=256)
    # Output
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seg-id", type=int, default=2)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    device = torch.device(args.device)
    logger.info("Device: %s", device)

    # -------------------------------------------------------------------
    # Load metadata and feature tables
    # -------------------------------------------------------------------
    train_meta = pd.read_csv(args.train_csv)
    test_meta = pd.read_csv(args.test_csv)
    for df in (train_meta, test_meta):
        df["number"] = df["number"].astype(str).str.strip()

    train_meas = pd.read_csv(args.train_measure)
    test_meas = pd.read_csv(args.test_measure)
    train_omics = pd.read_csv(args.train_omics, encoding="utf-8-sig")
    test_omics = pd.read_csv(args.test_omics, encoding="utf-8-sig")

    meas_cols = []
    for col in args.measure_cols:
        if col in train_meas.columns:
            meas_cols.append(col)
        else:
            raise KeyError(f"Measurement column '{col}' not found in CSV")

    if args.omics_cols is None or len(args.omics_cols) == 0:
        omics_cols = [c for c in train_omics.columns if c not in ("number", "sex", "dataset")]
    else:
        omics_cols = args.omics_cols

    # Merge metadata with features
    train_all = train_meta.merge(train_meas[["number"] + meas_cols], on="number", how="inner")
    train_all = train_all.merge(train_omics[["number"] + omics_cols], on="number", how="inner")
    test_all = test_meta.merge(test_meas[["number"] + meas_cols], on="number", how="inner")
    test_all = test_all.merge(test_omics[["number"] + omics_cols], on="number", how="inner")

    logger.info("Merged data: train %d, test %d", len(train_all), len(test_all))

    # -------------------------------------------------------------------
    # Load 2D transform config
    # -------------------------------------------------------------------
    with open(args.model_2d_cfg, "r") as f:
        pre_cfg = json.load(f)
    tfm_2d = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=pre_cfg["mean"], std=pre_cfg["std"]),
    ])

    # -------------------------------------------------------------------
    # Create initial datasets (they filter out missing samples internally)
    # -------------------------------------------------------------------
    ds2d_tr = Knee2DDataset(train_all, args.ct_dir, args.mask_dir, args.seg_id, (224, 224), tfm_2d)
    ds3d_tr = Knee3DDataset(train_all, args.ct_dir, args.mask_dir, (64, 64, 64), args.seg_id)
    ds2d_test = Knee2DDataset(test_all, args.ct_dir, args.mask_dir, args.seg_id, (224, 224), tfm_2d)
    ds3d_test = Knee3DDataset(test_all, args.ct_dir, args.mask_dir, (64, 64, 64), args.seg_id)

    # -------------------------------------------------------------------
    # Ensure only samples present in ALL modalities are used
    # -------------------------------------------------------------------
    def get_valid_ids(df_all, *datasets):
        """Return subset of df_all containing only IDs that appear in all datasets."""
        id_sets = [set(sid for sid, _, _, _ in ds.samples) for ds in datasets]
        common_ids = set.intersection(*id_sets) if id_sets else set()
        return df_all[df_all["number"].isin(common_ids)].reset_index(drop=True)

    train_all = get_valid_ids(train_all, ds2d_tr, ds3d_tr)
    test_all = get_valid_ids(test_all, ds2d_test, ds3d_test)
    logger.info("After aligning samples across modalities: train %d, test %d", len(train_all), len(test_all))

    # Rebuild datasets with the filtered dataframes
    ds2d_tr = Knee2DDataset(train_all, args.ct_dir, args.mask_dir, args.seg_id, (224, 224), tfm_2d)
    ds3d_tr = Knee3DDataset(train_all, args.ct_dir, args.mask_dir, (64, 64, 64), args.seg_id)
    ds2d_test = Knee2DDataset(test_all, args.ct_dir, args.mask_dir, args.seg_id, (224, 224), tfm_2d)
    ds3d_test = Knee3DDataset(test_all, args.ct_dir, args.mask_dir, (64, 64, 64), args.seg_id)

    # -------------------------------------------------------------------
    # Extract deep features
    # -------------------------------------------------------------------
    ext_2d = FeatureExtractor2D(args.model_2d_state).to(device)
    ext_3d = FeatureExtractor3D(args.model_3d_state).to(device)

    logger.info("Extracting 2D CNN features (training)...")
    X_2d_tr, ids_2d_tr = extract_deep_features(ds2d_tr, ext_2d, device, batch_size=args.batch_size, desc="2D train")
    logger.info("Extracting 3D CNN features (training)...")
    X_3d_tr, ids_3d_tr = extract_deep_features(ds3d_tr, ext_3d, device, batch_size=args.batch_size, desc="3D train")

    logger.info("Extracting 2D CNN features (test)...")
    X_2d_test, ids_2d_test = extract_deep_features(ds2d_test, ext_2d, device, batch_size=args.batch_size, desc="2D test")
    logger.info("Extracting 3D CNN features (test)...")
    X_3d_test, ids_3d_test = extract_deep_features(ds3d_test, ext_3d, device, batch_size=args.batch_size, desc="3D test")

    # -------------------------------------------------------------------
    # Conventional and radiomic features
    # -------------------------------------------------------------------
    X_cli_tr = train_all[meas_cols].values.astype(np.float32)
    X_rad_tr = train_all[omics_cols].values.astype(np.float32)
    y_tr = train_all["sex"].values

    X_cli_test = test_all[meas_cols].values.astype(np.float32)
    X_rad_test = test_all[omics_cols].values.astype(np.float32)
    y_test = test_all["sex"].values

    # -------------------------------------------------------------------
    # Preprocessing
    # -------------------------------------------------------------------
    scaler_cli = StandardScaler().fit(X_cli_tr)
    X_cli_tr = scaler_cli.transform(X_cli_tr)
    X_cli_test = scaler_cli.transform(X_cli_test)

    const_mask, spearman_keep_idx, rad_scaler, rad_selector = select_features_spearman_lasso(
        X_rad_tr, y_tr,
        spearman_threshold=args.spearman_threshold,
        cv_folds=args.cv_folds,
        random_state=args.seed,
    )
    X_rad_tr_sel = apply_radiomic_pipeline(X_rad_tr, const_mask, spearman_keep_idx, rad_scaler, rad_selector)
    X_rad_test_sel = apply_radiomic_pipeline(X_rad_test, const_mask, spearman_keep_idx, rad_scaler, rad_selector)

    # -------------------------------------------------------------------
    # PCA for deep features
    # -------------------------------------------------------------------
    pca_2d = PCA(n_components=min(args.pca_dim_2d, X_2d_tr.shape[1]), random_state=args.seed)
    pca_3d = PCA(n_components=min(args.pca_dim_3d, X_3d_tr.shape[1]), random_state=args.seed)
    X_2d_tr_pca = pca_2d.fit_transform(X_2d_tr)
    X_3d_tr_pca = pca_3d.fit_transform(X_3d_tr)
    X_2d_test_pca = pca_2d.transform(X_2d_test)
    X_3d_test_pca = pca_3d.transform(X_3d_test)

    # -------------------------------------------------------------------
    # Concatenate all features
    # -------------------------------------------------------------------
    X_train = np.hstack([X_cli_tr, X_rad_tr_sel, X_2d_tr_pca, X_3d_tr_pca])
    X_test = np.hstack([X_cli_test, X_rad_test_sel, X_2d_test_pca, X_3d_test_pca])
    input_dim = X_train.shape[1]
    logger.info("Final concatenated feature dimension: %d", input_dim)

    # -------------------------------------------------------------------
    # 5‑fold cross‑validation
    # -------------------------------------------------------------------
    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    fold_states = []
    fold_probs_test = []
    fold_thresholds = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_tr)):
        logger.info("=== Fold %d/%d ===", fold+1, args.cv_folds)
        X_f_tr, X_f_val = X_train[tr_idx], X_train[val_idx]
        y_f_tr, y_f_val = y_tr[tr_idx], y_tr[val_idx]

        class_counts = np.bincount(y_f_tr)
        if len(class_counts) < 2:
            logger.warning("Only one class in training fold, skipping")
            continue
        alpha = torch.tensor(
            class_counts.sum() / (len(class_counts) * class_counts),
            device=device, dtype=torch.float
        )

        train_set = TensorDataset(torch.from_numpy(X_f_tr).float(), torch.from_numpy(y_f_tr).long())
        val_set = TensorDataset(torch.from_numpy(X_f_val).float(), torch.from_numpy(y_f_val).long())
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

        model = FeatureConcatMLP(input_dim, hidden_dims=args.hidden_dims, dropout=args.dropout).to(device)
        criterion = FocalLoss(gamma=args.focal_gamma, alpha=alpha)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
        scaler = torch.cuda.amp.GradScaler()

        best_val_auc = 0.0
        best_state = None
        patience_counter = 0
        for epoch in range(1, args.epochs+1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device, scaler, clip_grad=1.0
            )
            val_loss, val_acc, val_auc, _, _ = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_auc)
            logger.info("Epoch %2d: train_loss=%.4f, val_auc=%.4f", epoch, train_loss, val_auc)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                # Move state dict to CPU immediately to avoid GPU pickling issues
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        if best_state is None:
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        fold_states.append(best_state)

        # Determine best threshold on validation set
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        model.eval()
        _, _, _, y_val_true, y_val_prob = evaluate(model, val_loader, criterion, device)
        fpr, tpr, thresholds = roc_curve(y_val_true, y_val_prob)
        best_thr = thresholds[np.argmax(tpr - fpr)]
        fold_thresholds.append(best_thr)
        logger.info("Fold %d best threshold (val): %.4f", fold+1, best_thr)

        # Predict on test set
        test_set = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
        _, _, _, _, probs = evaluate(model, test_loader, criterion, device)
        fold_probs_test.append(probs)

    if not fold_probs_test:
        logger.error("No valid folds were trained.")
        return 1

    # -------------------------------------------------------------------
    # Ensemble and final evaluation
    # -------------------------------------------------------------------
    avg_probs_test = np.mean(fold_probs_test, axis=0)
    auc_test = roc_auc_score(y_test, avg_probs_test)
    final_threshold = np.mean(fold_thresholds)
    logger.info("Average validation threshold: %.4f", final_threshold)

    pred_test = (avg_probs_test >= final_threshold).astype(int)
    acc_test = accuracy_score(y_test, pred_test)
    cm_test = confusion_matrix(y_test, pred_test)

    logger.info("\nInternal Test Results (threshold from CV):")
    logger.info("AUC: %.4f, Accuracy: %.4f", auc_test, acc_test)
    logger.info("\n%s", classification_report(y_test, pred_test, target_names=["Female", "Male"]))

    # -------------------------------------------------------------------
    # Save artefact (all states already on CPU)
    # -------------------------------------------------------------------
    artefact = {
        "fold_states": fold_states,
        "input_dim": input_dim,
        "hidden_dims": args.hidden_dims,
        "dropout": args.dropout,
        "scaler_cli": scaler_cli,
        "measure_cols": meas_cols,
        "rad_const_mask": const_mask,
        "rad_spearman_keep_idx": spearman_keep_idx,
        "rad_scaler": rad_scaler,
        "rad_selector": rad_selector,
        "omics_cols": omics_cols,
        "pca_2d": pca_2d,
        "pca_3d": pca_3d,
        "best_threshold": final_threshold,
        "internal_auc": auc_test,
        "internal_acc": acc_test,
        "device": str(device),
        "args": {k: str(v) for k, v in vars(args).items()},
    }
    output_path = args.output_dir / "feature_concat_mlp.pkl"
    joblib.dump(artefact, output_path)
    logger.info("Model saved to %s (%.2f MB)", output_path,
                output_path.stat().st_size / (1024 * 1024))
    return 0


if __name__ == "__main__":
    sys.exit(main())