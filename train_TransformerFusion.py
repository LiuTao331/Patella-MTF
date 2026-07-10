#!/usr/bin/env python3
"""
Train the Transformer‑based multimodal fusion model for sex estimation.

Implements the "Transformer Fusion" model described in the paper.
Modalities:
  1. Conventional morphometric parameters (6, standardised)
  2. Radiomic features (selected via Spearman + LASSO)
  3. 2048‑d 2D CNN features → PCA 256
  4. 2048‑d 3D CNN features → PCA 256

The four feature sets are projected into a shared 32‑d space, combined with
learnable modality embeddings, processed by a single‑layer Transformer encoder
(4 heads, GELU, pre‑layer normalisation, dropout 0.5), and classified by a
two‑layer MLP. Training uses class‑balanced focal loss (γ = 2.0) under 5‑fold
cross‑validation. The optimal decision threshold is determined on the validation
sets of each fold by maximising the Youden index; the final threshold is the
mean over folds. Predictions are reported on internal and optional external test
sets.

Output: transformer_fusion.pkl

Requirements: torch, torchvision, nibabel, pandas, numpy, scikit‑learn,
               tqdm, joblib, Pillow, MedicalNet (for 3D backbone).
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Image processing (no augmentation)
import nibabel as nib
from PIL import Image
from scipy.ndimage import zoom
from torchvision import models as tv_models, transforms

logger = logging.getLogger(__name__)


# =============================================================================
# Image preprocessing utilities (identical to training scripts)
# =============================================================================
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


# =============================================================================
# 2D Dataset: generates three orthogonal MIPs as a 3‑channel image
# =============================================================================
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
                label = int(row["sex"]) if "sex" in row else -1
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


# =============================================================================
# 3D Dataset: crops ROI, resizes to 64³
# =============================================================================
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
                label = int(row["sex"]) if "sex" in row else -1
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


# =============================================================================
# Feature extractors (pretrained 2D / 3D CNN)
# =============================================================================
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
                    nn.Linear(2048, 512), nn.ReLU(inplace=True),
                    nn.Dropout(0.3), nn.Linear(512, 1))
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
        self.feat_dim = base.fc[0].in_features
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(x); x = self.backbone.bn1(x); x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x); x = self.backbone.layer2(x); x = self.backbone.layer3(x); x = self.backbone.layer4(x)
        return self.global_avg_pool(x).flatten(1)


def extract_deep_features(
    dataset: Dataset, model: nn.Module, device: torch.device,
    batch_size: int = 32, desc: str = ""
) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    features = []
    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc=desc):
            imgs = imgs.to(device)
            feat = model(imgs).cpu().numpy()
            features.append(feat)
    return np.vstack(features)


# =============================================================================
# Radiomic feature selection (Spearman + LASSO)
# =============================================================================
def select_features_spearman_lasso(
    X_train: np.ndarray,
    y_train: np.ndarray,
    spearman_threshold: float = 0.9,
    cv_folds: int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler, SelectFromModel]:
    const_mask = X_train.var(axis=0) > 1e-10
    X_train = X_train[:, const_mask]

    corr = pd.DataFrame(X_train).corr(method="spearman").abs().values
    high_corr = np.where(corr >= spearman_threshold)
    remove = set()
    for i, j in zip(*high_corr):
        if i != j and i < j:
            remove.add(j)
    keep_idx = np.array([i for i in range(X_train.shape[1]) if i not in remove])
    X_train = X_train[:, keep_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    lasso = LassoCV(cv=cv_folds, random_state=random_state, max_iter=2000)
    lasso.fit(X_train, y_train)
    selector = SelectFromModel(lasso, prefit=True)

    return const_mask, keep_idx, scaler, selector


def apply_radiomic_pipeline(
    X: np.ndarray,
    const_mask: np.ndarray,
    keep_idx: np.ndarray,
    scaler: StandardScaler,
    selector: SelectFromModel,
) -> np.ndarray:
    X = X[:, const_mask]
    X = X[:, keep_idx]
    X = scaler.transform(X)
    return selector.transform(X)


# =============================================================================
# Transformer Fusion Model
# =============================================================================
class TransformerFusion(nn.Module):
    def __init__(
        self,
        input_dims: List[int],
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 1,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.embedding = nn.ModuleList([nn.Linear(d, d_model) for d in input_dims])
        self.modality_embed = nn.Parameter(torch.randn(1, len(input_dims), d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, modalities: List[torch.Tensor]) -> torch.Tensor:
        tokens = []
        for i, x in enumerate(modalities):
            tokens.append(self.embedding[i](x).unsqueeze(1))
        tokens = torch.cat(tokens, dim=1)
        tokens = tokens + self.modality_embed
        tokens = self.transformer(tokens)
        pooled = tokens.mean(dim=1)
        return self.classifier(pooled)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss
        return focal_loss.mean()


# =============================================================================
# Training & validation loops
# =============================================================================
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
    for x_cli, x_rad, x_2d, x_3d, labels in loader:
        x_cli = x_cli.to(device); x_rad = x_rad.to(device)
        x_2d = x_2d.to(device); x_3d = x_3d.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            outputs = model([x_cli, x_rad, x_2d, x_3d])
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()

        # Always unscale before stepping, then conditionally clip
        scaler.unscale_(optimizer)
        if clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)
        pred = outputs.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_labels, all_probs = [], []
    for x_cli, x_rad, x_2d, x_3d, labels in loader:
        x_cli = x_cli.to(device); x_rad = x_rad.to(device)
        x_2d = x_2d.to(device); x_3d = x_3d.to(device)
        labels = labels.to(device)

        outputs = model([x_cli, x_rad, x_2d, x_3d])
        loss = criterion(outputs, labels)
        total_loss += loss.item() * labels.size(0)
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        preds = outputs.argmax(dim=1).cpu().numpy()
        correct += (preds == labels.cpu().numpy()).sum()
        total += labels.size(0)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs)

    avg_loss = total_loss / total
    acc = correct / total
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5
    return avg_loss, acc, auc, np.array(all_labels), np.array(all_probs)


# =============================================================================
# Main script
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Transformer Fusion model for patellar sex estimation."
    )
    # Data
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--external-csv", type=Path, default=None)
    parser.add_argument("--train-measure", type=Path, required=True)
    parser.add_argument("--test-measure", type=Path, required=True)
    parser.add_argument("--external-measure", type=Path, default=None)
    parser.add_argument("--train-omics", type=Path, required=True)
    parser.add_argument("--test-omics", type=Path, required=True)
    parser.add_argument("--external-omics", type=Path, default=None)
    parser.add_argument("--ct-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--external-ct-dir", type=Path, default=None)
    parser.add_argument("--external-mask-dir", type=Path, default=None)

    # Pretrained feature extractors
    parser.add_argument("--model-2d-state", type=Path, required=True)
    parser.add_argument("--model-2d-cfg", type=Path, required=True,
                        help="JSON with mean/std for 2D normalisation")
    parser.add_argument("--model-3d-state", type=Path, required=True)

    # Column names
    parser.add_argument("--measure-cols", nargs="+", type=str,
                        default=["Patellar length", "Patellar width", "Patellar thickness",
                                 "Patellar volume", "Patellar surface area", "Patellar coronal perimeter"])
    parser.add_argument("--omics-cols", nargs="*", default=None,
                        help="Radiomic columns (auto‑detect if not given)")

    # Hyperparameters
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--spearman-threshold", type=float, default=0.9)
    parser.add_argument("--pca-dim-2d", type=int, default=256)
    parser.add_argument("--pca-dim-3d", type=int, default=256)

    # Output
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory to save transformer_fusion.pkl")
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
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # -------------------------------------------------------------------
    # Load metadata and align samples
    # -------------------------------------------------------------------
    train_meta = pd.read_csv(args.train_csv)
    test_meta = pd.read_csv(args.test_csv)
    for df in (train_meta, test_meta):
        df["number"] = df["number"].astype(str).str.strip()

    train_meas = pd.read_csv(args.train_measure)
    test_meas = pd.read_csv(args.test_measure)
    train_omics = pd.read_csv(args.train_omics, encoding="utf-8-sig")
    test_omics = pd.read_csv(args.test_omics, encoding="utf-8-sig")

    measure_cols = []
    for col in args.measure_cols:
        if col in train_meas.columns:
            measure_cols.append(col)
        else:
            raise KeyError(f"Measurement column '{col}' not found in CSV")

    if args.omics_cols is None or len(args.omics_cols) == 0:
        omics_cols = [c for c in train_omics.columns if c not in ("number", "sex", "dataset")]
    else:
        omics_cols = args.omics_cols

    train_all = train_meta.merge(train_meas[["number"] + measure_cols], on="number", how="inner")
    train_all = train_all.merge(train_omics[["number"] + omics_cols], on="number", how="inner")
    test_all = test_meta.merge(test_meas[["number"] + measure_cols], on="number", how="inner")
    test_all = test_all.merge(test_omics[["number"] + omics_cols], on="number", how="inner")

    with open(args.model_2d_cfg, "r") as f:
        cfg_2d = json.load(f)
    tfm_2d = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg_2d["mean"], std=cfg_2d["std"]),
    ])

    ds2d_tr = Knee2DDataset(train_all, args.ct_dir, args.mask_dir, args.seg_id, (224, 224), tfm_2d)
    ds3d_tr = Knee3DDataset(train_all, args.ct_dir, args.mask_dir, (64, 64, 64), args.seg_id)
    ds2d_test = Knee2DDataset(test_all, args.ct_dir, args.mask_dir, args.seg_id, (224, 224), tfm_2d)
    ds3d_test = Knee3DDataset(test_all, args.ct_dir, args.mask_dir, (64, 64, 64), args.seg_id)

    def filter_by_ids(df, *datasets):
        common = set.intersection(*[set(sid for sid, _, _, _ in ds.samples) for ds in datasets])
        return df[df["number"].isin(common)].reset_index(drop=True)

    train_all = filter_by_ids(train_all, ds2d_tr, ds3d_tr)
    test_all = filter_by_ids(test_all, ds2d_test, ds3d_test)
    logger.info("Aligned training samples: %d, test samples: %d", len(train_all), len(test_all))

    ds2d_tr = Knee2DDataset(train_all, args.ct_dir, args.mask_dir, args.seg_id, (224, 224), tfm_2d)
    ds3d_tr = Knee3DDataset(train_all, args.ct_dir, args.mask_dir, (64, 64, 64), args.seg_id)
    ds2d_test = Knee2DDataset(test_all, args.ct_dir, args.mask_dir, args.seg_id, (224, 224), tfm_2d)
    ds3d_test = Knee3DDataset(test_all, args.ct_dir, args.mask_dir, (64, 64, 64), args.seg_id)

    ext_2d = FeatureExtractor2D(args.model_2d_state).to(device)
    ext_3d = FeatureExtractor3D(args.model_3d_state).to(device)

    X_2d_tr = extract_deep_features(ds2d_tr, ext_2d, device, args.batch_size, "2D train")
    X_3d_tr = extract_deep_features(ds3d_tr, ext_3d, device, args.batch_size, "3D train")
    X_2d_test = extract_deep_features(ds2d_test, ext_2d, device, args.batch_size, "2D test")
    X_3d_test = extract_deep_features(ds3d_test, ext_3d, device, args.batch_size, "3D test")

    X_cli_tr = train_all[measure_cols].values.astype(np.float32)
    X_rad_tr = train_all[omics_cols].values.astype(np.float32)
    y_tr = train_all["sex"].values
    X_cli_test = test_all[measure_cols].values.astype(np.float32)
    X_rad_test = test_all[omics_cols].values.astype(np.float32)
    y_test = test_all["sex"].values

    scaler_cli = StandardScaler().fit(X_cli_tr)
    X_cli_tr = scaler_cli.transform(X_cli_tr)
    X_cli_test = scaler_cli.transform(X_cli_test)

    const_mask, keep_idx, rad_scaler, rad_selector = select_features_spearman_lasso(
        X_rad_tr, y_tr, spearman_threshold=args.spearman_threshold,
        cv_folds=5, random_state=args.seed,
    )
    X_rad_tr = apply_radiomic_pipeline(X_rad_tr, const_mask, keep_idx, rad_scaler, rad_selector)
    X_rad_test = apply_radiomic_pipeline(X_rad_test, const_mask, keep_idx, rad_scaler, rad_selector)

    pca_2d = PCA(n_components=min(args.pca_dim_2d, X_2d_tr.shape[1]), random_state=args.seed)
    pca_3d = PCA(n_components=min(args.pca_dim_3d, X_3d_tr.shape[1]), random_state=args.seed)
    X_2d_tr = pca_2d.fit_transform(X_2d_tr)
    X_3d_tr = pca_3d.fit_transform(X_3d_tr)
    X_2d_test = pca_2d.transform(X_2d_test)
    X_3d_test = pca_3d.transform(X_3d_test)

    # -------------------------------------------------------------------
    # 5‑fold CV with Youden threshold determination
    # -------------------------------------------------------------------
    class_counts = np.bincount(y_tr)
    alpha = torch.tensor(class_counts.sum() / (2 * class_counts), device=device, dtype=torch.float)
    criterion = FocalLoss(gamma=args.focal_gamma, alpha=alpha)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    fold_states = []
    fold_thresholds = []

    feature_dims = [X_cli_tr.shape[1], X_rad_tr.shape[1], X_2d_tr.shape[1], X_3d_tr.shape[1]]
    logger.info("Feature dimensions: %s", feature_dims)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_cli_tr, y_tr)):
        logger.info("=== Fold %d/5 ===", fold + 1)
        X_fold = [X_cli_tr[tr_idx], X_rad_tr[tr_idx], X_2d_tr[tr_idx], X_3d_tr[tr_idx]]
        X_val = [X_cli_tr[val_idx], X_rad_tr[val_idx], X_2d_tr[val_idx], X_3d_tr[val_idx]]
        y_fold, y_val = y_tr[tr_idx], y_tr[val_idx]

        tr_set = TensorDataset(*[torch.FloatTensor(a) for a in X_fold], torch.LongTensor(y_fold))
        val_set = TensorDataset(*[torch.FloatTensor(a) for a in X_val], torch.LongTensor(y_val))
        tr_loader = DataLoader(tr_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

        model = TransformerFusion(
            feature_dims, d_model=args.d_model, nhead=args.nhead,
            num_layers=args.num_layers, dropout=args.dropout,
        ).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
        # Explicitly enable GradScaler only when using CUDA
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

        best_auc, best_state, patience = 0.0, None, 0
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model, tr_loader, optimizer, criterion, device, scaler, args.gradient_clip
            )
            val_loss, val_acc, val_auc, vl, vp = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_auc)
            logger.info("Epoch %3d: train_loss=%.4f, val_auc=%.4f", epoch, train_loss, val_auc)

            if val_auc > best_auc:
                best_auc, best_state = val_auc, {k: v.cpu() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= args.patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        fold_states.append(best_state)

        # Determine optimal threshold on this fold's validation set (Youden)
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        _, _, _, vl, vp = evaluate(model, val_loader, criterion, device)
        fpr, tpr, thresholds = roc_curve(vl, vp)
        best_thr = thresholds[np.argmax(tpr - fpr)]
        fold_thresholds.append(best_thr)
        logger.info("Fold %d best threshold (Youden): %.4f", fold + 1, best_thr)

    final_threshold = np.mean(fold_thresholds)
    logger.info("Final threshold (mean over folds): %.4f", final_threshold)

    # -------------------------------------------------------------------
    # Internal test evaluation
    # -------------------------------------------------------------------
    def predict_proba(Xc, Xr, X2, X3, states, device):
        probs = []
        for s in states:
            m = TransformerFusion(feature_dims, args.d_model, args.nhead, args.num_layers, args.dropout).to(device)
            m.load_state_dict({k: v.to(device) for k, v in s.items()})
            m.eval()
            with torch.no_grad():
                x_list = [torch.FloatTensor(Xc).to(device),
                          torch.FloatTensor(Xr).to(device),
                          torch.FloatTensor(X2).to(device),
                          torch.FloatTensor(X3).to(device)]
                prob = torch.softmax(m(x_list), dim=1)[:, 1].cpu().numpy()
            probs.append(prob)
        return np.mean(probs, axis=0)

    prob_int = predict_proba(X_cli_test, X_rad_test, X_2d_test, X_3d_test, fold_states, device)
    pred_int = (prob_int >= final_threshold).astype(int)
    auc_int = roc_auc_score(y_test, prob_int)
    acc_int = accuracy_score(y_test, pred_int)
    logger.info("Internal test – AUC: %.4f, Accuracy: %.4f", auc_int, acc_int)

    # -------------------------------------------------------------------
    # External test evaluation (optional)
    # -------------------------------------------------------------------
    if args.external_csv is not None:
        if not all([args.external_measure, args.external_omics, args.external_ct_dir, args.external_mask_dir]):
            logger.error("External validation requested but some required paths are missing.")
            return 1

        logger.info("Evaluating external test set...")
        ext_meta = pd.read_csv(args.external_csv)
        ext_meta["number"] = ext_meta["number"].astype(str).str.strip()
        ext_meas = pd.read_csv(args.external_measure)
        ext_meas["number"] = ext_meas["number"].astype(str).str.strip()
        ext_omics = pd.read_csv(args.external_omics, encoding="utf-8-sig")
        ext_omics["number"] = ext_omics["number"].astype(str).str.strip()

        ext_all = ext_meta.merge(ext_meas[["number"] + measure_cols], on="number", how="inner")
        ext_all = ext_all.merge(ext_omics[["number"] + omics_cols], on="number", how="inner")

        ds2d_ext = Knee2DDataset(ext_all, args.external_ct_dir, args.external_mask_dir, args.seg_id, (224,224), tfm_2d)
        ds3d_ext = Knee3DDataset(ext_all, args.external_ct_dir, args.external_mask_dir, (64,64,64), args.seg_id)
        ext_all = filter_by_ids(ext_all, ds2d_ext, ds3d_ext)
        ds2d_ext = Knee2DDataset(ext_all, args.external_ct_dir, args.external_mask_dir, args.seg_id, (224,224), tfm_2d)
        ds3d_ext = Knee3DDataset(ext_all, args.external_ct_dir, args.external_mask_dir, (64,64,64), args.seg_id)

        X_2d_ext = extract_deep_features(ds2d_ext, ext_2d, device, args.batch_size, "2D ext")
        X_3d_ext = extract_deep_features(ds3d_ext, ext_3d, device, args.batch_size, "3D ext")
        X_cli_ext = scaler_cli.transform(ext_all[measure_cols].values.astype(np.float32))
        X_rad_ext = apply_radiomic_pipeline(ext_all[omics_cols].values.astype(np.float32),
                                            const_mask, keep_idx, rad_scaler, rad_selector)
        X_2d_ext = pca_2d.transform(X_2d_ext)
        X_3d_ext = pca_3d.transform(X_3d_ext)
        y_ext = ext_all["sex"].values

        prob_ext = predict_proba(X_cli_ext, X_rad_ext, X_2d_ext, X_3d_ext, fold_states, device)
        pred_ext = (prob_ext >= final_threshold).astype(int)
        auc_ext = roc_auc_score(y_ext, prob_ext)
        acc_ext = accuracy_score(y_ext, pred_ext)
        logger.info("External test – AUC: %.4f, Accuracy: %.4f", auc_ext, acc_ext)

    # -------------------------------------------------------------------
    # Save artefact
    # -------------------------------------------------------------------
    artefact = {
        "fold_states": fold_states,
        "feature_dims": feature_dims,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "threshold": final_threshold,
        "scaler_cli": scaler_cli,
        "measure_cols": measure_cols,
        "rad_const_mask": const_mask,
        "rad_keep_idx": keep_idx,
        "rad_scaler": rad_scaler,
        "rad_selector": rad_selector,
        "omics_cols": omics_cols,
        "pca_2d": pca_2d,
        "pca_3d": pca_3d,
        "internal_auc": auc_int,
        "internal_accuracy": acc_int,
        "device": str(device),
        "args": {k: str(v) for k, v in vars(args).items()},
    }
    out_path = args.output_dir / "transformer_fusion.pkl"
    joblib.dump(artefact, out_path)
    logger.info("Saved %s (%.2f MB)", out_path, out_path.stat().st_size / (1024 * 1024))
    return 0


if __name__ == "__main__":
    sys.exit(main())