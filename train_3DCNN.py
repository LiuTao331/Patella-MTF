#!/usr/bin/env python3
"""
Train a 3D ResNet‑50 model for sex estimation from patellar CT images.

Implements the "3D CNN" baseline as described in the paper:
- Patellar ROI resampled to 64×64×64 isotropic voxels.
- 3D ResNet‑50 backbone pretrained on MedicalNet.
- Two‑stage training under 5‑fold cross‑validation:
  1. Freeze backbone, train classification head for 10 epochs.
  2. Jointly fine‑tune all parameters for up to 60 epochs
     with early stopping (patience = 15).
- Adam optimiser (head lr = 1e‑3, full lr = 5e‑4, weight decay = 1e‑4).
- Label‑smoothing binary cross‑entropy loss (smoothing = 0.1).
- Batch size = 4.
- No additional data augmentation (e.g., slice‑thickness simulation,
  elastic deformation, MixUp) is applied, as per the paper.

Output:
  model_state.pth – state_dict of the fold with the best validation AUC.

Requirements:
  torch, torchvision, nibabel, numpy, pandas, scikit‑learn, tqdm,
  MedicalNet (clone from https://github.com/Tencent/MedicalNet and set PYTHONPATH)
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple, Optional

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.ndimage import zoom
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

# MedicalNet backbone
try:
    from models import resnet as medicalnet_resnet
except ImportError:
    raise ImportError(
        "Could not import MedicalNet models. "
        "Please clone https://github.com/Tencent/MedicalNet and add its 'models' "
        "directory to your Python path."
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
def load_nifti(path: Path) -> np.ndarray:
    """Return the data array from a NIfTI file."""
    return nib.load(str(path)).get_fdata().astype(np.float32)


def get_bbox_from_mask(mask: np.ndarray, margin: int = 5) -> Optional[Tuple[slice, ...]]:
    """Compute bounding box around non‑zero region of a mask."""
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


def resize_to_target(volume: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
    """Resize volume to target_size using linear interpolation, centering content.

    Handles both cases where resized volume is larger or smaller than the target
    by cropping or zero‑padding accordingly.
    """
    factors = [t / s for s, t in zip(volume.shape, target_size)]
    resized = zoom(volume, factors, order=1)

    out = np.zeros(target_size, dtype=np.float32)

    # Determine cropping from resized (if larger) and placement in target
    crop_slices = []
    out_slices = []
    for dim in range(3):
        if resized.shape[dim] >= target_size[dim]:
            # Crop: take centre part of resized
            start = (resized.shape[dim] - target_size[dim]) // 2
            crop_slices.append(slice(start, start + target_size[dim]))
            out_slices.append(slice(None))
        else:
            # Pad: keep entire resized, place in centre of target
            crop_slices.append(slice(None))
            pad_before = (target_size[dim] - resized.shape[dim]) // 2
            out_slices.append(slice(pad_before, pad_before + resized.shape[dim]))

    resized_cropped = resized[tuple(crop_slices)]
    out[tuple(out_slices)] = resized_cropped
    return out


def preprocess_sample(
    ct_path: Path,
    mask_path: Path,
    target_size: Tuple[int, int, int] = (64, 64, 64),
) -> np.ndarray:
    """
    Crop the patellar ROI using the mask, mask out background,
    and resize to target_size.  The input CT is assumed to already
    be preprocessed (bone window [0,1]).
    """
    ct_data = load_nifti(ct_path)
    mask_data = load_nifti(mask_path)

    bbox = get_bbox_from_mask(mask_data)
    if bbox is None:
        raise ValueError(f"Empty mask in {mask_path}")

    ct_roi = ct_data[bbox].copy()
    mask_roi = mask_data[bbox] > 0
    ct_roi[~mask_roi] = 0.0

    if any(d == 0 for d in ct_roi.shape):
        raise ValueError(f"Zero‑size ROI in {ct_path}")

    # Resize to the target dimensions
    return resize_to_target(ct_roi, target_size)


# ---------------------------------------------------------------------------
# Dataset (strict file naming)
# ---------------------------------------------------------------------------
class Patella3DDataset(Dataset):
    """
    Dataset that loads preprocessed NIfTI volumes and extracts the patellar ROI.

    Expects files to be named as ``<sample_id>-<seg_id>.nii.gz``.
    Only samples for which both the CT image and the mask exist are kept.
    """

    def __init__(
        self,
        csv_file: Path,
        ct_dir: Path,
        mask_dir: Path,
        target_size: Tuple[int, int, int] = (64, 64, 64),
        seg_id: int = 2,
        transform: Optional[object] = None,
    ) -> None:
        self.df = pd.read_csv(csv_file)
        self.df["number"] = self.df["number"].astype(str).str.strip()
        self.ct_dir = ct_dir
        self.mask_dir = mask_dir
        self.target_size = target_size
        self.seg_id = seg_id
        self.transform = transform

        # Validate samples and build file paths
        valid_samples = []
        for _, row in self.df.iterrows():
            sid = row["number"]
            ct_path = ct_dir / f"{sid}-{seg_id}.nii.gz"
            mask_path = mask_dir / f"{sid}-{seg_id}.nii.gz"
            if ct_path.exists() and mask_path.exists():
                label = 1.0 if str(row["sex"]).strip().lower() in ("male", "m", "1") else 0.0
                valid_samples.append((sid, ct_path, mask_path, label))
            else:
                logger.warning("Skipping sample %s – missing file(s)", sid)
        self.samples = valid_samples
        logger.info("Loaded %d valid samples.", len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sid, ct_path, mask_path, label = self.samples[idx]
        image = preprocess_sample(ct_path, mask_path, self.target_size)
        image_tensor = torch.from_numpy(image).unsqueeze(0).float()  # add channel dim

        if self.transform is not None:
            image_tensor = self.transform(image_tensor)

        return image_tensor, torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Model definition (MedicalNet backbone)
# ---------------------------------------------------------------------------
class MedicalNetGender(nn.Module):
    """3D ResNet‑50 with a binary classification head."""

    def __init__(self, backbone: nn.Module, feature_dim: int = 2048, dropout: float = 0.3) -> None:
        super().__init__()
        self.backbone = backbone
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        pooled = self.global_avg_pool(x).flatten(1)
        return self.fc(pooled)


def create_model(
    target_size: Tuple[int, int, int] = (64, 64, 64),
    pretrain_path: Optional[Path] = None,
) -> nn.Module:
    """Instantiate the MedicalNet backbone and load pretrained weights if available."""
    backbone = medicalnet_resnet.resnet50(
        sample_input_W=target_size[2],
        sample_input_H=target_size[1],
        sample_input_D=target_size[0],
        num_seg_classes=1,
    )
    model = MedicalNetGender(backbone)

    if pretrain_path and pretrain_path.exists():
        logger.info("Loading pretrained weights from %s", pretrain_path)
        state = torch.load(pretrain_path, map_location="cpu")
        # Ignore incompatible keys (fc, conv_seg, etc.)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state.items()
                           if k in model_dict and "fc" not in k and "conv_seg" not in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
    else:
        logger.warning("Pretrained weights not provided or not found – using random initialisation.")

    return model


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------
class LabelSmoothingBCEWithLogitsLoss(nn.Module):
    def __init__(self, smoothing: float = 0.1) -> None:
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            smooth_target = target * (1 - self.smoothing) + 0.5 * self.smoothing
        return nn.functional.binary_cross_entropy_with_logits(pred, smooth_target)


# ---------------------------------------------------------------------------
# Training & evaluation routines
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
    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        if clip_grad > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs.detach()) > 0.5).float()
        correct += (preds == labels).sum().item()
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
    for images, labels in tqdm(loader, desc="Eval", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(outputs).cpu().numpy()
        preds = (probs > 0.5).astype(float)
        correct += (preds == labels.cpu().numpy()).sum()
        total += labels.size(0)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs)

    acc = correct / total
    if len(set(all_labels)) < 2:
        logger.warning("Only one class present – AUC set to 0.5")
        auc = 0.5
    else:
        auc = roc_auc_score(all_labels, all_probs)
    return total_loss / total, acc, auc, np.array(all_labels), np.array(all_probs)


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a 3D ResNet‑50 (MedicalNet) for patellar sex estimation."
    )
    parser.add_argument("--train-csv", type=Path, required=True,
                        help="CSV for the internal training set (columns: number, sex).")
    parser.add_argument("--internal-test-csv", type=Path, required=True,
                        help="CSV for the internal test set.")
    parser.add_argument("--external-test-csv", type=Path, default=None,
                        help="CSV for the external test set (optional).")
    parser.add_argument("--ct-dir", type=Path, required=True,
                        help="Directory with preprocessed NIfTI CT images.")
    parser.add_argument("--mask-dir", type=Path, required=True,
                        help="Directory with segmentation masks.")
    parser.add_argument("--pretrain-path", type=Path, default=None,
                        help="Path to MedicalNet pretrained weights (resnet_50_23dataset.pth).")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory to save the best model state_dict.")
    parser.add_argument("--seg-id", type=int, default=2,
                        help="Rater suffix in filenames (default: 2).")
    parser.add_argument("--target-size", nargs=3, type=int, default=[64, 64, 64],
                        help="Input size (D H W) (default: 64 64 64).")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs-head", type=int, default=10,
                        help="Epochs for frozen‑backbone head training.")
    parser.add_argument("--epochs-full", type=int, default=60,
                        help="Maximum epochs for full fine‑tuning.")
    parser.add_argument("--lr-head", type=float, default=1e-3,
                        help="Learning rate for head training.")
    parser.add_argument("--lr-full", type=float, default=5e-4,
                        help="Learning rate for full fine‑tuning.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience for full fine‑tuning.")
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_size = tuple(args.target_size)

    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # -------------------------------------------------------------------
    # Datasets
    # -------------------------------------------------------------------
    logger.info("Loading internal training set...")
    full_train_ds = Patella3DDataset(
        args.train_csv, args.ct_dir, args.mask_dir,
        target_size=target_size, seg_id=args.seg_id,
    )
    int_test_ds = Patella3DDataset(
        args.internal_test_csv, args.ct_dir, args.mask_dir,
        target_size=target_size, seg_id=args.seg_id,
    )
    ext_test_ds = None
    if args.external_test_csv:
        ext_test_ds = Patella3DDataset(
            args.external_test_csv, args.ct_dir, args.mask_dir,
            target_size=target_size, seg_id=args.seg_id,
        )

    # Labels for stratification
    all_labels = [lbl for _, _, _, lbl in full_train_ds.samples]

    # -------------------------------------------------------------------
    # 5‑fold cross‑validation
    # -------------------------------------------------------------------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(full_train_ds)), all_labels)):
        logger.info("\n========== Fold %d/5 ==========", fold + 1)

        train_sub = Subset(full_train_ds, train_idx)
        val_sub = Subset(full_train_ds, val_idx)

        train_loader = DataLoader(
            train_sub, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
        )
        val_loader = DataLoader(
            val_sub, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )

        model = create_model(target_size, args.pretrain_path)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(device)

        criterion = LabelSmoothingBCEWithLogitsLoss(smoothing=args.label_smoothing)
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

        # ---- Stage 1: train head ----
        logger.info("Stage 1: training head (backbone frozen)")
        backbone = model.module.backbone if isinstance(model, nn.DataParallel) else model.backbone
        for param in backbone.parameters():
            param.requires_grad = False

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr_head, weight_decay=args.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5,
        )
        best_val_auc = 0.0
        head_patience = 0
        for epoch in range(1, args.epochs_head + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device, scaler, args.gradient_clip,
            )
            val_loss, val_acc, val_auc, _, _ = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_auc)
            logger.info("Head epoch %2d: loss=%.4f, val_auc=%.4f", epoch, train_loss, val_auc)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                head_patience = 0
            else:
                head_patience += 1
                if head_patience >= 5:
                    logger.info("Head early stop at epoch %d", epoch)
                    break

        # ---- Stage 2: full fine‑tuning ----
        logger.info("Stage 2: full model fine‑tuning")
        for param in model.parameters():
            param.requires_grad = True

        optimizer = optim.Adam(
            model.parameters(), lr=args.lr_full, weight_decay=args.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5,
        )
        best_fold_auc = best_val_auc
        best_fold_state = None
        full_patience = 0
        for epoch in range(1, args.epochs_full + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device, scaler, args.gradient_clip,
            )
            val_loss, val_acc, val_auc, _, _ = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_auc)
            logger.info("Full epoch %2d: loss=%.4f, val_auc=%.4f", epoch, train_loss, val_auc)

            if val_auc > best_fold_auc:
                best_fold_auc = val_auc
                best_fold_state = {k: v.cpu() for k, v in model.state_dict().items()}
                full_patience = 0
            else:
                full_patience += 1
                if full_patience >= args.patience:
                    logger.info("Full early stop at epoch %d", epoch)
                    break

        if best_fold_state is None:
            best_fold_state = {k: v.cpu() for k, v in model.state_dict().items()}

        # Evaluate on test sets
        model.load_state_dict(best_fold_state)
        _, _, int_auc, _, _ = evaluate(
            model,
            DataLoader(int_test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers),
            criterion, device,
        )
        ext_auc = np.nan
        if ext_test_ds:
            _, _, ext_auc, _, _ = evaluate(
                model,
                DataLoader(ext_test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers),
                criterion, device,
            )
        logger.info("Fold %d: Val AUC=%.4f, Internal AUC=%.4f, External AUC=%s",
                    fold + 1, best_fold_auc, int_auc, f"{ext_auc:.4f}" if not np.isnan(ext_auc) else "N/A")
        fold_results.append({"fold": fold + 1, "val_auc": best_fold_auc, "state_dict": best_fold_state})

    # Save best fold
    best_fold = max(fold_results, key=lambda x: x["val_auc"])
    logger.info("Best fold: %d (Val AUC = %.4f). Saving model_state.pth.",
                best_fold["fold"], best_fold["val_auc"])
    torch.save(best_fold["state_dict"], args.output_dir / "model_state.pth")
    logger.info("Training finished. Model saved to %s", args.output_dir / "model_state.pth")
    return 0


if __name__ == "__main__":
    sys.exit(main())