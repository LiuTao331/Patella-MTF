#!/usr/bin/env python3
"""
Train a 2D ResNet‑50 model for sex estimation from patellar CT images.

Implements the "2D CNN" baseline described in the paper:
- Three orthogonal maximum intensity projections (MIP) of the patellar ROI
  are stacked into a 3‑channel 224×224 image.
- ResNet‑50 pretrained on ImageNet is fine‑tuned.
- Within each of the 5 folds, shallow layers are frozen; only layer3, layer4,
  and the fully connected layer are optimised.
- Adam optimiser (lr = 1e‑5, weight decay = 1e‑4), label‑smoothing
  cross‑entropy (smoothing = 0.1), batch size = 32.
- Training runs up to 30 epochs with early stopping (patience = 10) based on
  validation AUC.
- No additional data augmentation (e.g. slice‑thickness simulation, elastic
  deformation, MixUp) is applied, as it is not part of the paper's methodology.

Outputs:
  model_state.pth        – state_dict of the best model
  full_model.pth         – entire model object
 
Requirements: torch, torchvision, nibabel, numpy, pandas, scikit‑learn, tqdm, Pillow
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# ImageNet standard normalisation (used for transfer learning)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TARGET_SIZE = (224, 224)


# ---------------------------------------------------------------------------
# Image preprocessing helpers
# ---------------------------------------------------------------------------
def load_nifti(path: Path) -> np.ndarray:
    """Load a NIfTI file and return the data array."""
    return nib.load(str(path)).get_fdata()


def get_bbox_from_mask(mask: np.ndarray, margin: int = 5) -> Optional[Tuple[slice, ...]]:
    """Compute a bounding box around the non‑zero region of the mask."""
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
    """Maximum intensity projection along the given axis."""
    return np.max(vol, axis=axis)


def process_sample(
    ct_path: Path, mask_path: Path, target_size: Tuple[int, int] = TARGET_SIZE
) -> np.ndarray:
    """Generate a 3‑channel MIP image (uint8) from a CT volume and its mask.

    The input CT is expected to already be in the bone window [0, 1].
    The function rescales to 0–255 and returns a uint8 array of shape (H, W, 3).
    """
    ct_data = load_nifti(ct_path).astype(np.float32)
    mask_data = load_nifti(mask_path)

    # Scale bone‑window [0,1] to 0‑255 for MIP generation
    ct_uint8 = np.clip(ct_data * 255.0, 0, 255).astype(np.uint8)

    bbox = get_bbox_from_mask(mask_data)
    if bbox is None:
        raise ValueError(f"Empty mask in {mask_path}")

    ct_roi = ct_uint8[bbox].copy()
    mask_roi = mask_data[bbox] > 0
    ct_roi[~mask_roi] = 0  # set background to 0

    if any(s == 0 for s in ct_roi.shape):
        raise ValueError(f"Zero‑size ROI in {ct_path}")

    mip_z = max_intensity_projection(ct_roi, 0)
    mip_y = max_intensity_projection(ct_roi, 1)
    mip_x = max_intensity_projection(ct_roi, 2)

    # Convert to PIL, resize, then stack
    resize = transforms.Resize(target_size)
    to_pil = lambda arr: Image.fromarray(arr)  # uint8 arrays are fine

    z_img = np.array(resize(to_pil(mip_z)))
    y_img = np.array(resize(to_pil(mip_y)))
    x_img = np.array(resize(to_pil(mip_x)))

    return np.stack([z_img, y_img, x_img], axis=-1).astype(np.uint8)  # (H, W, 3)


# ---------------------------------------------------------------------------
# Dataset with strict file checking
# ---------------------------------------------------------------------------
class PatellaMIPDataset(Dataset):
    """Dataset that reads preprocessed NIfTI images and creates MIPs."""

    def __init__(
        self,
        csv_file: Path,
        ct_dir: Path,
        mask_dir: Path,
        seg_id: int = 2,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.df = pd.read_csv(csv_file)
        self.df["number"] = self.df["number"].astype(str).str.strip()
        self.ct_dir = ct_dir
        self.mask_dir = mask_dir
        self.seg_id = seg_id
        self.transform = transform

        # Pre‑compute file paths and validate existence
        self.samples = []
        for _, row in self.df.iterrows():
            sid = row["number"]
            ct_path = ct_dir / f"{sid}-{seg_id}.nii.gz"
            mask_path = mask_dir / f"{sid}-{seg_id}.nii.gz"
            if not ct_path.exists() or not mask_path.exists():
                raise FileNotFoundError(
                    f"Missing file for sample {sid}: CT={ct_path}, Mask={mask_path}"
                )
            label = 1 if str(row["sex"]).strip().lower() in ["male", "m", "1"] else 0
            self.samples.append((sid, ct_path, mask_path, label))
        logger.info("Loaded %d samples with verified files.", len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sid, ct_path, mask_path, label = self.samples[idx]
        img = process_sample(ct_path, mask_path)

        # img is (H, W, 3) uint8 → PIL Image
        pil_img = Image.fromarray(img)
        if self.transform is not None:
            pil_img = self.transform(pil_img)
        return pil_img, torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------------
# Model & training utilities
# ---------------------------------------------------------------------------
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.1) -> None:
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = pred.size(1)
        log_probs = nn.functional.log_softmax(pred, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred).fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))


def create_model(num_classes: int = 2) -> models.ResNet:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def freeze_shallow_layers(model: nn.Module) -> None:
    """Freeze all layers except layer3, layer4 and the classifier (fc)."""
    trainable_params = 0
    for name, param in model.named_parameters():
        if "layer3" in name or "layer4" in name or "fc" in name:
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False
    logger.info("Trainable parameters: %d", trainable_params)


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
    total_loss = 0.0
    all_preds, all_labels = [], []
    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        if clip_grad > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader.dataset), accuracy_score(all_labels, all_preds)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    for images, labels in tqdm(loader, desc="Eval", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)

        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    if len(set(all_labels)) < 2:
        logger.warning("Only one class present in evaluation set – AUC set to 0.5")
        auc = 0.5
    else:
        auc = roc_auc_score(all_labels, all_probs)
    return total_loss / len(loader.dataset), acc, auc, np.array(all_labels), np.array(all_probs)


# ---------------------------------------------------------------------------
# Main training script
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a 2D ResNet‑50 model for patellar sex estimation (5‑fold CV)."
    )
    parser.add_argument("--train-csv", type=Path, required=True,
                        help="CSV with columns 'number', 'sex' for the training set.")
    parser.add_argument("--internal-test-csv", type=Path, required=True,
                        help="CSV for the internal test set.")
    parser.add_argument("--external-test-csv", type=Path, default=None,
                        help="CSV for the external test set (optional).")
    parser.add_argument("--ct-dir", type=Path, required=True,
                        help="Directory with preprocessed NIfTI images.")
    parser.add_argument("--mask-dir", type=Path, required=True,
                        help="Directory with segmentation masks.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory to save models and results.")
    parser.add_argument("--seg-id", type=int, default=2,
                        help="Rater suffix in filenames (default: 2).")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32).")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate (default: 1e-5).")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay (default: 1e-4).")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="Label smoothing factor (default: 0.1).")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Maximum epochs (default: 30).")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (default: 10).")
    parser.add_argument("--gradient-clip", type=float, default=1.0,
                        help="Gradient clipping norm (default: 1.0).")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (default: 0).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # -----------------------------------------------------------------------
    # Transforms (ImageNet normalisation as standard for pretrained models)
    # -----------------------------------------------------------------------
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # -----------------------------------------------------------------------
    # Datasets
    # -----------------------------------------------------------------------
    logger.info("Loading internal training set...")
    full_train_ds = PatellaMIPDataset(
        args.train_csv, args.ct_dir, args.mask_dir, seg_id=args.seg_id,
        transform=train_transform,
    )
    int_test_ds = PatellaMIPDataset(
        args.internal_test_csv, args.ct_dir, args.mask_dir, seg_id=args.seg_id,
        transform=eval_transform,
    )

    ext_test_ds = None
    if args.external_test_csv:
        ext_test_ds = PatellaMIPDataset(
            args.external_test_csv, args.ct_dir, args.mask_dir, seg_id=args.seg_id,
            transform=eval_transform,
        )

    # Collect labels for stratified splitting
    all_labels = [sample[-1] for sample in full_train_ds.samples]

    # -----------------------------------------------------------------------
    # 5‑fold cross‑validation
    # -----------------------------------------------------------------------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(full_train_ds)), all_labels)):
        logger.info("\n========== Fold %d/5 ==========", fold + 1)

        train_sub = Subset(full_train_ds, train_idx)
        val_sub = Subset(full_train_ds, val_idx)

        train_loader = DataLoader(
            train_sub, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
        )
        val_loader = DataLoader(
            val_sub, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )

        model = create_model().to(device)
        freeze_shallow_layers(model)

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=args.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )
        criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

        best_val_auc = 0.0
        best_state = None
        patience_counter = 0

        for epoch in tqdm(range(1, args.epochs + 1), desc=f"Fold {fold+1} Epochs"):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device, scaler, args.gradient_clip
            )
            val_loss, val_acc, val_auc, _, _ = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_auc)

            logger.info(
                "Epoch %d: train_loss=%.3f, train_acc=%.4f, val_loss=%.3f, val_auc=%.4f",
                epoch, train_loss, train_acc, val_loss, val_auc,
            )

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        # Evaluate on test sets
        model.load_state_dict(best_state)
        _, int_acc, int_auc, _, _ = evaluate(
            model,
            DataLoader(int_test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers),
            criterion, device,
        )

        ext_acc, ext_auc = np.nan, np.nan
        if ext_test_ds:
            _, ext_acc, ext_auc, _, _ = evaluate(
                model,
                DataLoader(ext_test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers),
                criterion, device,
            )

        logger.info(
            "Fold %d: Internal AUC=%.4f, External AUC=%.4f",
            fold + 1, int_auc, ext_auc if not np.isnan(ext_auc) else -1,
        )
        fold_results.append({
            "fold": fold + 1,
            "best_val_auc": best_val_auc,
            "internal_auc": int_auc,
            "external_auc": ext_auc if not np.isnan(ext_auc) else None,
            "state_dict": best_state,
        })

    # -----------------------------------------------------------------------
    # Summary of cross‑validation
    # -----------------------------------------------------------------------
    int_aucs = [r["internal_auc"] for r in fold_results]
    logger.info("\n=== 5‑fold CV results ===")
    for r in fold_results:
        logger.info(
            "Fold %d: Val AUC=%.4f, Internal AUC=%.4f, External AUC=%s",
            r["fold"], r["best_val_auc"], r["internal_auc"],
            f"{r['external_auc']:.4f}" if r["external_auc"] is not None else "N/A",
        )
    logger.info("Mean Internal AUC: %.4f ± %.4f", np.mean(int_aucs), np.std(int_aucs))

    # -----------------------------------------------------------------------
    # Final model training on the full training set with validation split
    # -----------------------------------------------------------------------
    logger.info("\nTraining final model on the full training set with 10% validation...")
    # Split full_train_ds into training and validation for final model
    final_train_idx, final_val_idx = train_test_split(
        range(len(full_train_ds)),
        test_size=0.1,
        stratify=all_labels,
        random_state=args.random_state,
    )
    final_train_sub = Subset(full_train_ds, final_train_idx)
    final_val_sub = Subset(full_train_ds, final_val_idx)
    final_train_loader = DataLoader(
        final_train_sub, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    final_val_loader = DataLoader(
        final_val_sub, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    final_model = create_model().to(device)
    freeze_shallow_layers(final_model)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, final_model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )
    criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_auc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in tqdm(range(1, args.epochs + 1), desc="Final model training"):
        train_loss, train_acc = train_one_epoch(
            final_model, final_train_loader, optimizer, criterion, device, scaler, args.gradient_clip
        )
        val_loss, val_acc, val_auc, _, _ = evaluate(final_model, final_val_loader, criterion, device)
        scheduler.step(val_auc)

        logger.info(
            "Final Epoch %d: train_loss=%.3f, val_auc=%.4f", epoch, train_loss, val_auc
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu() for k, v in final_model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    # Load best weights
    if best_state is not None:
        final_model.load_state_dict(best_state)
    else:
        logger.warning("No improvement during final training; using last state.")

    # Save final model
    model_state_path = args.output_dir / "model_state.pth"
    torch.save(final_model.state_dict(), model_state_path)
    logger.info("Model state_dict saved to %s", model_state_path)

    full_model_path = args.output_dir / "full_model.pth"
    torch.save(final_model, full_model_path)
    logger.info("Full model saved to %s", full_model_path)

    # Save performance summary
    results_summary = {
        "cv_folds": fold_results,
        "mean_internal_auc": float(np.mean(int_aucs)),
        "std_internal_auc": float(np.std(int_aucs)),
        "final_val_auc": best_val_auc,
    }
    with open(args.output_dir / "2dcnn_results.json", "w") as f:
        json.dump(results_summary, f, indent=2, default=str)

    logger.info("All outputs saved to %s", args.output_dir)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())