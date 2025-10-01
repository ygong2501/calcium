"""
SAM2 training script for single-cell segmentation.
Based on CellSAM strategies for epithelial cells.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm
import json
from pathlib import Path

# SAM2 imports (ensure sam2 is installed)
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("Please install SAM2: pip install git+https://github.com/facebookresearch/segment-anything-2.git")
    exit(1)

from utils.sam2_dataloader import create_dataloader


class CellSegmentationLoss(nn.Module):
    """
    Multi-component loss for tightly connected epithelial cells.
    Combines focal, dice, and boundary-weighted losses.
    """

    def __init__(self, boundary_weight=2.0, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.boundary_weight = boundary_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def forward(self, pred_masks, gt_masks, pred_iou=None):
        """
        Compute combined loss.

        Args:
            pred_masks: (B, 1, H, W) predicted probability masks
            gt_masks: (B, 1, H, W) ground truth binary masks
            pred_iou: (B, 1) predicted IoU scores (optional)

        Returns:
            total_loss, loss_components dict
        """
        # Focal loss for class imbalance
        focal = self.focal_loss(pred_masks, gt_masks)

        # Dice loss for overlap
        dice = self.dice_loss(pred_masks, gt_masks)

        # Boundary-weighted BCE for tight junctions
        boundary = self.boundary_weighted_bce(pred_masks, gt_masks)

        # Combine losses
        total = focal + dice + 0.5 * boundary

        losses = {
            'focal': focal.item(),
            'dice': dice.item(),
            'boundary': boundary.item()
        }

        # Add IoU loss if predictions provided
        if pred_iou is not None:
            actual_iou = self.compute_iou(torch.sigmoid(pred_masks) > 0.5, gt_masks)
            iou_loss = F.mse_loss(pred_iou.squeeze(), actual_iou)
            total = total + 0.1 * iou_loss
            losses['iou'] = iou_loss.item()

        return total, losses

    def focal_loss(self, inputs, targets):
        """Focal loss for handling imbalance between cell and background."""
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.focal_gamma)

        if self.focal_alpha >= 0:
            alpha_t = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean()

    def dice_loss(self, pred, target, smooth=1e-6):
        """Dice coefficient loss."""
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()

    def boundary_weighted_bce(self, pred, target):
        """BCE with higher weight on boundary pixels."""
        batch_size = target.shape[0]

        # Create boundary weight map
        weight_map = torch.ones_like(target)

        for i in range(batch_size):
            mask_np = target[i, 0].cpu().numpy()

            # Distance from boundaries
            dist = distance_transform_edt(mask_np)
            dist_inv = distance_transform_edt(1 - mask_np)

            # Boundary pixels (within 3 pixels of edge)
            boundary = np.minimum(dist, dist_inv) <= 3

            # Convert back to tensor
            boundary_tensor = torch.from_numpy(boundary).float().to(target.device)
            weight_map[i, 0][boundary_tensor > 0] = self.boundary_weight

        # Weighted BCE
        bce = F.binary_cross_entropy_with_logits(pred, target, weight=weight_map, reduction='mean')
        return bce

    @staticmethod
    def compute_iou(pred, target):
        """Compute IoU between prediction and target."""
        intersection = (pred * target).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)
        return iou


def setup_sam2_model(checkpoint_path, model_cfg, device='cuda'):
    """
    Setup SAM2 model for training.

    Args:
        checkpoint_path: Path to SAM2 checkpoint
        model_cfg: Model configuration file
        device: Device to use

    Returns:
        predictor, trainable_params
    """
    # Load model
    sam2_model = build_sam2(model_cfg, checkpoint_path, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    # Freeze image encoder to save memory
    predictor.model.image_encoder.requires_grad_(False)

    # Enable training for mask decoder and prompt encoder
    predictor.model.sam_mask_decoder.train()
    predictor.model.sam_prompt_encoder.train()

    # Collect trainable parameters
    trainable_params = []
    trainable_params += list(predictor.model.sam_mask_decoder.parameters())
    trainable_params += list(predictor.model.sam_prompt_encoder.parameters())

    num_params = sum(p.numel() for p in trainable_params)
    print(f"Training {num_params:,} parameters")
    print(f"Frozen image encoder: {sum(p.numel() for p in predictor.model.image_encoder.parameters()):,} parameters")

    return predictor, trainable_params


def train_epoch(
    predictor,
    dataloader,
    optimizer,
    scheduler,
    loss_fn,
    scaler,
    device,
    accumulation_steps=8
):
    """Train for one epoch."""
    predictor.model.train()
    epoch_losses = []

    progress_bar = tqdm(dataloader, desc="Training")

    for batch_idx, batch in enumerate(progress_bar):
        # Move data to device
        image = batch['image'].to(device)
        points = batch['points'].numpy()
        labels = batch['labels'].numpy()
        gt_masks = batch['masks'].to(device)

        num_cells = len(points)

        with autocast():
            # Set image (compute embeddings once)
            predictor.set_image(image.cpu().numpy().transpose(1, 2, 0))

            # Process each cell
            total_loss = 0
            all_losses = {'focal': 0, 'dice': 0, 'boundary': 0, 'iou': 0}

            for i in range(num_cells):
                # Get prediction
                masks, iou_pred, _ = predictor.predict(
                    point_coords=points[i:i+1],
                    point_labels=labels[i:i+1],
                    multimask_output=False
                )

                # Convert to tensors
                pred_mask = torch.from_numpy(masks).float().to(device)
                pred_iou = torch.from_numpy(iou_pred).float().to(device)

                # Reshape for loss
                pred_mask = pred_mask.unsqueeze(0)  # (1, 1, H, W)
                target_mask = gt_masks[i:i+1].unsqueeze(0)  # (1, 1, H, W)

                # Compute loss
                loss, loss_components = loss_fn(pred_mask, target_mask, pred_iou)

                # Accumulate
                total_loss += loss / num_cells
                for k, v in loss_components.items():
                    all_losses[k] += v / num_cells

            # Normalize by accumulation steps
            total_loss = total_loss / accumulation_steps

        # Backward pass
        scaler.scale(total_loss).backward()

        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Update learning rate
            scheduler.step()

        # Track losses
        epoch_losses.append(total_loss.item() * accumulation_steps)

        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{np.mean(epoch_losses[-100:]):.4f}",
            'focal': f"{all_losses['focal']:.4f}",
            'dice': f"{all_losses['dice']:.4f}",
            'lr': f"{scheduler.get_last_lr()[0]:.2e}"
        })

    return np.mean(epoch_losses)


def main():
    """Main training function."""
    # Configuration
    config = {
        'checkpoint_path': 'checkpoints/sam2.1_hiera_small.pt',
        'model_cfg': 'sam2_hiera_s.yaml',
        'dataset_dir': 'dataset',
        'output_dir': 'checkpoints',
        'cells_per_batch': 32,
        'batch_size': 1,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'accumulation_steps': 8,
        'num_workers': 4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)

    # Setup model
    print("Loading SAM2 model...")
    predictor, trainable_params = setup_sam2_model(
        config['checkpoint_path'],
        config['model_cfg'],
        config['device']
    )

    # Setup dataloader
    print("Creating dataloader...")
    train_dataloader = create_dataloader(
        dataset_dir=config['dataset_dir'],
        split='train',
        cells_per_batch=config['cells_per_batch'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=True,
        pin_memory=True
    )

    # Setup loss
    loss_fn = CellSegmentationLoss(
        boundary_weight=2.0,
        focal_alpha=0.25,
        focal_gamma=2.0
    )

    # Setup optimizer
    optimizer = optim.AdamW(
        trainable_params,
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # Setup scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=1000,
        T_mult=2,
        eta_min=1e-6
    )

    # Mixed precision scaler
    scaler = GradScaler()

    # Training loop
    print(f"Starting training for {config['num_epochs']} epochs...")
    best_loss = float('inf')

    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")

        # Train
        avg_loss = train_epoch(
            predictor,
            train_dataloader,
            optimizer,
            scheduler,
            loss_fn,
            scaler,
            config['device'],
            config['accumulation_steps']
        )

        print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % 5 == 0 or avg_loss < best_loss:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': predictor.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss
            }

            if avg_loss < best_loss:
                best_loss = avg_loss
                checkpoint_path = os.path.join(config['output_dir'], 'best_model.pt')
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved best model (loss: {avg_loss:.4f})")

            # Regular checkpoint
            checkpoint_path = os.path.join(config['output_dir'], f'checkpoint_epoch{epoch+1}.pt')
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()