"""
SAM2 DataLoader for high-density cell segmentation.
Based on CellSAM approach with memory-efficient batch sampling.
"""

import os
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import torch
from torch.utils.data import Dataset, DataLoader


class CellSegmentationDataset(Dataset):
    """
    Memory-efficient dataset for SAM2 training on high-density cells.
    Samples a subset of cells per image to fit GPU memory.
    """

    def __init__(
        self,
        dataset_dir: str,
        split: str = 'train',
        cells_per_batch: int = 32,
        include_negative: bool = True,
        transform=None,
        image_size: Tuple[int, int] = (512, 512)
    ):
        """
        Initialize dataset.

        Args:
            dataset_dir: Root dataset directory
            split: Dataset split ('train', 'val', 'test')
            cells_per_batch: Number of cells to sample per image (memory constraint)
            include_negative: Whether to include negative prompts
            transform: Optional augmentation pipeline
            image_size: Expected image dimensions
        """
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.cells_per_batch = cells_per_batch
        self.include_negative = include_negative
        self.transform = transform
        self.image_size = image_size

        # Setup paths
        self.image_dir = self.dataset_dir / 'images' / split
        self.mask_dir = self.dataset_dir / 'masks' / split
        self.prompt_dir = self.dataset_dir / 'prompts' / split

        # Validate directories exist
        if not self.image_dir.exists():
            raise ValueError(f"Image directory not found: {self.image_dir}")

        # Collect image files
        self.image_files = sorted(self.image_dir.glob('*.png'))

        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.image_dir}")

        print(f"Found {len(self.image_files)} images for {split} split")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load an image and sample a subset of its cells.

        Returns:
            Dictionary containing:
            - 'image': (3, H, W) tensor
            - 'points': (N, 2) cell center points [x, y]
            - 'labels': (N,) all 1s for positive prompts
            - 'masks': (N, H, W) binary masks for each cell
            - 'negative_points': (M, 2) background points (optional)
            - 'negative_labels': (M,) all 0s (optional)
            - 'image_path': Path to source image
            - 'num_total_cells': Total cells in image
            - 'num_sampled_cells': Number of cells sampled
        """
        # Load image
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))

        # Handle grayscale images with alpha channel
        if image.shape[2] == 4:
            # Remove alpha channel
            image = image[:, :, :3]
        elif len(image.shape) == 2 or image.shape[2] == 1:
            # Convert grayscale to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load instance mask
        mask_path = self.mask_dir / f"{img_path.stem}.npz"
        if mask_path.exists():
            mask_data = np.load(mask_path)
            instance_mask = mask_data['mask']
        else:
            # Fallback to .npy format
            mask_path = self.mask_dir / f"{img_path.stem}.npy"
            if mask_path.exists():
                instance_mask = np.load(mask_path)
            else:
                raise FileNotFoundError(f"Mask not found: {mask_path}")

        # Load prompts
        prompt_path = self.prompt_dir / f"{img_path.stem}.json"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompts not found: {prompt_path}")

        with open(prompt_path, 'r') as f:
            prompt_data = json.load(f)

        positive_prompts = prompt_data['positive_prompts']
        negative_prompts = prompt_data.get('negative_prompts', [])

        # Get all cell IDs
        cell_ids = list(positive_prompts.keys())
        num_total_cells = len(cell_ids)

        # Sample subset of cells for memory efficiency
        if num_total_cells > self.cells_per_batch:
            sampled_ids = random.sample(cell_ids, self.cells_per_batch)
        else:
            sampled_ids = cell_ids

        # Prepare data for sampled cells
        points = []
        labels = []
        masks = []

        for cell_id in sampled_ids:
            prompt_info = positive_prompts[cell_id]

            # Get point (already in [x, y] format)
            point = prompt_info['point']
            points.append(point)
            labels.append(1)  # Positive prompt

            # Extract binary mask for this cell
            cell_mask = (instance_mask == int(cell_id)).astype(np.uint8)
            masks.append(cell_mask)

        # Convert to numpy arrays
        points = np.array(points, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        masks = np.array(masks, dtype=np.uint8)

        # Handle negative prompts
        negative_points = []
        negative_labels = []

        if self.include_negative and negative_prompts:
            # Sample a few negative points
            num_neg = min(5, len(negative_prompts))
            sampled_neg = random.sample(negative_prompts, num_neg) if len(negative_prompts) > num_neg else negative_prompts

            for neg_prompt in sampled_neg:
                negative_points.append(neg_prompt['point'])
                negative_labels.append(0)

        negative_points = np.array(negative_points, dtype=np.float32) if negative_points else np.zeros((0, 2), dtype=np.float32)
        negative_labels = np.array(negative_labels, dtype=np.int32) if negative_labels else np.zeros((0,), dtype=np.int32)

        # Apply augmentations if specified
        if self.transform:
            # Augmentation should handle image, masks, and keypoints together
            augmented = self.transform(
                image=image,
                masks=list(masks),
                keypoints=points.tolist()
            )
            image = augmented['image']
            masks = np.array(augmented['masks'])
            points = np.array(augmented['keypoints'], dtype=np.float32)

        # Convert to tensors
        image_tensor = torch.from_numpy(image).float()
        if len(image_tensor.shape) == 2:
            image_tensor = image_tensor.unsqueeze(0)  # Add channel dimension
        elif image_tensor.shape[2] == 3:
            image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW

        # Normalize image to [0, 1]
        image_tensor = image_tensor / 255.0

        points_tensor = torch.from_numpy(points).float()
        labels_tensor = torch.from_numpy(labels).long()
        masks_tensor = torch.from_numpy(masks).float()
        neg_points_tensor = torch.from_numpy(negative_points).float()
        neg_labels_tensor = torch.from_numpy(negative_labels).long()

        return {
            'image': image_tensor,
            'points': points_tensor,
            'labels': labels_tensor,
            'masks': masks_tensor,
            'negative_points': neg_points_tensor,
            'negative_labels': neg_labels_tensor,
            'image_path': str(img_path),
            'num_total_cells': num_total_cells,
            'num_sampled_cells': len(sampled_ids)
        }

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function for handling variable number of cells.

        Since each image may have different number of cells, we keep
        batch_size=1 for simplicity in training.
        """
        return batch[0]  # Return single item since batch_size=1


def create_dataloader(
    dataset_dir: str,
    split: str = 'train',
    cells_per_batch: int = 32,
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    transform=None
) -> DataLoader:
    """
    Create a DataLoader for SAM2 training.

    Args:
        dataset_dir: Root dataset directory
        split: Dataset split
        cells_per_batch: Number of cells to sample per image
        batch_size: Images per batch (usually 1 due to memory)
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        pin_memory: Whether to pin memory for GPU transfer
        transform: Optional augmentation pipeline

    Returns:
        DataLoader instance
    """
    dataset = CellSegmentationDataset(
        dataset_dir=dataset_dir,
        split=split,
        cells_per_batch=cells_per_batch,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=CellSegmentationDataset.collate_fn if batch_size == 1 else None
    )

    return dataloader


# Example usage
if __name__ == "__main__":
    # Test the dataloader
    dataloader = create_dataloader(
        dataset_dir='dataset',
        split='train',
        cells_per_batch=32,
        batch_size=1,
        num_workers=0,  # Use 0 for debugging
        shuffle=False
    )

    # Load one batch
    for batch in dataloader:
        print(f"Image shape: {batch['image'].shape}")
        print(f"Points shape: {batch['points'].shape}")
        print(f"Masks shape: {batch['masks'].shape}")
        print(f"Total cells: {batch['num_total_cells']}")
        print(f"Sampled cells: {batch['num_sampled_cells']}")
        break