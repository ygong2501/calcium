"""
SAM2 data generation utilities for single-cell segmentation.

This module provides functions to generate instance masks and point prompts
for fine-tuning SAM2 on epithelial cell segmentation tasks.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.ndimage import distance_transform_edt
import cv2


def create_instance_mask_from_cells(
    cell_masks: np.ndarray,
    active_cells: Optional[List[int]] = None
) -> np.ndarray:
    """
    Create an instance mask where each cell has a unique ID.

    Args:
        cell_masks: Multi-instance mask from pouch.get_cell_masks()
        active_cells: Optional list of active cell IDs to include

    Returns:
        Instance mask with unique IDs for each cell (uint16 for >255 cells)
    """
    if active_cells is not None:
        # Filter to only active cells
        instance_mask = np.zeros_like(cell_masks, dtype=np.uint16)
        for new_id, cell_id in enumerate(active_cells, start=1):
            instance_mask[cell_masks == cell_id] = new_id
    else:
        # Use all non-zero cells
        instance_mask = cell_masks.astype(np.uint16)

    return instance_mask


def generate_point_prompts_from_mask(
    instance_mask: np.ndarray,
    strategy: str = 'distance_based',
    min_cell_area: int = 10
) -> Dict[int, Dict]:
    """
    Generate optimal point prompts for each cell in the instance mask.

    Args:
        instance_mask: Instance segmentation mask with unique cell IDs
        strategy: 'distance_based', 'centroid', or 'random'
        min_cell_area: Minimum pixels for a valid cell

    Returns:
        Dictionary mapping cell_id to prompt information:
        {
            cell_id: {
                'point': [x, y],  # Note: x,y format for SAM2
                'label': 1,  # Always 1 for positive prompt
                'area': cell_area,
                'confidence': distance_from_boundary
            }
        }
    """
    prompts = {}
    unique_ids = np.unique(instance_mask)
    unique_ids = unique_ids[unique_ids != 0]  # Remove background

    for cell_id in unique_ids:
        # Get binary mask for this cell
        cell_binary = (instance_mask == cell_id).astype(np.uint8)
        cell_area = np.sum(cell_binary)

        # Skip very small cells (likely noise)
        if cell_area < min_cell_area:
            continue

        if strategy == 'distance_based':
            # Compute distance transform
            dist_transform = distance_transform_edt(cell_binary)

            if dist_transform.max() == 0:
                continue

            # Find pixels furthest from boundaries (top 20%)
            threshold = np.percentile(dist_transform[dist_transform > 0], 80)
            candidate_pixels = np.where(dist_transform >= threshold)

            if len(candidate_pixels[0]) > 0:
                # Randomly sample one point from candidates
                idx = np.random.randint(len(candidate_pixels[0]))
                y, x = candidate_pixels[0][idx], candidate_pixels[1][idx]
                confidence = float(dist_transform[y, x])
            else:
                # Fallback to maximum distance point
                y, x = np.unravel_index(dist_transform.argmax(), dist_transform.shape)
                confidence = float(dist_transform[y, x])

        elif strategy == 'centroid':
            # Simple centroid calculation
            coords = np.where(cell_binary)
            if len(coords[0]) == 0:
                continue
            y = int(np.mean(coords[0]))
            x = int(np.mean(coords[1]))

            # Confidence based on distance from boundary at centroid
            dist_transform = distance_transform_edt(cell_binary)
            confidence = float(dist_transform[y, x])

        elif strategy == 'random':
            # Random interior point
            coords = np.where(cell_binary)
            if len(coords[0]) == 0:
                continue
            idx = np.random.randint(len(coords[0]))
            y, x = coords[0][idx], coords[1][idx]

            # Confidence based on distance from boundary
            dist_transform = distance_transform_edt(cell_binary)
            confidence = float(dist_transform[y, x])

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        prompts[int(cell_id)] = {
            'point': [int(x), int(y)],  # SAM2 expects [x, y] format
            'label': 1,  # Positive prompt
            'area': int(cell_area),
            'confidence': confidence
        }

    return prompts


def generate_negative_prompts(
    instance_mask: np.ndarray,
    num_negative: int = 5,
    min_distance_from_cells: int = 5
) -> List[Dict]:
    """
    Generate negative point prompts from background regions.

    Args:
        instance_mask: Instance segmentation mask
        num_negative: Number of negative prompts to generate
        min_distance_from_cells: Minimum pixels from any cell

    Returns:
        List of negative prompts with format:
        [{'point': [x, y], 'label': 0}, ...]
    """
    # Create binary mask of all cells
    cell_mask = (instance_mask > 0).astype(np.uint8)

    # Compute distance from cells
    background_dist = distance_transform_edt(1 - cell_mask)

    # Find background pixels far from cells
    valid_background = background_dist >= min_distance_from_cells
    bg_coords = np.where(valid_background)

    negative_prompts = []

    if len(bg_coords[0]) > 0:
        # Sample negative points
        num_samples = min(num_negative, len(bg_coords[0]))
        indices = np.random.choice(len(bg_coords[0]), num_samples, replace=False)

        for idx in indices:
            y, x = bg_coords[0][idx], bg_coords[1][idx]
            negative_prompts.append({
                'point': [int(x), int(y)],
                'label': 0,  # Negative prompt
                'confidence': float(background_dist[y, x])
            })

    return negative_prompts


def save_instance_mask(
    instance_mask: np.ndarray,
    output_path: str,
    compress: bool = True
) -> str:
    """
    Save instance mask as numpy array.

    Args:
        instance_mask: Instance segmentation mask
        output_path: Full path to save the mask
        compress: Whether to compress the file

    Returns:
        Path to saved file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if compress:
        np.savez_compressed(output_path, mask=instance_mask)
        return output_path
    else:
        np.save(output_path, instance_mask)
        return output_path


def save_prompts(
    prompts: Dict,
    output_path: str,
    negative_prompts: Optional[List] = None
) -> str:
    """
    Save point prompts as JSON file.

    Args:
        prompts: Dictionary of cell prompts
        output_path: Full path to save the prompts
        negative_prompts: Optional list of negative prompts

    Returns:
        Path to saved file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Prepare data for JSON serialization
    data = {
        'positive_prompts': prompts,
        'negative_prompts': negative_prompts or [],
        'num_cells': len(prompts),
        'version': '1.0'
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    return output_path


def visualize_instance_mask(
    instance_mask: np.ndarray,
    prompts: Optional[Dict] = None,
    colormap: str = 'tab20'
) -> np.ndarray:
    """
    Create a visualization of instance mask with optional prompts.

    Args:
        instance_mask: Instance segmentation mask
        prompts: Optional dictionary of point prompts
        colormap: Matplotlib colormap name

    Returns:
        RGB visualization image
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # Create colored visualization
    cmap = cm.get_cmap(colormap)
    num_cells = instance_mask.max()

    # Normalize IDs to colormap range
    normalized_mask = instance_mask.astype(float) / max(num_cells, 1)
    colored_mask = cmap(normalized_mask)[:, :, :3]
    colored_mask[instance_mask == 0] = [0, 0, 0]  # Black background

    # Convert to uint8
    vis_image = (colored_mask * 255).astype(np.uint8)

    # Add prompt points if provided
    if prompts:
        for cell_id, prompt_info in prompts.items():
            x, y = prompt_info['point']
            # Draw a small cross at prompt location
            cv2.drawMarker(
                vis_image,
                (x, y),
                color=(255, 255, 255),  # White
                markerType=cv2.MARKER_CROSS,
                markerSize=5,
                thickness=1
            )

    return vis_image


def process_batch_for_sam2(
    pouch,
    time_step: int,
    output_dir: str,
    image_name: str,
    dataset_split: str = 'train',
    save_visualization: bool = False
) -> Dict:
    """
    Process a single time step and save SAM2-ready data.

    Args:
        pouch: Pouch simulation object
        time_step: Time step to process
        output_dir: Base dataset directory
        image_name: Base name for this image
        dataset_split: 'train', 'val', or 'test'
        save_visualization: Whether to save debug visualizations

    Returns:
        Dictionary with paths to saved files
    """
    results = {}

    # Get active cells at this time step
    active_cells = pouch.get_active_cells(time_step)

    if not active_cells:
        return results  # No active cells, skip

    # Get instance mask with only active cells
    multi_instance_mask = pouch.get_cell_masks(active_only=True, time_step=time_step)

    # Create clean instance mask with sequential IDs
    instance_mask = create_instance_mask_from_cells(multi_instance_mask, active_cells)

    # Generate point prompts
    prompts = generate_point_prompts_from_mask(instance_mask, strategy='distance_based')

    # Generate a few negative prompts
    negative_prompts = generate_negative_prompts(instance_mask, num_negative=5)

    # Save instance mask as .npy
    mask_path = os.path.join(output_dir, 'masks', dataset_split, f'{image_name}.npz')
    save_instance_mask(instance_mask, mask_path, compress=True)
    results['mask_path'] = mask_path

    # Save prompts as JSON
    prompt_path = os.path.join(output_dir, 'prompts', dataset_split, f'{image_name}.json')
    save_prompts(prompts, prompt_path, negative_prompts)
    results['prompt_path'] = prompt_path

    # Save visualization if requested
    if save_visualization:
        vis_image = visualize_instance_mask(instance_mask, prompts)
        vis_path = os.path.join(output_dir, 'visualizations', dataset_split, f'{image_name}_vis.png')
        os.makedirs(os.path.dirname(vis_path), exist_ok=True)
        cv2.imwrite(vis_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        results['vis_path'] = vis_path

    # Return statistics
    results['num_cells'] = len(prompts)
    results['num_negative'] = len(negative_prompts)
    results['total_pixels'] = int(np.sum(instance_mask > 0))

    return results


def create_dataset_metadata(
    output_dir: str,
    dataset_split: str = 'train'
) -> str:
    """
    Create metadata file for the dataset.

    Args:
        output_dir: Base dataset directory
        dataset_split: Dataset split to process

    Returns:
        Path to metadata file
    """
    from pathlib import Path

    # Collect all files
    image_dir = Path(output_dir) / 'images' / dataset_split
    mask_dir = Path(output_dir) / 'masks' / dataset_split
    prompt_dir = Path(output_dir) / 'prompts' / dataset_split

    image_files = sorted(image_dir.glob('*.png'))

    metadata = {
        'dataset_split': dataset_split,
        'num_images': len(image_files),
        'samples': []
    }

    for img_path in image_files:
        stem = img_path.stem

        # Check if corresponding files exist
        mask_path = mask_dir / f'{stem}.npz'
        prompt_path = prompt_dir / f'{stem}.json'

        if mask_path.exists() and prompt_path.exists():
            # Load prompt info for statistics
            with open(prompt_path, 'r') as f:
                prompt_data = json.load(f)

            metadata['samples'].append({
                'image': str(img_path.relative_to(output_dir)),
                'mask': str(mask_path.relative_to(output_dir)),
                'prompts': str(prompt_path.relative_to(output_dir)),
                'num_cells': prompt_data['num_cells']
            })

    # Save metadata
    metadata_path = Path(output_dir) / f'metadata_{dataset_split}.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Created metadata for {len(metadata['samples'])} samples in {dataset_split}")
    return str(metadata_path)