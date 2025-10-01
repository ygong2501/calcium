"""
Label generation utilities for calcium simulation.

This module provides functionality for:
- Generating labels (bounding boxes, segmentation masks, cell clusters)
- Saving label data to JSON files
- Creating CSV mappings between images and masks
"""
import os
import json
import re
from typing import Dict, List, Tuple, Set, Optional

import numpy as np
from skimage import measure


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types."""

    def default(self, obj):
        """Convert NumPy types to Python native types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def generate_bounding_boxes(cells_mask: np.ndarray, active_cells: List[int]) -> List[List[int]]:
    """
    Generate bounding boxes for active cells.

    Args:
        cells_mask: Cell mask array where each pixel value represents a cell ID.
        active_cells: List of active cell IDs.

    Returns:
        List of bounding boxes in format [x_min, y_min, width, height].
    """
    bounding_boxes = []

    for cell_id in active_cells:
        # Create binary mask for this cell
        cell_mask = (cells_mask == cell_id)

        # Find coordinates of mask pixels
        y_indices, x_indices = np.where(cell_mask)

        if len(x_indices) > 0 and len(y_indices) > 0:
            # Calculate bounding box
            x_min = int(np.min(x_indices))
            y_min = int(np.min(y_indices))
            x_max = int(np.max(x_indices))
            y_max = int(np.max(y_indices))

            width = x_max - x_min + 1
            height = y_max - y_min + 1

            bounding_boxes.append([x_min, y_min, width, height])

    return bounding_boxes


def generate_segmentation_masks(cells_mask: np.ndarray, active_cells: List[int]) -> Dict[int, List[List[float]]]:
    """
    Generate segmentation masks (polygon contours) for active cells.

    Args:
        cells_mask: Cell mask array where each pixel value represents a cell ID.
        active_cells: List of active cell IDs.

    Returns:
        Dictionary mapping cell IDs to lists of polygon contours.
    """
    segmentation_masks = {}

    for cell_id in active_cells:
        # Create binary mask for this cell
        cell_mask = (cells_mask == cell_id).astype(np.uint8)

        # Find contours using scikit-image
        contours = measure.find_contours(cell_mask, 0.5)

        if contours:
            # Convert contours to polygon format
            polygons = [contour.ravel().tolist() for contour in contours]
            segmentation_masks[cell_id] = polygons

    return segmentation_masks


def identify_cell_clusters(active_cells: List[int], adjacency_matrix: np.ndarray) -> List[List[int]]:
    """
    Identify clusters of spatially connected active cells using graph traversal.

    Args:
        active_cells: List of active cell IDs.
        adjacency_matrix: Cell adjacency matrix (shape: [n_cells, n_cells]).

    Returns:
        List of cell clusters, where each cluster is a list of connected cell IDs.
    """
    # Use set for O(1) membership checks
    unassigned = set(active_cells)
    clusters = []

    while unassigned:
        # Start a new cluster with first unassigned cell
        current_cell = unassigned.pop()
        cluster = [current_cell]
        to_check = [current_cell]

        # BFS to expand cluster
        while to_check:
            cell = to_check.pop(0)

            # Find adjacent active cells
            for neighbor_idx in range(adjacency_matrix.shape[1]):
                if adjacency_matrix[cell, neighbor_idx] > 0 and neighbor_idx in unassigned:
                    cluster.append(neighbor_idx)
                    to_check.append(neighbor_idx)
                    unassigned.remove(neighbor_idx)

        clusters.append(cluster)

    return clusters


def generate_labels(pouch, time_step: int, threshold: float = 0.1) -> Dict:
    """
    Generate comprehensive labels for a given time step.

    Args:
        pouch: Pouch object containing simulation state.
        time_step: Time step index to generate labels for.
        threshold: Activity threshold for determining active cells.

    Returns:
        Dictionary containing:
        - metadata: Simulation metadata
        - active_cells: List of active cell IDs
        - cell_clusters: List of connected cell clusters
        - bounding_boxes: List of bounding boxes for active cells
        - activity_values: Dictionary of calcium concentration values
        - segmentation_masks: Dictionary of polygon contours
        - parameters: Simulation parameters
    """
    # Get cell masks and active cells
    cells_mask = pouch.get_cell_masks()
    active_cells = pouch.get_active_cells(time_step, threshold)

    # Generate bounding boxes
    bounding_boxes = generate_bounding_boxes(cells_mask, active_cells)

    # Generate segmentation masks (polygon contours)
    segmentation_masks = generate_segmentation_masks(cells_mask, active_cells)

    # Identify cell clusters
    cell_clusters = identify_cell_clusters(active_cells, pouch.adj_matrix)

    # Get calcium concentration values for active cells
    activity_values = {
        cell_id: float(pouch.disc_dynamics[cell_id, 0, time_step])
        for cell_id in active_cells
    }

    # Construct label data
    label_data = {
        'metadata': {
            'simulation_id': pouch.sim_number,
            'simulation_type': pouch.save_name,
            'pouch_size': pouch.size,
            'time_step': time_step,
            'num_active_cells': len(active_cells)
        },
        'active_cells': active_cells,
        'cell_clusters': cell_clusters,
        'bounding_boxes': bounding_boxes,
        'activity_values': activity_values,
        'segmentation_masks': segmentation_masks,
        'parameters': pouch.param_dict
    }

    return label_data


def save_label(label_data: Dict, output_path: str, filename: str) -> str:
    """
    Save label data to a JSON file.

    Args:
        label_data: Label data dictionary.
        output_path: Output directory path.
        filename: Filename (with or without .json extension).

    Returns:
        Full path to the saved file.
    """
    # Create output directory if needed
    os.makedirs(output_path, exist_ok=True)

    # Ensure filename has .json extension
    if not filename.lower().endswith('.json'):
        filename += '.json'

    # Save to file
    file_path = os.path.join(output_path, filename)
    with open(file_path, 'w') as f:
        json.dump(label_data, f, indent=2, cls=NumpyEncoder)

    return file_path


def _extract_image_key_from_mask(mask_basename: str) -> Optional[Tuple[str, str]]:
    """
    Extract the corresponding image key from a mask filename.

    Supports multiple naming patterns:
    - {sim_name}_t{time_step}_{batch_run_id}_mask_combined.{ext}
    - {sim_name}_t{time_step}_mask_combined.{ext}
    - {sim_name}_mask_combined.{ext} (legacy)

    Args:
        mask_basename: Mask filename (basename only).

    Returns:
        Tuple of (image_key, extension) if pattern matches, None otherwise.
    """
    # Pattern 1: With timestamp - e.g., "Intercellular_transients_0_t00000_20250413231243_mask_combined.jpg"
    match = re.match(r'(.+?_t\d{5})_\d{14}_mask_combined(\.\w+)$', mask_basename)
    if match:
        return match.group(1), match.group(2)

    # Pattern 2: Without timestamp - e.g., "Intercellular_transients_0_t00000_mask_combined.jpg"
    match = re.match(r'(.+?_t\d{5})_mask_combined(\.\w+)$', mask_basename)
    if match:
        return match.group(1), match.group(2)

    # Pattern 3: Legacy without time step - e.g., "Intercellular_transients_0_mask_combined.jpg"
    match = re.match(r'(.+?)_mask_combined(\.\w+)$', mask_basename)
    if match:
        return match.group(1), match.group(2)

    return None


def create_image_mask_mapping(image_paths: List[str], mask_paths: List[str]) -> List[Tuple[str, str]]:
    """
    Create image-to-mask mappings from file paths.

    Each image is mapped to exactly one combined mask file.
    Only exact matches (where mask has corresponding image) are included.

    Args:
        image_paths: List of image file paths.
        mask_paths: List of mask file paths.

    Returns:
        List of tuples (image_basename, mask_basename) sorted by image name.
    """
    # Build image lookup set
    image_basenames = {os.path.basename(path) for path in image_paths}

    # Build image-to-mask mapping (ensures 1:1 relationship)
    image_to_mask = {}

    for mask_path in mask_paths:
        mask_basename = os.path.basename(mask_path)

        # Extract corresponding image key
        result = _extract_image_key_from_mask(mask_basename)
        if not result:
            continue

        img_key_base, img_extension = result
        img_key = f"{img_key_base}{img_extension}"

        # Only add if corresponding image exists and not already mapped
        if img_key in image_basenames and img_key not in image_to_mask:
            image_to_mask[img_key] = mask_basename

    # Convert to sorted list of tuples
    return sorted(image_to_mask.items())


def save_csv_mapping(mappings: List[Tuple[str, str]], output_path: str, filename: str = "train.csv") -> str:
    """
    Save image-mask mappings to a CSV file.

    Args:
        mappings: List of (image_basename, mask_basename) tuples.
        output_path: Output directory path.
        filename: CSV filename (default: "train.csv").

    Returns:
        Full path to the saved CSV file.
    """
    import csv

    # Create output directory if needed
    os.makedirs(output_path, exist_ok=True)

    # Ensure filename has .csv extension
    if not filename.lower().endswith('.csv'):
        filename += '.csv'

    file_path = os.path.join(output_path, filename)

    # Write CSV file
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["ImageId", "MaskId"])  # Header
        writer.writerows(mappings)  # Data rows

    return file_path


def create_dataset_csv_mapping(dataset_dir: str, filename: str = "train.csv") -> Optional[str]:
    """
    Create a CSV mapping file for an entire dataset directory.

    Scans the 'images' and 'masks' subdirectories and creates mappings.

    Args:
        dataset_dir: Base directory containing 'images' and 'masks' subdirectories.
        filename: CSV filename (default: "train.csv").

    Returns:
        Full path to the saved CSV file, or None if directories don't exist.
    """
    import glob

    img_dir = os.path.join(dataset_dir, 'images')
    mask_dir = os.path.join(dataset_dir, 'masks')

    # Check if directories exist
    if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
        print(f"Warning: 'images' or 'masks' directory not found in {dataset_dir}")
        return None

    # Find all image files (jpg and png)
    image_paths = glob.glob(os.path.join(img_dir, '*.jpg')) + \
                  glob.glob(os.path.join(img_dir, '*.png'))

    # Find all combined mask files
    mask_paths = glob.glob(os.path.join(mask_dir, '*_mask_combined.jpg')) + \
                 glob.glob(os.path.join(mask_dir, '*_mask_combined.png'))

    if not image_paths or not mask_paths:
        print(f"Warning: No images or masks found in {dataset_dir}")
        print(f"  Images: {len(image_paths)}, Masks: {len(mask_paths)}")
        return None

    # Create mappings
    mappings = create_image_mask_mapping(image_paths, mask_paths)

    # Save to CSV
    csv_path = save_csv_mapping(mappings, dataset_dir, filename)

    # Print statistics
    print(f"CSV mapping created: {csv_path}")
    print(f"  Total mappings: {len(mappings)}")
    print(f"  Total images: {len(image_paths)}")
    print(f"  Total masks: {len(mask_paths)}")
    print(f"  Unmatched: {len(mask_paths) - len(mappings)}")

    return csv_path
