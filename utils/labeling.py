"""
Label generation utilities for calcium simulation.
"""
import os
import json
import numpy as np
import cv2
import csv
import re
from skimage import measure
import glob


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def generate_bounding_boxes(cells_mask, active_cells):
    """
    Generate bounding boxes for active cells.
    
    Args:
        cells_mask (numpy.ndarray): Cell mask array.
        active_cells (list): List of active cell IDs.
    
    Returns:
        list: List of bounding boxes [x_min, y_min, width, height]
    """
    bounding_boxes = []
    
    for cell_id in active_cells:
        # Create binary mask for this cell
        cell_mask = (cells_mask == cell_id)
        
        # Find coordinates of mask pixels
        y_indices, x_indices = np.where(cell_mask)
        
        if len(x_indices) > 0 and len(y_indices) > 0:
            # Calculate bounding box
            x_min = np.min(x_indices)
            y_min = np.min(y_indices)
            x_max = np.max(x_indices)
            y_max = np.max(y_indices)
            
            width = x_max - x_min + 1
            height = y_max - y_min + 1
            
            bounding_boxes.append([int(x_min), int(y_min), int(width), int(height)])
    
    return bounding_boxes


def generate_segmentation_masks(cells_mask, active_cells):
    """
    Generate segmentation masks for active cells.
    
    Args:
        cells_mask (numpy.ndarray): Cell mask array.
        active_cells (list): List of active cell IDs.
    
    Returns:
        dict: Dictionary mapping cell IDs to segmentation polygons
    """
    segmentation_masks = {}
    
    for cell_id in active_cells:
        # Create binary mask for this cell
        cell_mask = (cells_mask == cell_id).astype(np.uint8)
        
        # Find contours
        contours = measure.find_contours(cell_mask, 0.5)
        
        if len(contours) > 0:
            # Convert to polygon format
            polygons = []
            for contour in contours:
                # Simplify contour to reduce points
                polygon = contour.ravel().tolist()
                polygons.append(polygon)
            
            segmentation_masks[cell_id] = polygons
    
    return segmentation_masks


def identify_cell_clusters(cells_mask, active_cells, adjacency_matrix):
    """
    Identify clusters of connected active cells.
    
    Args:
        cells_mask (numpy.ndarray): Cell mask array.
        active_cells (list): List of active cell IDs.
        adjacency_matrix (numpy.ndarray): Cell adjacency matrix.
    
    Returns:
        list: List of cell clusters (lists of cell IDs)
    """
    # Convert to set for faster lookups
    active_set = set(active_cells)
    
    # Initialize clusters
    clusters = []
    unassigned = set(active_cells)
    
    while unassigned:
        # Start a new cluster with first unassigned cell
        current_cell = next(iter(unassigned))
        cluster = [current_cell]
        to_check = [current_cell]
        unassigned.remove(current_cell)
        
        # Expand cluster
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


def generate_labels(pouch, time_step, threshold=0.1):
    """
    Generate labels for current time step.
    
    Args:
        pouch: Pouch object.
        time_step (int): Time step.
        threshold (float): Activity threshold.
    
    Returns:
        dict: Label data dictionary.
    """
    # Get cell masks
    cells_mask = pouch.get_cell_masks()
    
    # Get active cells
    active_cells = pouch.get_active_cells(time_step, threshold)
    
    # Generate bounding boxes
    bounding_boxes = generate_bounding_boxes(cells_mask, active_cells)
    
    # Generate segmentation masks
    segmentation_masks = generate_segmentation_masks(cells_mask, active_cells)
    
    # Identify cell clusters
    cell_clusters = identify_cell_clusters(cells_mask, active_cells, pouch.adj_matrix)
    
    # Get intensity values for active cells
    activity_values = {}
    for cell_id in active_cells:
        activity_values[cell_id] = float(pouch.disc_dynamics[cell_id, 0, time_step])
    
    # Create label data
    label_data = {
        'metadata': {
            'simulation_id': pouch.sim_number,
            'simulation_type': pouch.save_name,
            'pouch_size': pouch.size,
            'time_step': time_step
        },
        'active_cells': active_cells,
        'cell_clusters': cell_clusters,
        'bounding_boxes': bounding_boxes,
        'activity_values': activity_values,
        'segmentation_masks': segmentation_masks,
        'parameters': pouch.param_dict
    }
    
    return label_data


def save_label(label_data, output_path, filename):
    """
    Save label data to JSON file.
    
    Args:
        label_data (dict): Label data.
        output_path (str): Output directory.
        filename (str): Filename without extension.
    
    Returns:
        str: Full path to saved file.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    
    # Ensure filename has extension
    if not filename.lower().endswith('.json'):
        filename += '.json'
    
    # Full path to output file
    file_path = os.path.join(output_path, filename)
    
    # Save label data
    with open(file_path, 'w') as f:
        json.dump(label_data, f, indent=4, cls=NumpyEncoder)
    
    return file_path


def create_csv_mapping(image_files, mask_files, output_path, filename="train.csv"):
    """
    Create a CSV file mapping original images to their combined mask files.
    Each image is mapped to a single mask file (the combined mask).
    Only exact matches (where mask has corresponding image) are included.
    Removes duplicate image-mask pairs to ensure 1:1 mapping.
    
    Args:
        image_files (list): List of original image file paths.
        mask_files (list): List of mask image file paths.
        output_path (str): Output directory.
        filename (str, optional): CSV filename.
    
    Returns:
        str: Full path to saved CSV file.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    
    # Ensure filename has extension
    if not filename.lower().endswith('.csv'):
        filename += '.csv'
    
    # Full path to output file
    file_path = os.path.join(output_path, filename)
    
    # Convert image paths to base filenames for quick lookup
    image_basenames = set()
    image_dict = {}
    for img_path in image_files:
        img_basename = os.path.basename(img_path)
        image_basenames.add(img_basename)
        image_dict[img_basename] = img_path
    
    # Find valid image-mask pairs, using a dict to ensure each image has only one mask
    # This prevents duplicate entries in the CSV
    image_to_mask_dict = {}
    
    # Process masks
    for mask_path in mask_files:
        mask_basename = os.path.basename(mask_path)
        # Updated pattern to match both old and new naming conventions (with batch ID)
        match = re.match(r'(.+?)_\d{14}_mask_combined(\.\w+)$', mask_basename)
        if not match:
            # Try old pattern as fallback
            match = re.match(r'(.+?)_mask_combined(\.\w+)$', mask_basename)
        
        if match:
            img_key_base = match.group(1)
            img_extension = match.group(2)
            img_key = f"{img_key_base}{img_extension}"
            
            # Only add the pair if the corresponding image exists
            if img_key in image_basenames and img_key not in image_to_mask_dict:
                image_to_mask_dict[img_key] = mask_basename
    
    # Convert to list of pairs for sorting
    valid_pairs = list(image_to_mask_dict.items())
    
    # Write the CSV file with the direct filename mapping
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(["ImageId", "MaskId"])
        
        # Sort the pairs for consistency
        valid_pairs.sort()
        
        # Write data rows
        for img_key, mask_name in valid_pairs:
            writer.writerow([img_key, mask_name])
    
    # Report statistics
    print(f"CSV mapping created with {len(valid_pairs)} valid image-mask pairs")
    print(f"Total unique images: {len(image_basenames)}")
    print(f"Total masks files: {len(mask_files)}")
    print(f"Duplicate entries removed: {len(mask_files) - len(valid_pairs)}")
    
    return file_path


def create_dataset_csv_mapping(batch_dir, filename="train.csv"):
    """
    Create a CSV file mapping original images to their combined mask files for an entire dataset.
    This function directly scans the images and masks directories.
    
    Args:
        batch_dir (str): Base directory containing the images and masks directories.
        filename (str, optional): CSV filename.
    
    Returns:
        str: Full path to saved CSV file.
    """
    # Directly use the images and masks directories
    img_dir = os.path.join(batch_dir, 'images')
    mask_dir = os.path.join(batch_dir, 'masks')
    
    all_images = []
    all_masks = []
    
    # Check if directories exist
    if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
        print(f"Warning: Image or mask directory not found in {batch_dir}")
        return os.path.join(batch_dir, filename)
    
    # Find all image files (both jpg and png)
    images = glob.glob(os.path.join(img_dir, '*.jpg')) + glob.glob(os.path.join(img_dir, '*.png'))
    
    # Only look for combined mask files (ending with _mask_combined.jpg/png)
    masks = glob.glob(os.path.join(mask_dir, '*_mask_combined.jpg')) + glob.glob(os.path.join(mask_dir, '*_mask_combined.png'))
    
    all_images.extend(images)
    all_masks.extend(masks)
    
    # Create the mapping if we found images and masks
    if all_images and all_masks:
        return create_csv_mapping(all_images, all_masks, batch_dir, filename)
    else:
        print(f"Warning: No images or combined masks found in {batch_dir} (Images: {len(all_images)}, Masks: {len(all_masks)})")
        # Return a dummy file path as the CSV wasn't created
        return os.path.join(batch_dir, filename)


def append_to_existing_csv(batch_dir, existing_csv_path):
    """
    Append new image-mask mappings to an existing CSV file.
    
    Args:
        batch_dir (str): Base directory containing the images and masks directories.
        existing_csv_path (str): Path to the existing CSV file.
    
    Returns:
        str: Full path to the updated CSV file.
    """
    # Get current mappings from the existing CSV
    existing_mappings = set()
    with open(existing_csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        # Skip header
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                existing_mappings.add((row[0], row[1]))
    
    # Find new image-mask pairs
    img_dir = os.path.join(batch_dir, 'images')
    mask_dir = os.path.join(batch_dir, 'masks')
    
    # Check if directories exist
    if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
        print(f"Warning: Image or mask directory not found in {batch_dir}")
        return existing_csv_path
    
    # Find all image files (both jpg and png)
    images = glob.glob(os.path.join(img_dir, '*.jpg')) + glob.glob(os.path.join(img_dir, '*.png'))
    
    # Only look for combined mask files (ending with _mask_combined.jpg/png)
    masks = glob.glob(os.path.join(mask_dir, '*_mask_combined.jpg')) + glob.glob(os.path.join(mask_dir, '*_mask_combined.png'))
    
    # Create image-to-mask dict with all image paths
    image_basenames = set()
    image_dict = {}
    for img_path in images:
        img_basename = os.path.basename(img_path)
        image_basenames.add(img_basename)
        image_dict[img_basename] = img_path
    
    # Find valid image-mask pairs, ensure no duplicates
    new_pairs = []
    
    # Process masks
    for mask_path in masks:
        mask_basename = os.path.basename(mask_path)
        # Updated pattern to match both old and new naming conventions (with batch ID)
        match = re.match(r'(.+?)_\d{14}_mask_combined(\.\w+)$', mask_basename)
        if not match:
            # Try old pattern as fallback
            match = re.match(r'(.+?)_mask_combined(\.\w+)$', mask_basename)
        
        if match:
            img_key_base = match.group(1)
            img_extension = match.group(2)
            img_key = f"{img_key_base}{img_extension}"
            
            # Only add the pair if the corresponding image exists
            if img_key in image_basenames:
                pair = (img_key, mask_basename)
                # Only add if not already in existing mappings
                if pair not in existing_mappings:
                    new_pairs.append(pair)
    
    # Append new pairs to the existing CSV
    with open(existing_csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write new data rows
        for img_key, mask_name in sorted(new_pairs):
            writer.writerow([img_key, mask_name])
    
    # Report statistics
    print(f"CSV mapping updated with {len(new_pairs)} new image-mask pairs")
    print(f"Total existing mappings: {len(existing_mappings)}")
    print(f"Total mappings now: {len(existing_mappings) + len(new_pairs)}")
    
    return existing_csv_path