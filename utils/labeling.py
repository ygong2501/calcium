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


def create_csv_mapping(image_files, mask_files, output_path, filename="image_mask_mapping.csv"):
    """
    Create a CSV file mapping original images to their mask files.
    
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
    
    # Create a mapping of images to masks
    image_to_masks = {}
    
    # Extract base filenames from full paths
    for img_path in image_files:
        img_basename = os.path.basename(img_path)
        # Use the complete filename with extension as the ImageId
        image_to_masks[img_basename] = []
    
    # Find all masks for each image
    for mask_path in mask_files:
        mask_basename = os.path.basename(mask_path)
        # Extract original image name from mask name
        # Example: "Intercellular_waves_42_t00050_mask_007.jpg" -> "Intercellular_waves_42_t00050.jpg"
        match = re.match(r'(.+?)_mask_\d+(\.\w+)$', mask_basename)
        if match:
            img_key_base = match.group(1)
            img_extension = match.group(2)
            img_key = f"{img_key_base}{img_extension}"
            
            if img_key in image_to_masks:
                image_to_masks[img_key].append(mask_basename)
    
    # Write the CSV file with the direct filename mapping
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(["ImageId", "MaskId"])
        
        # Write data rows using complete filenames
        for img_key, masks in image_to_masks.items():
            # If no masks, continue to next image
            if not masks:
                continue
                
            # For each mask of this image, write a row
            for mask_name in masks:
                writer.writerow([img_key, mask_name])
    
    return file_path


def create_dataset_csv_mapping(batch_dir, filename="image_mask_mapping.csv"):
    """
    Create a CSV file mapping original images to their mask files for an entire dataset.
    This function scans the batch directory to find all images and masks.
    
    Args:
        batch_dir (str): Base directory containing simulation results.
        filename (str, optional): CSV filename.
    
    Returns:
        str: Full path to saved CSV file.
    """
    # Find all simulation directories
    sim_dirs = [d for d in os.listdir(batch_dir) 
                if os.path.isdir(os.path.join(batch_dir, d)) and not d.startswith('.')]
    
    all_images = []
    all_masks = []
    
    # Find all images and masks
    for sim_dir in sim_dirs:
        sim_path = os.path.join(batch_dir, sim_dir)
        img_dir = os.path.join(sim_path, 'images')
        mask_dir = os.path.join(sim_path, 'masks')
        
        # Skip if directories don't exist
        if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
            continue
        
        # Find all image files (both jpg and png)
        images = glob.glob(os.path.join(img_dir, '*.jpg')) + glob.glob(os.path.join(img_dir, '*.png'))
        masks = glob.glob(os.path.join(mask_dir, '*_mask_*.jpg')) + glob.glob(os.path.join(mask_dir, '*_mask_*.png'))
        
        all_images.extend(images)
        all_masks.extend(masks)
    
    # Create the mapping if we found images and masks
    if all_images and all_masks:
        return create_csv_mapping(all_images, all_masks, batch_dir, filename)
    else:
        print(f"Warning: No images or masks found in {batch_dir} (Images: {len(all_images)}, Masks: {len(all_masks)})")
        # Return a dummy file path as the CSV wasn't created
        return os.path.join(batch_dir, filename)