"""
Label generation utilities for calcium simulation.
"""
import os
import json
import numpy as np
import cv2
from skimage import measure


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
        # Convert any numpy values to Python native types
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Use custom converter for numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                return convert_numpy(obj) or super(NumpyEncoder, self).default(obj)
        
        json.dump(label_data, f, indent=4, cls=NumpyEncoder)
    
    return file_path