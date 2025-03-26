"""
Background defects for calcium simulation images.
"""
import numpy as np
import cv2


def add_background_fluorescence(image, intensity=0.1, non_uniform=True):
    """
    Add background fluorescence.
    
    Args:
        image (numpy.ndarray): Input image.
        intensity (float): Intensity of the background fluorescence (0-1).
        non_uniform (bool): If True, adds non-uniform background.
    
    Returns:
        numpy.ndarray: Image with background fluorescence.
    """
    result = image.copy()
    max_val = np.max(image)
    
    if non_uniform:
        # Create a non-uniform background with some gradient and noise
        rows, cols = image.shape[:2]
        
        # Create a gradient
        y, x = np.mgrid[0:rows, 0:cols]
        gradient = np.sin(x/cols*3*np.pi) * np.sin(y/rows*2*np.pi) * 0.5 + 0.5
        
        # Add some low-frequency noise
        noise = cv2.GaussianBlur(
            np.random.rand(rows, cols), (99, 99), 30
        )
        
        # Combine gradient and noise
        background = (gradient * 0.7 + noise * 0.3) * intensity * max_val
        
        # Add background to each channel if RGB
        if len(image.shape) == 3:
            for i in range(image.shape[2]):
                result[:, :, i] = np.clip(
                    result[:, :, i] + background, 0, max_val
                )
        else:
            result = np.clip(result + background, 0, max_val)
    else:
        # Add uniform background
        uniform_value = intensity * max_val
        if len(image.shape) == 3:
            for i in range(image.shape[2]):
                result[:, :, i] = np.clip(
                    result[:, :, i] + uniform_value, 0, max_val
                )
        else:
            result = np.clip(result + uniform_value, 0, max_val)
    
    return result.astype(image.dtype)


def add_cell_spontaneous_luminescence(image, cells_mask, intensity_range=(0.05, 0.15), probability=0.2):
    """
    Add spontaneous fluorescence to non-active cells.
    
    Args:
        image (numpy.ndarray): Original image.
        cells_mask (numpy.ndarray): Cell mask array indicating which pixel belongs to which cell.
        intensity_range (tuple): Spontaneous fluorescence intensity range (min, max).
        probability (float): Probability of cells showing spontaneous fluorescence.
    
    Returns:
        numpy.ndarray: Image with spontaneous cell luminescence.
    """
    result = image.copy()
    max_val = np.max(image)
    
    # Get unique cell IDs (excluding background which is 0)
    cell_ids = np.unique(cells_mask)
    cell_ids = cell_ids[cell_ids > 0]  # Exclude background (0)
    
    # Randomly select cells to add luminescence
    num_cells = len(cell_ids)
    num_spontaneous = int(num_cells * probability)
    spontaneous_cells = np.random.choice(cell_ids, num_spontaneous, replace=False)
    
    # Add luminescence to each selected cell
    for cell_id in spontaneous_cells:
        # Create mask for this cell
        cell_mask = (cells_mask == cell_id)
        
        # Determine luminescence level for this cell
        intensity = np.random.uniform(intensity_range[0], intensity_range[1]) * max_val
        
        # Add luminescence to this cell
        if len(image.shape) == 3:
            # For RGB images, add primarily to green channel (index 1) for calcium
            green_factor = 1.0
            red_factor = 0.3
            blue_factor = 0.3
            
            result[:, :, 0][cell_mask] = np.clip(
                result[:, :, 0][cell_mask] + intensity * red_factor, 0, max_val
            )
            result[:, :, 1][cell_mask] = np.clip(
                result[:, :, 1][cell_mask] + intensity * green_factor, 0, max_val
            )
            result[:, :, 2][cell_mask] = np.clip(
                result[:, :, 2][cell_mask] + intensity * blue_factor, 0, max_val
            )
        else:
            # For grayscale images
            result[cell_mask] = np.clip(result[cell_mask] + intensity, 0, max_val)
    
    return result.astype(image.dtype)


def add_cell_fragments(image, num_fragments=5, size_range=(10, 30), intensity_range=(0.1, 0.3)):
    """
    Add blurry cell fragments to the image.
    
    Args:
        image (numpy.ndarray): Input image.
        num_fragments (int): Number of fragments to add.
        size_range (tuple): Size range of fragments (min, max).
        intensity_range (tuple): Intensity range of fragments (min, max).
    
    Returns:
        numpy.ndarray: Image with cell fragments.
    """
    result = image.copy()
    rows, cols = image.shape[:2]
    max_val = np.max(image)
    
    for _ in range(num_fragments):
        # Random position
        x = np.random.randint(0, cols)
        y = np.random.randint(0, rows)
        
        # Random size
        size = np.random.randint(size_range[0], size_range[1])
        
        # Random intensity
        intensity = np.random.uniform(intensity_range[0], intensity_range[1]) * max_val
        
        # Create the fragment (blurry circle)
        fragment = np.zeros((rows, cols), dtype=np.float32)
        cv2.circle(fragment, (x, y), size, 1.0, -1)
        fragment = cv2.GaussianBlur(fragment, (size//2*2+1, size//2*2+1), size/3)
        
        # Add fragment to the image
        if len(image.shape) == 3:
            # For RGB images, add primarily to green channel for calcium
            green_factor = 1.0
            red_factor = 0.2
            blue_factor = 0.2
            
            result[:, :, 0] = np.clip(
                result[:, :, 0] + fragment * intensity * red_factor, 0, max_val
            )
            result[:, :, 1] = np.clip(
                result[:, :, 1] + fragment * intensity * green_factor, 0, max_val
            )
            result[:, :, 2] = np.clip(
                result[:, :, 2] + fragment * intensity * blue_factor, 0, max_val
            )
        else:
            result = np.clip(result + fragment * intensity, 0, max_val)
    
    return result.astype(image.dtype)


def apply_background_defects(image, cells_mask, config):
    """
    Apply all background defects based on configuration.
    
    Args:
        image (numpy.ndarray): Input image.
        cells_mask (numpy.ndarray): Cell mask array.
        config (dict): Defect configuration parameters.
    
    Returns:
        numpy.ndarray: Image with background defects applied.
    """
    result = image.copy()
    
    # Add background fluorescence
    if config.get('background_fluorescence', False):
        intensity = config.get('background_intensity', 0.1)
        non_uniform = config.get('non_uniform_background', True)
        result = add_background_fluorescence(result, intensity, non_uniform)
    
    # Add spontaneous cell luminescence
    if config.get('spontaneous_luminescence', False):
        intensity_min = config.get('spontaneous_min', 0.05)
        intensity_max = config.get('spontaneous_max', 0.15)
        probability = config.get('spontaneous_probability', 0.2)
        result = add_cell_spontaneous_luminescence(
            result, cells_mask, (intensity_min, intensity_max), probability
        )
    
    # Add cell fragments
    if config.get('cell_fragments', False):
        num_fragments = config.get('fragment_count', 5)
        size_min = config.get('fragment_min_size', 10)
        size_max = config.get('fragment_max_size', 30)
        intensity_min = config.get('fragment_min_intensity', 0.1)
        intensity_max = config.get('fragment_max_intensity', 0.3)
        result = add_cell_fragments(
            result, num_fragments, (size_min, size_max), (intensity_min, intensity_max)
        )
    
    return result