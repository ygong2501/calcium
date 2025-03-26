"""
Optical defects for calcium simulation images.
"""
import numpy as np
import cv2
from scipy import ndimage


def add_radial_distortion(image, k1=0.1, k2=0.05):
    """
    Add radial distortion to simulate lens distortion.
    
    Args:
        image (numpy.ndarray): Input image.
        k1 (float): First radial distortion coefficient.
        k2 (float): Second radial distortion coefficient.
    
    Returns:
        numpy.ndarray: Distorted image.
    """
    rows, cols = image.shape[:2]
    
    # Create maps for the distortion
    center_x, center_y = cols / 2, rows / 2
    
    # Create coordinate grid
    y, x = np.indices((rows, cols))
    
    # Shift origin to center
    x = x - center_x
    y = y - center_y
    
    # Calculate radius^2
    r2 = x**2 + y**2
    r4 = r2**2
    
    # Calculate distortion
    distortion = 1 + k1 * r2 + k2 * r4
    
    # Apply distortion
    x_distorted = x * distortion + center_x
    y_distorted = y * distortion + center_y
    
    # Ensure coordinates are within bounds
    x_distorted = np.clip(x_distorted, 0, cols - 1)
    y_distorted = np.clip(y_distorted, 0, rows - 1)
    
    # Sample the image at the distorted coordinates
    # For RGB images
    if len(image.shape) == 3:
        distorted_image = np.zeros_like(image)
        for i in range(image.shape[2]):
            distorted_image[:, :, i] = ndimage.map_coordinates(
                image[:, :, i], [y_distorted.ravel(), x_distorted.ravel()], 
                order=1).reshape(rows, cols)
    else:
        # For grayscale images
        distorted_image = ndimage.map_coordinates(
            image, [y_distorted.ravel(), x_distorted.ravel()], 
            order=1).reshape(rows, cols)
    
    return distorted_image


def add_chromatic_aberration(image, offset=3):
    """
    Add chromatic aberration by shifting RGB channels.
    
    Args:
        image (numpy.ndarray): Input RGB image.
        offset (int): Pixel offset for channel shift.
    
    Returns:
        numpy.ndarray: Image with chromatic aberration.
    """
    if len(image.shape) < 3 or image.shape[2] < 3:
        # Cannot apply chromatic aberration to grayscale images
        return image.copy()
    
    height, width = image.shape[:2]
    result = image.copy()  # Start with the original image
    
    # Use cv2.warpAffine for better channel shifting without edge gaps
    M_red = np.float32([[1, 0, offset], [0, 1, 0]])
    M_blue = np.float32([[1, 0, -offset], [0, 1, 0]])
    
    # Shift red channel to the right
    red_channel = cv2.warpAffine(image[:, :, 0], M_red, (width, height), 
                                borderMode=cv2.BORDER_REFLECT)
    
    # Shift blue channel to the left
    blue_channel = cv2.warpAffine(image[:, :, 2], M_blue, (width, height), 
                                 borderMode=cv2.BORDER_REFLECT)
    
    # Apply the shifted channels
    result[:, :, 0] = red_channel
    result[:, :, 2] = blue_channel
    
    return result


def add_vignetting(image, strength=0.5):
    """
    Add vignetting effect (darkening at the edges).
    
    Args:
        image (numpy.ndarray): Input image.
        strength (float): Vignetting strength (0 to 1).
    
    Returns:
        numpy.ndarray: Image with vignetting effect.
    """
    rows, cols = image.shape[:2]
    
    # Create a normalized coordinate grid
    x = np.linspace(-1, 1, cols)
    y = np.linspace(-1, 1, rows)
    xv, yv = np.meshgrid(x, y)
    
    # Calculate distance from center (normalized)
    r = np.sqrt(xv**2 + yv**2)
    
    # Apply vignetting mask
    # Adjust formula to control vignetting profile
    vignette = 1 - strength * np.clip(r, 0, 1)**2
    
    # Apply vignetting to each channel
    result = image.copy()
    if len(image.shape) == 3:
        for i in range(image.shape[2]):
            result[:, :, i] = image[:, :, i] * vignette
    else:
        result = image * vignette
    
    return result.astype(image.dtype)


def apply_optical_defects(image, config):
    """
    Apply all optical defects based on configuration.
    
    Args:
        image (numpy.ndarray): Input image.
        config (dict): Defect configuration parameters.
    
    Returns:
        numpy.ndarray: Image with optical defects applied.
    """
    result = image.copy()
    
    # Apply radial distortion
    if config.get('radial_distortion', False):
        k1 = config.get('radial_k1', 0.1)
        k2 = config.get('radial_k2', 0.05)
        result = add_radial_distortion(result, k1, k2)
    
    # Apply chromatic aberration
    if config.get('chromatic_aberration', False):
        offset = config.get('chromatic_offset', 3)
        result = add_chromatic_aberration(result, offset)
    
    # Apply vignetting
    if config.get('vignetting', False):
        strength = config.get('vignetting_strength', 0.5)
        result = add_vignetting(result, strength)
    
    return result