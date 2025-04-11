"""
Optical defects for calcium simulation images.
"""
import numpy as np
import cv2
from scipy import ndimage


# Radial distortion function has been removed


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
    
    # Radial distortion application has been removed
    
    # Apply chromatic aberration
    if config.get('chromatic_aberration', False):
        offset = config.get('chromatic_offset', 3)
        result = add_chromatic_aberration(result, offset)
    
    # Apply vignetting
    if config.get('vignetting', False):
        strength = config.get('vignetting_strength', 0.5)
        result = add_vignetting(result, strength)
    
    return result