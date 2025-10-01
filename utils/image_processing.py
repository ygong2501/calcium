"""
Image processing utilities for calcium simulation.
"""
import numpy as np
import cv2
from skimage import transform
import os

from artifacts.optical import apply_optical_defects
from artifacts.sensor import apply_sensor_defects
from artifacts.background import apply_background_defects


def resize_image(image, target_size=(512, 512)):
    """
    Resize image to target size.
    
    Args:
        image (numpy.ndarray): Input image.
        target_size (tuple): Target size (width, height).
    
    Returns:
        numpy.ndarray: Resized image.
    """
    current_height, current_width = image.shape[:2]
    if (current_width, current_height) == target_size:
        return image
    
    # Resize while preserving aspect ratio
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
    return resized


def apply_defocus_blur(image, kernel_size=9, sigma=3):
    """
    Simulate defocus blur.
    
    Args:
        image (numpy.ndarray): Input image.
        kernel_size (int): Convolution kernel size.
        sigma (float): Gaussian function standard deviation.
    
    Returns:
        numpy.ndarray: Blurred image.
    """
    # Ensure kernel size is odd
    kernel_size = max(3, kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def apply_partial_defocus(image, region_center=None, radius=100, blur_strength=3):
    """
    Apply defocus blur to only part of the image.
    
    Args:
        image (numpy.ndarray): Input image.
        region_center (tuple): Center of the blurred region (x, y).
        radius (int): Radius of the blurred region.
        blur_strength (float): Blur strength (sigma).
    
    Returns:
        numpy.ndarray: Partially blurred image.
    """
    rows, cols = image.shape[:2]
    
    # If no center specified, use a random position
    if region_center is None:
        region_center = (np.random.randint(cols // 4, 3 * cols // 4),
                        np.random.randint(rows // 4, 3 * rows // 4))
    
    # Create a blurred version of the entire image
    blurred = apply_defocus_blur(image, kernel_size=9, sigma=blur_strength)
    
    # Create a mask for blending
    mask = np.zeros((rows, cols), dtype=np.float32)
    cv2.circle(mask, region_center, radius, 1.0, -1)
    
    # Smooth the mask edges
    mask = cv2.GaussianBlur(mask, (radius//2*2+1, radius//2*2+1), radius/3)
    
    # Ensure mask is properly shaped for blending
    if len(image.shape) == 3:
        mask = np.expand_dims(mask, axis=2)
        mask = np.repeat(mask, image.shape[2], axis=2)
    
    # Blend original and blurred image using the mask
    result = image * (1 - mask) + blurred * mask
    
    return result.astype(image.dtype)


def adjust_brightness_contrast(image, brightness=0, contrast=1.0):
    """
    Adjust image brightness and contrast.
    
    Args:
        image (numpy.ndarray): Input image.
        brightness (float): Brightness adjustment [-1, 1].
        contrast (float): Contrast adjustment [0, 3].
    
    Returns:
        numpy.ndarray: Adjusted image.
    """
    # Convert to float for processing
    img_float = image.astype(np.float32)
    
    # Normalize to [0, 1] if needed
    if img_float.max() > 1.0:
        img_float /= 255.0
    
    # Apply contrast adjustment
    img_float = (img_float - 0.5) * contrast + 0.5
    
    # Apply brightness adjustment
    img_float += brightness
    
    # Clip values to valid range
    img_float = np.clip(img_float, 0, 1.0)
    
    # Return to original range
    if image.max() > 1.0:
        img_float *= 255.0
    
    return img_float.astype(image.dtype)


def apply_all_defects(image, cells_mask, config):
    """
    Apply complete defect pipeline.
    
    Args:
        image (numpy.ndarray): Original image (512×512).
        cells_mask (numpy.ndarray): Cell mask.
        config (dict): Defect configuration dictionary.
    
    Returns:
        numpy.ndarray: Image with all defects applied.
    """
    result = image.copy()
    
    # 1. Pre-optical defects (background and cell effects)
    result = apply_background_defects(result, cells_mask, config)
    
    # 2. Optical propagation defects
    result = apply_optical_defects(result, config)
    
    # 3. Sensor defects
    result = apply_sensor_defects(result, config)
    
    # 4. Post-processing defects
    
    # Apply defocus blur
    if config.get('defocus_blur', False):
        kernel_size = config.get('defocus_kernel_size', 9)
        sigma = config.get('defocus_sigma', 3)
        if config.get('partial_defocus', False):
            # Apply partial defocus
            region_center = config.get('defocus_center', None)
            radius = config.get('defocus_radius', 100)
            result = apply_partial_defocus(result, region_center, radius, sigma)
        else:
            # Apply global defocus
            result = apply_defocus_blur(result, kernel_size, sigma)
    
    # Apply brightness/contrast adjustments
    if config.get('adjust_brightness_contrast', False):
        brightness = config.get('brightness', 0)
        contrast = config.get('contrast', 1.0)
        result = adjust_brightness_contrast(result, brightness, contrast)
    
    # Don't enforce 512x512 - allow dynamic sizing based on Pouch output_size
    return result


def save_image(image, output_path, filename, format='png', quality=90, target_size=None, bit_depth=10):
    """
    Save image to file, optionally resizing to target size.

    Args:
        image (numpy.ndarray): Image to save.
        output_path (str): Output directory.
        filename (str): Filename.
        format (str): Image format, 'jpg', 'png', or 'png16' (default: 'png').
        quality (int): JPEG quality (0-100), higher is better quality. Only used for JPEG format.
        target_size (tuple, optional): Target size (width, height) to resize image before saving.
        bit_depth (int): Bit depth for PNG images (8, 10, or 16). 10-bit values stored in 16-bit PNG.

    Returns:
        str: Full path to saved file.

    Raises:
        ValueError: If image format is invalid
        IOError: If saving fails
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    # Normalize format to lowercase
    format = format.lower()

    # Get the base filename without extension
    base_filename = os.path.splitext(filename)[0]

    # Add appropriate extension based on specified format
    if format == 'jpg' or format == 'jpeg':
        file_extension = '.jpg'
    elif format in ['png', 'png16']:
        file_extension = '.png'
    else:
        file_extension = '.png'  # Default to PNG

    # Final filename with correct extension
    final_filename = base_filename + file_extension

    # Full path to output file
    file_path = os.path.join(output_path, final_filename)

    # Validate image
    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be a numpy array")

    if len(image.shape) not in [2, 3]:
        raise ValueError(f"Unexpected image shape: {image.shape}")

    # Resize image if target_size is specified
    if target_size is not None:
        image = resize_image(image, target_size)

    try:
        # Handle different formats
        if format == 'jpg' or format == 'jpeg':
            # Convert to uint8 for JPEG
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)

            # Check if grayscale or RGB
            if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
                cv2.imwrite(file_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            else:
                cv2.imwrite(file_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, quality])

        elif format in ['png', 'png16'] and bit_depth == 10:
            # 10-bit grayscale stored in 16-bit PNG
            # Convert to grayscale if RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # Convert to 10-bit range (0-1023) stored in uint16
            if image.dtype == np.uint8:
                # Scale from 8-bit (0-255) to 10-bit (0-1023)
                image_16bit = (image.astype(np.uint16) * 4)  # 255 * 4 = 1020 (close to 1023)
            elif image.max() <= 1.0:
                # Normalized float, scale to 10-bit
                image_16bit = (image * 1023).astype(np.uint16)
            else:
                # Assume already in appropriate range
                image_16bit = np.clip(image, 0, 1023).astype(np.uint16)

            # Save as 16-bit PNG
            cv2.imwrite(file_path, image_16bit)

        else:
            # Standard 8-bit PNG
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)

            # Check if grayscale or RGB
            if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
                cv2.imwrite(file_path, image)
            else:
                cv2.imwrite(file_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        return file_path
    except Exception as e:
        raise IOError(f"Failed to save image to {file_path}: {str(e)}")