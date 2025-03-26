"""
Sensor defects for calcium simulation images.
"""
import numpy as np
import cv2


def add_poisson_noise(image, scaling_factor=1.0):
    """
    Add Poisson noise to simulate photon counting statistical fluctuations.
    
    Args:
        image (numpy.ndarray): Input image (values should be in [0, 1]).
        scaling_factor (float): Scaling factor controlling noise intensity.
    
    Returns:
        numpy.ndarray: Image with Poisson noise.
    """
    # Ensure image is in correct range [0, 1]
    if image.max() > 1.0:
        normalized_image = image / 255.0
    else:
        normalized_image = image.copy()
    
    # Scale values to get reasonable photon counts
    lambda_values = normalized_image * 255.0 * scaling_factor
    
    # Add Poisson noise
    if len(image.shape) == 3:
        # Process each channel separately
        noisy_image = np.zeros_like(lambda_values)
        for i in range(image.shape[2]):
            noisy_image[:, :, i] = np.random.poisson(lambda_values[:, :, i])
    else:
        noisy_image = np.random.poisson(lambda_values)
    
    # Rescale to [0, 1]
    noisy_image = noisy_image / (255.0 * scaling_factor)
    
    # Preserve original image range
    if image.max() > 1.0:
        noisy_image = noisy_image * 255.0
    
    return np.clip(noisy_image, 0, image.max()).astype(image.dtype)


def add_readout_noise(image, pattern=None, strength=0.05):
    """
    Add fixed pattern noise (readout noise).
    
    Args:
        image (numpy.ndarray): Input image.
        pattern (numpy.ndarray, optional): Fixed noise pattern. If None, generates random pattern.
        strength (float): Noise strength as a fraction of image dynamic range.
    
    Returns:
        numpy.ndarray: Image with readout noise.
    """
    # If no pattern provided, generate a random one
    if pattern is None:
        if len(image.shape) == 3:
            pattern = np.random.normal(0, 1, (image.shape[0], image.shape[1], 1))
            pattern = np.repeat(pattern, image.shape[2], axis=2)
        else:
            pattern = np.random.normal(0, 1, image.shape)
    
    # Scale pattern to desired strength
    max_val = image.max()
    scaled_pattern = pattern * max_val * strength
    
    # Add pattern to image
    noisy_image = image + scaled_pattern
    
    # Ensure values stay in valid range
    return np.clip(noisy_image, 0, max_val).astype(image.dtype)


def add_gaussian_noise(image, mean=0, sigma=0.1):
    """
    Add Gaussian noise to the image.
    
    Args:
        image (numpy.ndarray): Input image.
        mean (float): Mean of the Gaussian noise.
        sigma (float): Standard deviation of the Gaussian noise.
    
    Returns:
        numpy.ndarray: Image with Gaussian noise.
    """
    # Generate noise
    max_val = image.max()
    sigma_val = max_val * sigma
    
    if len(image.shape) == 3:
        noise = np.random.normal(mean, sigma_val, image.shape)
    else:
        noise = np.random.normal(mean, sigma_val, image.shape)
    
    # Add noise to image
    noisy_image = image + noise
    
    # Ensure values stay in valid range
    return np.clip(noisy_image, 0, max_val).astype(image.dtype)


def apply_dynamic_range_compression(image, gamma=0.7):
    """
    Apply dynamic range compression to simulate sensor limitations.
    
    Args:
        image (numpy.ndarray): Input image.
        gamma (float): Gamma correction factor. Values < 1 compress highlights.
    
    Returns:
        numpy.ndarray: Image with compressed dynamic range.
    """
    # Normalize to [0, 1] for gamma correction
    if image.max() > 1.0:
        normalized = image / 255.0
    else:
        normalized = image.copy()
    
    # Apply gamma correction
    compressed = np.power(normalized, gamma)
    
    # Return to original range
    if image.max() > 1.0:
        compressed = compressed * 255.0
    
    return compressed.astype(image.dtype)


def apply_quantization(image, bits=8):
    """
    Apply quantization to simulate bit depth limitations.
    
    Args:
        image (numpy.ndarray): Input image.
        bits (int): Target bit depth.
    
    Returns:
        numpy.ndarray: Quantized image.
    """
    max_val = image.max()
    
    # Calculate steps based on bit depth
    levels = 2 ** bits
    step = max_val / levels
    
    # Quantize image
    quantized = np.round(image / step) * step
    
    return np.clip(quantized, 0, max_val).astype(image.dtype)


def apply_sensor_defects(image, config):
    """
    Apply all sensor defects based on configuration.
    
    Args:
        image (numpy.ndarray): Input image.
        config (dict): Defect configuration parameters.
    
    Returns:
        numpy.ndarray: Image with sensor defects applied.
    """
    result = image.copy()
    
    # Apply Poisson noise (intensity-dependent)
    if config.get('poisson_noise', False):
        scaling = config.get('poisson_scaling', 1.0)
        result = add_poisson_noise(result, scaling)
    
    # Apply readout noise
    if config.get('readout_noise', False):
        strength = config.get('readout_strength', 0.05)
        pattern = config.get('readout_pattern', None)
        result = add_readout_noise(result, pattern, strength)
    
    # Apply Gaussian noise
    if config.get('gaussian_noise', False):
        mean = config.get('gaussian_mean', 0)
        sigma = config.get('gaussian_sigma', 0.1)
        result = add_gaussian_noise(result, mean, sigma)
    
    # Apply dynamic range compression
    if config.get('dynamic_range_compression', False):
        gamma = config.get('gamma', 0.7)
        result = apply_dynamic_range_compression(result, gamma)
    
    # Apply quantization
    if config.get('quantization', False):
        bits = config.get('bit_depth', 8)
        result = apply_quantization(result, bits)
    
    return result