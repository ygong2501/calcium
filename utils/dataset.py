"""Utility functions for basic statistics."""
import os
import glob

def generate_stats(output_dir):
    """
    Generate basic statistics about images and masks.
    
    Args:
        output_dir (str): The directory to analyze.
        
    Returns:
        dict: Statistics about images and masks.
    """
    # Find all simulation directories
    sim_dirs = [d for d in os.listdir(output_dir) 
                if os.path.isdir(os.path.join(output_dir, d)) and not d.startswith('.')]
    
    # Count total images and masks
    total_images = 0
    total_masks = 0
    
    for sim_dir in sim_dirs:
        sim_path = os.path.join(output_dir, sim_dir)
        img_dir = os.path.join(sim_path, 'images')
        mask_dir = os.path.join(sim_path, 'masks')
        
        if os.path.exists(img_dir):
            total_images += len(glob.glob(os.path.join(img_dir, '*.jpg'))) + len(glob.glob(os.path.join(img_dir, '*.png')))
        
        if os.path.exists(mask_dir):
            total_masks += len(glob.glob(os.path.join(mask_dir, '*.jpg'))) + len(glob.glob(os.path.join(mask_dir, '*.png')))
    
    stats = {
        'total_simulations': len(sim_dirs),
        'total_images': total_images,
        'total_masks': total_masks,
    }
    
    return stats