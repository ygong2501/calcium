"""
Script to generate CSV mapping from existing images and masks.
"""
import os
import csv
import re
import glob
import json
from pathlib import Path


def get_simulation_info(json_path):
    """
    Extract simulation information from the simulation_results.json file.
    
    Args:
        json_path (str): Path to the simulation_results.json file.
        
    Returns:
        dict: Dictionary with simulation information.
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading simulation results: {e}")
        return {}


def generate_image_mask_mapping(output_dir, csv_filename="train.csv"):
    """
    Generate CSV mapping between images and masks based on simulation_results.json.
    
    Args:
        output_dir (str): Directory containing images and masks folders.
        csv_filename (str): Name of the CSV file to create.
        
    Returns:
        str: Path to the created CSV file.
    """
    # Paths to directories
    img_dir = os.path.join(output_dir, 'images')
    mask_dir = os.path.join(output_dir, 'masks')
    results_json = os.path.join(output_dir, 'simulation_results.json')
    
    # Verify directories exist
    if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
        print(f"Error: Images or masks directory not found in {output_dir}")
        return None
    
    # Get simulation information
    sim_info = get_simulation_info(results_json)
    simulations = sim_info.get('simulations', [])
    
    # Create list of image-mask pairs based on simulation details
    image_mask_pairs = []
    
    # Get count of simulation and time steps from simulation_results.json
    for sim in simulations:
        sim_name = sim.get('simulation_name')
        img_count = sim.get('image_count', 0)
        
        if not sim_name or img_count == 0:
            continue
        
        # Get all time steps based on image count
        # For each simulation, time steps typically range from 0 to img_count-1
        # Simulations normally create images with the pattern: {sim_name}_t{time_step}.jpg
        # and masks with the pattern: {sim_name}_t{time_step}_{batch_id}_mask_combined.jpg
        
        # Instead of guessing time steps, directly find all files for this simulation
        # Pattern for looking up existing image files
        img_pattern = f"{sim_name}_t*.jpg"
        mask_pattern = f"{sim_name}_t*_mask_combined.jpg"
        
        # Find all image and mask files
        all_img_files = glob.glob(os.path.join(img_dir, img_pattern))
        all_mask_files = glob.glob(os.path.join(mask_dir, mask_pattern))
        
        print(f"Found {len(all_img_files)} images and {len(all_mask_files)} masks for {sim_name}")
        
        # Process all found images
        for img_path in all_img_files:
            img_filename = os.path.basename(img_path)
            
            # Extract the time step from the image filename
            # Pattern: {sim_name}_t{time_step}_{batch_id}.jpg
            match = re.match(f"{sim_name}_t(\\d{{5}})_(\\d+)\\.jpg", img_filename)
            if match:
                time_step = match.group(1)
                batch_id = match.group(2)
                
                # Corresponding mask pattern
                mask_filename = f"{sim_name}_t{time_step}_{batch_id}_mask_combined.jpg"
                mask_path = os.path.join(mask_dir, mask_filename)
                
                # Check if mask exists
                if os.path.exists(mask_path):
                    # Add to pairs list
                    image_mask_pairs.append((img_filename, mask_filename))
                    # Print for debugging
                    if len(image_mask_pairs) % 100 == 0:
                        print(f"Processed {len(image_mask_pairs)} pairs")
            else:
                # Try alternative pattern without batch ID
                match = re.match(f"{sim_name}_t(\\d{{5}})\\.jpg", img_filename)
                if match:
                    time_step = match.group(1)
                    
                    # Look for any mask with this time step
                    mask_pattern = f"{sim_name}_t{time_step}_*_mask_combined.jpg"
                    matching_masks = glob.glob(os.path.join(mask_dir, mask_pattern))
                    
                    if matching_masks:
                        mask_filename = os.path.basename(matching_masks[0])
                        image_mask_pairs.append((img_filename, mask_filename))
    
    # Write the CSV file
    csv_path = os.path.join(output_dir, "new_" + csv_filename)
    try:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(["ImageId", "MaskId"])
            
            # Write data rows (sorted)
            for img, mask in sorted(image_mask_pairs):
                writer.writerow([img, mask])
    except PermissionError:
        print(f"Permission denied when writing to {csv_path}")
        # Try in a different location
        alternative_path = os.path.join(os.path.dirname(output_dir), "new_" + csv_filename)
        print(f"Trying alternative location: {alternative_path}")
        with open(alternative_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(["ImageId", "MaskId"])
            
            # Write data rows (sorted)
            for img, mask in sorted(image_mask_pairs):
                writer.writerow([img, mask])
        csv_path = alternative_path
    
    # Report statistics
    print(f"CSV mapping created with {len(image_mask_pairs)} image-mask pairs")
    print(f"CSV file saved to: {csv_path}")
    
    return csv_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate CSV mapping for image-mask pairs.")
    parser.add_argument("--output_dir", type=str, default="output", 
                        help="Directory containing images and masks folders (default: output)")
    parser.add_argument("--csv_name", type=str, default="train.csv",
                        help="Name of the CSV file to create (default: train.csv)")
    
    args = parser.parse_args()
    
    generate_image_mask_mapping(args.output_dir, args.csv_name)