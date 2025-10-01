"""
Script to generate CSV mapping from existing images and masks.

This module provides a simple wrapper around the labeling module's
CSV generation functionality, with additional support for reading
simulation metadata.
"""
import os
import json
from typing import Optional

from .labeling import create_dataset_csv_mapping


def get_simulation_info(json_path: str) -> dict:
    """
    Extract simulation information from the simulation_results.json file.

    Args:
        json_path: Path to the simulation_results.json file.

    Returns:
        Dictionary with simulation information, or empty dict if file not found.
    """
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not read simulation results: {e}")
        return {}


def generate_image_mask_mapping(output_dir: str, csv_filename: str = "train.csv") -> Optional[str]:
    """
    Generate CSV mapping between images and masks in a dataset directory.

    This function uses the labeling module's create_dataset_csv_mapping()
    and optionally reads simulation metadata from simulation_results.json.

    Args:
        output_dir: Directory containing 'images' and 'masks' subdirectories.
        csv_filename: Name of the CSV file to create (default: "train.csv").

    Returns:
        Path to the created CSV file, or None if creation failed.
    """
    # Check if simulation results exist and print info
    results_json = os.path.join(output_dir, 'simulation_results.json')
    if os.path.exists(results_json):
        sim_info = get_simulation_info(results_json)
        num_sims = len(sim_info.get('simulations', []))
        print(f"Found simulation results with {num_sims} simulations")

    # Use the refactored labeling module function
    return create_dataset_csv_mapping(output_dir, csv_filename)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate CSV mapping for image-mask pairs in a dataset directory."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory containing images and masks folders (default: output)"
    )
    parser.add_argument(
        "--csv_name",
        type=str,
        default="train.csv",
        help="Name of the CSV file to create (default: train.csv)"
    )

    args = parser.parse_args()

    csv_path = generate_image_mask_mapping(args.output_dir, args.csv_name)
    if csv_path:
        print(f"Success! CSV file created at: {csv_path}")
    else:
        print("Failed to create CSV mapping")
