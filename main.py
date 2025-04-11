"""
Main entry point for calcium simulation system.
"""
from utils.dataset import DatasetManager
from utils.labeling import generate_labels, save_label
from utils.image_processing import apply_all_defects, save_image
from core.geometry_loader import GeometryLoader
from core.parameters import SimulationParameters
from core.pouch import Pouch
import os
import sys
import argparse
import random
import json
import numpy as np
import time
import multiprocessing
from pathlib import Path
import gc
import re
import psutil
import tkinter as tk

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_random_defect_config():
    """
    Generate a random defect configuration.

    Returns:
        dict: Random defect configuration.
    """
    config = {
        # Background defects - keep enabled
        'background_fluorescence': random.choice([True, False]),
        'background_intensity': random.uniform(0.05, 0.2),
        'non_uniform_background': random.choice([True, False]),

        'spontaneous_luminescence': random.choice([True, False]),
        'spontaneous_min': random.uniform(0.03, 0.1),
        'spontaneous_max': random.uniform(0.1, 0.2),
        'spontaneous_probability': random.uniform(0.1, 0.3),

        'cell_fragments': random.choice([True, False]),
        'fragment_count': random.randint(3, 10),
        'fragment_min_size': random.randint(5, 15),
        'fragment_max_size': random.randint(15, 40),
        'fragment_min_intensity': random.uniform(0.05, 0.15),
        'fragment_max_intensity': random.uniform(0.15, 0.3),

        # Optical defects - disabled by default

        'chromatic_aberration': False,
        'chromatic_offset': random.randint(1, 5),

        'vignetting': False,
        'vignetting_strength': random.uniform(0.2, 0.7),

        # Sensor defects - disabled by default
        'poisson_noise': False,
        'poisson_scaling': random.uniform(0.5, 2.0),

        'readout_noise': False,
        'readout_strength': random.uniform(0.02, 0.1),

        'gaussian_noise': False,
        'gaussian_mean': 0,
        'gaussian_sigma': random.uniform(0.05, 0.15),

        'dynamic_range_compression': False,
        'gamma': random.uniform(0.6, 0.9),

        'quantization': False,
        'bit_depth': random.choice([6, 8, 10]),

        # Post-processing - disabled by default
        'defocus_blur': False,
        'defocus_kernel_size': random.choice([5, 7, 9, 11]),
        'defocus_sigma': random.uniform(1.0, 5.0),

        'partial_defocus': False,
        'defocus_radius': random.randint(50, 200),

        'adjust_brightness_contrast': False,
        'brightness': random.uniform(-0.1, 0.1),
        'contrast': random.uniform(0.8, 1.2)
    }

    return config


def get_next_batch_index(output_dir):
    """
    Get the next available batch index for simulation results.
    
    Args:
        output_dir (str): Output directory to scan for existing batches.
        
    Returns:
        int: Next available batch index.
    """
    # Check for directories matching simulation_batch_X pattern
    pattern = re.compile(r'simulation_batch_(\d+)')
    max_index = -1
    
    # Look for existing batch directories
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path):
            match = pattern.match(item)
            if match:
                try:
                    index = int(match.group(1))
                    max_index = max(max_index, index)
                except ValueError:
                    continue
    
    # Return next available index
    return max_index + 1


def monitor_memory():
    """
    Monitor system memory usage and return current usage percentage.
    
    Returns:
        float: Memory usage percentage (0-100).
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_usage_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
    
    # Get system memory info
    system_memory = psutil.virtual_memory()
    system_memory_percent = system_memory.percent
    
    return system_memory_percent, memory_usage_mb


def generate_simulation_batch(num_simulations, output_dir, pouch_sizes=None,
                              sim_types=None, time_steps=None, defect_configs=None,
                              progress_callback=None, num_threads=None,
                              memory_threshold=85, create_dataset=True, 
                              dataset_split_ratios=(0.7, 0.15, 0.15),
                              edge_blur=False, blur_kernel_size=3, blur_type='mean'):
    """
    Batch generate simulation images, labels, and individual binary masks for each cell.
    Includes memory management, sequential batch processing, and file sequence management.
    
    Args:
        num_simulations (int): Number of simulations to generate.
        output_dir (str): Output directory.
        pouch_sizes (list, optional): List of pouch sizes to use.
        sim_types (list, optional): List of simulation types.
        time_steps (list, optional): List of time steps to generate.
        defect_configs (list, optional): List of defect configurations.
        progress_callback (callable, optional): Callback function to report progress.
            Function signature: callback(current_sim, total_sims)
        num_threads (int, optional): Number of threads to use for parallel processing.
            If None, uses available CPU cores.
        memory_threshold (int): Memory usage percentage threshold for forced garbage collection.
        create_dataset (bool): Whether to create a dataset from the generated images.
        dataset_split_ratios (tuple): Train/validation/test split ratios for dataset creation.
        edge_blur (bool): Whether to apply convolution blur to cell edges.
        blur_kernel_size (int): Size of the convolution kernel for edge blur.
        blur_type (str): Type of convolution blur ('mean' or 'motion').
    
    Returns:
        dict: Simulation results info.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set number of threads for parallel processing
    if num_threads is None:
        num_threads = max(1, multiprocessing.cpu_count() - 1)  # Leave one core free

    # Default values
    if pouch_sizes is None:
        pouch_sizes = ['xsmall', 'small', 'medium', 'large']

    if sim_types is None:
        sim_types = [
            "Single cell spikes",
            "Intercellular transients",
            "Intercellular waves",
            "Fluttering"
        ]

    if time_steps is None:
        # Select time steps around 5 frames per second (1 frame every 0.2 seconds)
        # For a 1-hour simulation (3600 seconds), that's 18000 frames
        # Select a subset of frames for efficiency
        time_steps = list(range(0, 18000, 200))  # Every 40 seconds

    # Get the next batch index for folder naming
    batch_idx = get_next_batch_index(output_dir)
    batch_name = f"simulation_batch_{batch_idx}"
    batch_dir = os.path.join(output_dir, batch_name)
    os.makedirs(batch_dir, exist_ok=True)
    
    # Results to return
    results = {
        'simulations': [],
        'output_dir': batch_dir,
        'num_simulations': num_simulations,
        'batch_index': batch_idx
    }

    # Create a dataset manager if requested
    dataset_manager = None
    dataset_info = None
    if create_dataset:
        dataset_manager = DatasetManager(batch_dir)
        dataset_info = dataset_manager.create_dataset(
            name=f"dataset_{batch_idx}",
            split_ratios=dataset_split_ratios
        )

    # Process each simulation
    for sim_idx in range(num_simulations):
        # Monitor memory and force garbage collection if needed
        memory_percent, memory_mb = monitor_memory()
        if memory_percent > memory_threshold:
            print(f"Memory usage high ({memory_percent:.1f}%, {memory_mb:.1f}MB), performing garbage collection...")
            gc.collect()
            # Give the system a moment to actually free memory
            time.sleep(0.5)
        
        # Update progress if callback provided
        if progress_callback:
            progress_callback(sim_idx + 1, num_simulations)

        # Choose random parameters for this simulation
        pouch_size = random.choice(pouch_sizes)
        sim_type = random.choice(sim_types)

        try:
            # Create parameter set
            params = SimulationParameters(sim_type=sim_type)

            # Generate random parameters
            sim_params = params.generate_random_params(seed=sim_idx)

            # Create the simulation
            sim_name = f"{sim_type.replace(' ', '_')}_{sim_idx}"
            pouch = Pouch(
                params=sim_params,
                size=pouch_size,
                sim_number=sim_idx,
                save=True,
                save_name=sim_name,
                output_size=(512, 512)
            )

            # Run simulation
            print(f"Running simulation {sim_idx+1}/{num_simulations}: {sim_type} on {pouch_size} pouch")
            pouch.simulate()

            # Create directories for this simulation
            sim_dir = os.path.join(batch_dir, sim_name)
            img_dir = os.path.join(sim_dir, 'images')
            label_dir = os.path.join(sim_dir, 'labels')
            mask_dir = os.path.join(sim_dir, 'masks')
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(label_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)

            # Generate images, labels, and masks for each time step
            image_files = []
            label_files = []
            mask_files = []

            # Process time steps in smaller batches to control memory usage
            time_step_batches = [time_steps[i:i+10] for i in range(0, len(time_steps), 10)]
            
            for time_step_batch in time_step_batches:
                for time_step in time_step_batch:
                    if time_step >= pouch.T:
                        continue

                    # Get defect config (random if not provided)
                    if defect_configs is not None and sim_idx < len(defect_configs):
                        defect_config = defect_configs[sim_idx]
                    else:
                        defect_config = generate_random_defect_config()

                    try:
                        # Generate clean image with edge blur if requested
                        clean_image = pouch.generate_image(
                            time_step, 
                            with_border=False,
                            edge_blur=edge_blur,
                            blur_kernel_size=blur_kernel_size,
                            blur_type=blur_type
                        )

                        # Apply defects (using pouch.get_cell_masks() for defect computations)
                        processed_image = apply_all_defects(
                            clean_image,
                            pouch.get_cell_masks(),
                            defect_config
                        )

                        # Generate labels
                        label_data = generate_labels(pouch, time_step)
                        label_data['defect_config'] = defect_config

                        # Save image and label
                        img_filename = f"{sim_name}_t{time_step:05d}.png"
                        label_filename = f"{sim_name}_t{time_step:05d}.json"
                        img_path = save_image(processed_image, img_dir, img_filename)
                        label_path = save_label(label_data, label_dir, label_filename)
                        image_files.append(img_path)
                        label_files.append(label_path)

                        # Generate and save individual binary masks for each cell
                        multi_instance_mask = pouch.get_cell_masks()
                        cell_ids = np.unique(multi_instance_mask)
                        
                        for cell_id in cell_ids:
                            if cell_id == 0:
                                continue  # Skip background
                            # Create a binary mask for the current cell instance
                            instance_mask = (multi_instance_mask == cell_id).astype(np.uint8)
                            # Multiply by 255 to prepare the mask for saving as PNG
                            instance_mask = instance_mask * 255

                            # Generate a file name for this instance mask (using cell_id)
                            mask_filename = f"{sim_name}_t{time_step:05d}_mask_{cell_id:03d}.png"
                            mask_path = save_image(instance_mask, mask_dir, mask_filename)
                            mask_files.append(mask_path)
                            
                            # Explicitly delete temporary objects to free memory
                            del instance_mask

                        # Explicitly delete temporary objects to free memory
                        del clean_image, processed_image, multi_instance_mask
                        
                    except Exception as e:
                        print(f"Error processing time step {time_step} for simulation {sim_idx}: {str(e)}")
                        continue  # Skip this time step but continue with others
                
                # Periodic garbage collection after each batch of time steps
                gc.collect()

            # Add simulation info to results
            sim_info = {
                'simulation_id': sim_idx,
                'simulation_name': sim_name,
                'simulation_type': sim_type,
                'pouch_size': pouch_size,
                'parameters': sim_params,
                'image_count': len(image_files),
                'image_dir': img_dir,
                'label_dir': label_dir,
                'mask_dir': mask_dir
            }
            results['simulations'].append(sim_info)

            # Add to dataset if enabled
            if create_dataset and dataset_manager is not None:
                dataset_manager.add_simulation_to_dataset(
                    image_files, label_files, sim_info, split='random'
                )

            # Explicitly delete the pouch object to free memory
            del pouch
            gc.collect()

        except Exception as e:
            print(f"Error in simulation {sim_idx}: {str(e)}")
            continue  # Skip this simulation but continue with others

    # Generate dataset statistics if enabled
    if create_dataset and dataset_manager is not None:
        dataset_stats = dataset_manager.generate_dataset_stats()
        results['dataset_stats'] = dataset_stats

    # Save results to file
    try:
        with open(os.path.join(batch_dir, 'simulation_results.json'), 'w') as f:
            # Use custom converter for numpy types
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return super(NumpyEncoder, self).default(obj)
            json.dump(results, f, indent=4, cls=NumpyEncoder)
    except Exception as e:
        print(f"Error saving results: {str(e)}")

    # Final garbage collection
    gc.collect()
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Calcium Ion Dynamic Simulation System")

    # Add arguments
    parser.add_argument('--output', type=str, default='./output',
                        help='Output directory for simulation results')
    parser.add_argument('--num_simulations', type=int, default=5,
                        help='Number of simulations to generate')
    parser.add_argument('--pouch_sizes', type=str, nargs='+', choices=['xsmall', 'small', 'medium', 'large'],
                        help='Pouch sizes to use (default: all sizes)')
    parser.add_argument('--sim_types', type=str, nargs='+',
                        choices=["Single cell spikes", "Intercellular transients",
                                 "Intercellular waves", "Fluttering"],
                        help='Simulation types to use (default: all types)')
    parser.add_argument('--num_threads', type=int,
                        help='Number of threads to use for parallel processing')
    parser.add_argument('--gui', action='store_true',
                        help='Launch GUI interface')
    parser.add_argument('--version', action='version', version='Calcium Simulation v1.0',
                        help='Show program version and exit')

    args = parser.parse_args()

    if args.gui:
        try:
            # Import GUI modules
            from gui.main_window import launch_gui
            if not launch_gui():
                print("Failed to launch GUI. Using command line mode instead.")
                # Fall through to the CLI mode
            else:
                return
        except ImportError as e:
            print(f"Error importing GUI modules: {str(e)}")
            print("Falling back to command line mode.")

    # Progress callback for CLI
    def cli_progress_callback(current, total):
        percent = int(current / total * 100)
        bar_length = 40
        filled_length = int(bar_length * current / total)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        print(f"\r[{bar}] {percent}% - Simulation {current}/{total}", end='')
        if current == total:
            print()  # Print newline on completion

    # Run batch simulation
    print(
        f"Starting batch simulation with {args.num_simulations} simulations...")
    start_time = time.time()

    try:
        results = generate_simulation_batch(
            num_simulations=args.num_simulations,
            output_dir=args.output,
            pouch_sizes=args.pouch_sizes,
            sim_types=args.sim_types,
            num_threads=args.num_threads,
            progress_callback=cli_progress_callback
        )

        # Print statistics
        end_time = time.time()
        duration = end_time - start_time

        print(
            f"\nGenerated {args.num_simulations} simulations in {duration:.2f} seconds")
        print(
            f"Average time per simulation: {duration/args.num_simulations:.2f} seconds")

        # Get dataset statistics
        stats = results.get('dataset_stats', {})
        train_count = stats.get('num_images', {}).get('train', 0)
        val_count = stats.get('num_images', {}).get('val', 0)
        test_count = stats.get('num_images', {}).get('test', 0)
        total_images = train_count + val_count + test_count

        print(f"Generated {total_images} total images")
        print(
            f"Train: {train_count}, Validation: {val_count}, Test: {test_count}")
        print(f"Results saved to {os.path.abspath(args.output)}")

    except Exception as e:
        print(f"\nError during batch simulation: {str(e)}")
        print("Please check the error message and try again.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
