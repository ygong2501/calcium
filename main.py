"""
Main entry point for calcium simulation system.
"""
import os
import sys
import argparse
import random
import json
import time
import gc
import multiprocessing
import datetime

import numpy as np
import cv2
import psutil

from utils.dataset import generate_stats
from utils.labeling import generate_labels, save_label
from utils.image_processing import apply_all_defects, save_image
from core.parameters import SimulationParameters
from core.pouch import Pouch

# Add parent directory to path (only needed when running as script)
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
                              progress_callback=None,
                              memory_threshold=70, create_stats=True, 
                              edge_blur=False, blur_kernel_size=3, blur_type='mean',
                              generate_masks=True, generate_labels=False, image_size="512x512", jpeg_quality=90,
                              save_pouch=False):
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
        memory_threshold (int): Memory usage percentage threshold for forced garbage collection.
        create_stats (bool): Whether to create statistics about the generated images.
        edge_blur (bool): Whether to apply convolution blur to cell edges.
        blur_kernel_size (int): Size of the convolution kernel for edge blur.
        blur_type (str): Type of convolution blur ('mean' or 'motion').
        generate_masks (bool): Whether to generate individual masks for each cell.
    
    Returns:
        dict: Simulation results info.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Use available CPU cores for any parallel processing
    max_cores = max(1, multiprocessing.cpu_count() - 1)  # Leave one core free

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

    # Use the output directory directly instead of creating batch subdirectories
    batch_dir = output_dir
    os.makedirs(batch_dir, exist_ok=True)

    # Generate a unique run identifier for this batch
    batch_run_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Create common directories for all simulations
    common_img_dir = os.path.join(batch_dir, 'images')
    common_mask_dir = os.path.join(batch_dir, 'masks')
    os.makedirs(common_img_dir, exist_ok=True)
    os.makedirs(common_mask_dir, exist_ok=True)
    
    # Only create labels directory if needed
    common_label_dir = None
    if generate_labels:
        common_label_dir = os.path.join(batch_dir, 'labels')
        os.makedirs(common_label_dir, exist_ok=True)
    
    # Results to return
    results = {
        'simulations': [],
        'output_dir': batch_dir,
        'num_simulations': num_simulations,
        'batch_index': 0  # No longer using batch indices
    }

    # Will collect basic statistics at the end if requested

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
            # Parse image_size string to tuple if it's a string
            output_size = image_size
            if isinstance(image_size, str):
                try:
                    width, height = map(int, image_size.split('x'))
                    output_size = (width, height)
                    print(f"Using custom image size: {width}x{height}")
                except (ValueError, AttributeError):
                    # Default to 512x512 if parsing fails
                    output_size = (512, 512)
                    print(f"Failed to parse image size '{image_size}', using default 512x512")
            
            # Pass jpeg_quality to Pouch constructor
            pouch = Pouch(
                params=sim_params,
                size=pouch_size,
                sim_number=sim_idx,
                save=True,
                save_name=sim_name,
                output_size=output_size,
                jpeg_quality=jpeg_quality
            )

            # Run simulation
            print(f"Running simulation {sim_idx+1}/{num_simulations}: {sim_type} on {pouch_size} pouch")
            pouch.simulate()

            # Use common directories for all simulations
            img_dir = common_img_dir
            label_dir = common_label_dir
            mask_dir = common_mask_dir

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

                        # Use consistent naming pattern for images and masks
                        # Format: {sim_name}_t{time_step}_{batch_run_id}.jpg
                        batch_id_suffix = f"_{batch_run_id}"
                        img_filename = f"{sim_name}_t{time_step:05d}{batch_id_suffix}.jpg"
                        img_path = save_image(processed_image, img_dir, img_filename, format='jpg', quality=jpeg_quality, target_size=output_size)
                        image_files.append(img_path)
                        
                        # Generate and save labels only if requested
                        if generate_labels and common_label_dir is not None:
                            label_data = generate_labels(pouch, time_step)
                            label_data['defect_config'] = defect_config
                            label_filename = f"{sim_name}_t{time_step:05d}.json"
                            label_path = save_label(label_data, common_label_dir, label_filename)
                            label_files.append(label_path)

                        # Generate and save only the combined mask if requested
                        if generate_masks:
                            # Get active cells for the current time step
                            active_cells = pouch.get_active_cells(time_step)
                            
                            # Create a combined mask for all active cells (or a blank mask if no active cells)
                            # Initialize a blank mask with the same size as the output image
                            width, height = output_size
                            combined_mask = np.zeros((height, width), dtype=np.uint8)
                            
                            # Only generate mask if there are active cells
                            if active_cells:
                                # Get masks with only active cells
                                multi_instance_mask = pouch.get_cell_masks(active_only=True, time_step=time_step)
                                cell_ids = np.unique(multi_instance_mask)
                                
                                for cell_id in cell_ids:
                                    if cell_id == 0:
                                        continue  # Skip background
                                    # Create a binary mask for the current cell instance
                                    instance_mask = (multi_instance_mask == cell_id).astype(np.uint8)
                                    # Multiply by 255 to prepare the mask for saving
                                    instance_mask = instance_mask * 255
                                    
                                    # Add this mask to the combined mask using bitwise OR
                                    combined_mask = cv2.bitwise_or(combined_mask, instance_mask)
                                    
                                    # Explicitly delete temporary objects to free memory
                                    del instance_mask
                            
                            # Save the combined mask with the same base filename as the image (including batch run ID)
                            # Format: {sim_name}_t{time_step}_{batch_run_id}_mask_combined.jpg
                            combined_mask_filename = f"{sim_name}_t{time_step:05d}{batch_id_suffix}_mask_combined.jpg"
                            combined_mask_path = save_image(combined_mask, mask_dir, combined_mask_filename, format='jpg', quality=jpeg_quality, target_size=output_size)
                            mask_files.append(combined_mask_path)
                            
                            # Free memory
                            del combined_mask

                        # Explicitly delete temporary objects to free memory
                        del clean_image, processed_image
                        if 'multi_instance_mask' in locals():
                            del multi_instance_mask
                        
                    except Exception as e:
                        print(f"Error processing time step {time_step} for simulation {sim_idx}: {str(e)}")
                        # Add more detailed error information
                        import traceback
                        print(f"Detailed error information:")
                        traceback.print_exc()
                        
                        # Force garbage collection after an error
                        gc.collect()
                        continue  # Skip this time step but continue with others
                
                # More aggressive memory cleanup after each batch
                gc.collect()
                # Give the system a moment to actually release memory
                time.sleep(0.2)

            # Add simulation info to results
            sim_info = {
                'simulation_id': sim_idx,
                'simulation_name': sim_name,
                'simulation_type': sim_type,
                'pouch_size': pouch_size,
                'parameters': sim_params,
                'image_count': len(image_files),
                'image_dir': common_img_dir,
                'mask_dir': common_mask_dir,
                'pouch': pouch if save_pouch else None
            }
            
            # Add label directory only if labels were generated
            if generate_labels and common_label_dir is not None:
                sim_info['label_dir'] = common_label_dir
                sim_info['label_count'] = len(label_files)
            
            results['simulations'].append(sim_info)

            # No dataset creation - just continue with next simulation

            # Explicitly delete the pouch object to free memory if not saving
            if not save_pouch:
                del pouch
                # Force a full garbage collection
                for _ in range(3):  # Multiple GC passes
                    gc.collect()
                    time.sleep(0.5)  # Give system time to release memory

        except Exception as e:
            print(f"Error in simulation {sim_idx}: {str(e)}")
            continue  # Skip this simulation but continue with others

    # After all simulations, create/update the CSV mapping file using the new generator
    from utils.generate_csv import generate_image_mask_mapping
    try:
        # Generate the CSV mapping
        csv_path = generate_image_mask_mapping(batch_dir, csv_filename="train.csv")
        if csv_path:
            print(f"Created CSV mapping: {csv_path}")
            results['image_mask_csv'] = csv_path
            
            # Rename to train.csv if it's a different name
            if os.path.basename(csv_path) != "train.csv":
                try:
                    standard_csv_path = os.path.join(batch_dir, "train.csv")
                    # Try to ensure we can overwrite
                    if os.path.exists(standard_csv_path):
                        os.chmod(standard_csv_path, 0o666)
                    # Copy instead of move to avoid permission issues
                    import shutil
                    shutil.copy2(csv_path, standard_csv_path)
                    print(f"Copied to standard name: {standard_csv_path}")
                    results['image_mask_csv'] = standard_csv_path
                except Exception as rename_error:
                    print(f"Warning: Could not rename CSV file: {rename_error}")
        else:
            print("Failed to create CSV mapping")
            results['csv_error'] = "Failed to create CSV mapping"
    except Exception as e:
        print(f"Error creating CSV mapping: {str(e)}")
        results['csv_error'] = str(e)

    # Generate basic statistics
    if create_stats:
        stats = generate_stats(batch_dir)
        results['stats'] = stats

    # Save results to file
    try:
        # Import NumpyEncoder from utils.labeling for consistent JSON handling
        from utils.labeling import NumpyEncoder
        with open(os.path.join(batch_dir, 'simulation_results.json'), 'w') as f:
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
    parser.add_argument('--output', type=str, default='output',
                        help='Output directory for simulation results (default: output)')
    parser.add_argument('--num_simulations', type=int, default=5,
                        help='Number of simulations to generate')
    parser.add_argument('--pouch_sizes', type=str, nargs='+', choices=['xsmall', 'small', 'medium', 'large'],
                        help='Pouch sizes to use (default: all sizes)')
    parser.add_argument('--sim_types', type=str, nargs='+',
                        choices=["Single cell spikes", "Intercellular transients",
                                 "Intercellular waves", "Fluttering"],
                        help='Simulation types to use (default: all types)')
    # Removed unused num_threads argument
    parser.add_argument('--gui', action='store_true',
                        help='Launch GUI interface')
    parser.add_argument('--generate-labels', action='store_true',
                        help='Generate JSON label files (default: False)')
    parser.add_argument('--max-time-steps', type=int, default=None,
                        help='Maximum number of time steps to process (default: all)')
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
        # Prepare time steps if max_time_steps is specified
        time_steps = None
        if args.max_time_steps is not None:
            # Generate a subset of time steps up to the max
            time_steps = list(range(0, min(18000, args.max_time_steps), 200))
            print(f"Limited to {len(time_steps)} time steps (max {args.max_time_steps})")
            
        results = generate_simulation_batch(
            num_simulations=args.num_simulations,
            output_dir=args.output,
            pouch_sizes=args.pouch_sizes,
            sim_types=args.sim_types,
            time_steps=time_steps,
            progress_callback=cli_progress_callback,
            generate_labels=args.generate_labels
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
