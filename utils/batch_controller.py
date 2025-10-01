"""
Batch generation controller for managing large-scale simulation workflows.

This module handles the orchestration of batch simulation generation,
including multi-batch processing, progress tracking, CSV creation,
and video generation. Separated from GUI for reusability and testability.
"""
import os
import gc
import time
import json
from typing import Dict, List, Optional, Callable, Any

import psutil

from main import generate_simulation_batch, monitor_memory
from utils.labeling import create_dataset_csv_mapping


class BatchGenerationController:
    """
    Controls batch generation workflow independent of GUI.

    This class manages the complete batch generation process including:
    - Single or multi-batch simulation generation
    - Progress tracking and callbacks
    - Memory management and cleanup
    - CSV mapping file creation
    - Video generation from simulations

    Attributes:
        cancel_requested (bool): Flag to signal cancellation of batch process.
    """

    def __init__(self):
        """Initialize the batch generation controller."""
        self.cancel_requested = False

    def run_batch_generation(
        self,
        batch_params: Dict[str, Any],
        output_dir: str,
        progress_callback: Optional[Callable[[int, int, Optional[Dict]], bool]] = None,
        status_callback: Optional[Callable[[str], None]] = None
    ) -> Dict[str, Any]:
        """
        Execute batch generation process.

        Args:
            batch_params: Dictionary containing batch configuration:
                - enable_multi_batch (bool): Whether to generate multiple batches
                - num_batches (int): Number of batches (if multi-batch mode)
                - sims_per_batch (int): Simulations per batch (if multi-batch)
                - num_simulations (int): Total simulations (if single batch)
                - pouch_sizes (List[str]): Pouch sizes to generate
                - sim_types (List[str]): Simulation types to generate
                - time_steps (List[int]): Time steps to capture
                - image_size (str): Image dimensions (e.g., "512x512")
                - jpeg_quality (int): JPEG quality setting
                - edge_blur (bool): Whether to apply edge blur
                - blur_kernel_size (int): Blur kernel size
                - blur_type (str): Type of blur
                - generate_masks (bool): Whether to generate mask files
                - create_csv (bool): Whether to create CSV mapping
                - enable_video (bool): Whether to generate video

            output_dir: Directory to save generated files
            progress_callback: Optional callback(current, total, info) -> bool
                Returns True to cancel process
            status_callback: Optional callback(status_message) to update status

        Returns:
            Dictionary containing:
            - simulations (List[Dict]): Information about generated simulations
            - output_dir (str): Output directory path
            - csv_path (str, optional): Path to CSV mapping file
            - success (bool): Whether generation completed successfully
            - error (str, optional): Error message if failed

        Raises:
            ValueError: If batch_params are invalid
            RuntimeError: If batch generation fails
        """
        self.cancel_requested = False

        try:
            # Update status
            if status_callback:
                status_callback("Initializing batch generation...")

            # Parse and validate parameters
            config = self._parse_batch_config(batch_params)

            # Execute batch generation based on mode
            if config['enable_multi_batch']:
                results = self._run_multi_batch_generation(
                    config, output_dir, progress_callback, status_callback
                )
            else:
                results = self._run_single_batch_generation(
                    config, output_dir, progress_callback, status_callback
                )

            # Check if cancelled
            if self.cancel_requested:
                if status_callback:
                    status_callback("Batch generation cancelled")
                results['success'] = False
                results['cancelled'] = True
                return results

            # Create CSV mapping if requested
            if config['create_csv'] and results.get('simulations'):
                csv_path = self._create_csv_mapping(
                    results, output_dir, config['generate_masks'], status_callback
                )
                results['csv_path'] = csv_path

            # Generate video if requested
            if config['enable_video'] and results.get('simulations'):
                self._generate_video(
                    results, output_dir, batch_params, status_callback
                )

            # Final status
            if status_callback:
                total_sims = len(results.get('simulations', []))
                total_images = sum(s.get('image_count', 0) for s in results.get('simulations', []))
                status_callback(f"Batch complete: {total_sims} simulations, {total_images} images")

            results['success'] = True
            return results

        except Exception as e:
            error_msg = f"Batch generation error: {str(e)}"
            if status_callback:
                status_callback(error_msg)

            return {
                'success': False,
                'error': error_msg,
                'simulations': [],
                'output_dir': output_dir
            }

    def cancel(self):
        """Request cancellation of current batch generation."""
        self.cancel_requested = True

    def _parse_batch_config(self, batch_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse and validate batch configuration parameters.

        Args:
            batch_params: Raw batch parameters from GUI or API

        Returns:
            Validated configuration dictionary

        Raises:
            ValueError: If parameters are invalid
        """
        config = {}

        # Multi-batch mode
        config['enable_multi_batch'] = batch_params.get('enable_multi_batch', False)

        if config['enable_multi_batch']:
            config['num_batches'] = int(batch_params.get('num_batches', 1))
            config['sims_per_batch'] = int(batch_params.get('sims_per_batch', 10))
            config['total_simulations'] = config['num_batches'] * config['sims_per_batch']
        else:
            config['num_simulations'] = int(batch_params.get('num_simulations', 5))
            config['total_simulations'] = config['num_simulations']

        # Parse image size
        image_size_str = batch_params.get('image_size', '512x512')
        try:
            width, height = map(int, image_size_str.split('x'))
            config['output_size'] = (width, height)
        except (ValueError, AttributeError):
            config['output_size'] = (512, 512)

        # Other configuration
        config['pouch_sizes'] = batch_params.get('pouch_sizes', ['small'])
        config['sim_types'] = batch_params.get('sim_types', None)
        config['time_steps'] = batch_params.get('time_steps', None)
        config['jpeg_quality'] = batch_params.get('jpeg_quality', 90)
        config['edge_blur'] = batch_params.get('edge_blur', False)
        config['blur_kernel_size'] = int(batch_params.get('blur_kernel_size', 3))
        config['blur_type'] = batch_params.get('blur_type', 'mean')
        config['generate_masks'] = batch_params.get('generate_masks', True)
        config['create_csv'] = batch_params.get('create_csv', False)
        config['enable_video'] = batch_params.get('enable_video', False)
        config['defect_config'] = batch_params.get('defect_config', None)

        # Determine memory threshold based on pouch size
        if 'large' in config['pouch_sizes'] or 'xlarge' in config['pouch_sizes']:
            config['memory_threshold'] = 60  # Conservative for large geometries
        else:
            config['memory_threshold'] = 70  # Standard threshold

        return config

    def _run_single_batch_generation(
        self,
        config: Dict[str, Any],
        output_dir: str,
        progress_callback: Optional[Callable],
        status_callback: Optional[Callable]
    ) -> Dict[str, Any]:
        """Execute single batch generation."""
        if status_callback:
            status_callback(f"Generating {config['num_simulations']} simulations...")

        # Prepare defect configs
        defect_config = config.get('defect_config')
        defect_configs = [defect_config] * config['num_simulations'] if defect_config else None

        # Run batch generation
        results = generate_simulation_batch(
            num_simulations=config['num_simulations'],
            output_dir=output_dir,
            pouch_sizes=config['pouch_sizes'],
            sim_types=config['sim_types'],
            time_steps=config['time_steps'],
            defect_configs=defect_configs,
            progress_callback=progress_callback,
            create_stats=True,
            edge_blur=config['edge_blur'],
            blur_kernel_size=config['blur_kernel_size'],
            blur_type=config['blur_type'],
            generate_masks=config['generate_masks'],
            image_size=config['output_size'],
            jpeg_quality=config['jpeg_quality'],
            memory_threshold=config['memory_threshold'],
            save_pouch=config['enable_video']
        )

        return results

    def _run_multi_batch_generation(
        self,
        config: Dict[str, Any],
        output_dir: str,
        progress_callback: Optional[Callable],
        status_callback: Optional[Callable]
    ) -> Dict[str, Any]:
        """Execute multi-batch generation with cleanup between batches."""
        all_results = {'simulations': [], 'output_dir': output_dir}

        for batch_idx in range(config['num_batches']):
            # Check cancellation
            if self.cancel_requested:
                break

            if status_callback:
                status_callback(
                    f"Batch {batch_idx + 1}/{config['num_batches']}: "
                    f"Generating {config['sims_per_batch']} simulations..."
                )

            # Memory cleanup between batches (except first)
            if batch_idx > 0:
                self._force_memory_cleanup(status_callback)

            # Create progress callback for this batch
            def batch_progress_callback(current, total):
                if progress_callback:
                    global_current = current + batch_idx * config['sims_per_batch']
                    global_total = config['total_simulations']
                    info = {'batch_idx': batch_idx, 'batch_total': config['num_batches']}
                    return progress_callback(global_current, global_total, info)
                return False

            # Prepare defect configs for this batch
            defect_config = config.get('defect_config')
            defect_configs = [defect_config] * config['sims_per_batch'] if defect_config else None

            # Run this batch
            batch_results = generate_simulation_batch(
                num_simulations=config['sims_per_batch'],
                output_dir=output_dir,
                pouch_sizes=config['pouch_sizes'],
                sim_types=config['sim_types'],
                time_steps=config['time_steps'],
                defect_configs=defect_configs,
                progress_callback=batch_progress_callback,
                create_stats=True,
                edge_blur=config['edge_blur'],
                blur_kernel_size=config['blur_kernel_size'],
                blur_type=config['blur_type'],
                generate_masks=config['generate_masks'],
                image_size=config['output_size'],
                jpeg_quality=config['jpeg_quality'],
                memory_threshold=config['memory_threshold'],
                save_pouch=False  # Don't save pouch in multi-batch to save memory
            )

            # Accumulate results
            if 'simulations' in batch_results:
                all_results['simulations'].extend(batch_results['simulations'])

            # Aggressive cleanup for large geometries
            if 'large' in config['pouch_sizes'] or 'xlarge' in config['pouch_sizes']:
                self._force_memory_cleanup(status_callback)
                time.sleep(0.5)
                self._force_memory_cleanup(status_callback)

        return all_results

    def _create_csv_mapping(
        self,
        results: Dict[str, Any],
        output_dir: str,
        generate_masks: bool,
        status_callback: Optional[Callable]
    ) -> Optional[str]:
        """Create CSV mapping file for image-mask pairs."""
        if status_callback:
            status_callback("Creating CSV mapping file...")

        try:
            csv_path = create_dataset_csv_mapping(output_dir, filename="train.csv")

            if csv_path and status_callback:
                status_callback(f"CSV mapping created: {os.path.basename(csv_path)}")

            return csv_path

        except Exception as e:
            if status_callback:
                status_callback(f"Warning: CSV creation failed: {str(e)}")
            return None

    def _generate_video(
        self,
        results: Dict[str, Any],
        output_dir: str,
        batch_params: Dict[str, Any],
        status_callback: Optional[Callable]
    ):
        """Generate video from last simulation."""
        if status_callback:
            status_callback("Generating video from last simulation...")

        try:
            # Get last simulation's pouch
            last_sim = results.get('simulations', [])[-1]
            last_pouch = last_sim.get('pouch')

            if not last_pouch:
                if status_callback:
                    status_callback("Warning: No pouch saved for video generation")
                return

            # Configure video options
            video_format = batch_params.get('video_format', 'mp4')
            fps = batch_params.get('video_fps', 10)
            skip_frames = batch_params.get('video_skip_frames', 50)
            quality = batch_params.get('video_quality', 23)

            video_filename = f"simulation_video.{video_format}"
            video_path = os.path.join(output_dir, video_filename)

            # Generate video
            extra_args = ['-codec:v', 'h264', '-crf', str(quality), '-pix_fmt', 'yuv420p']
            last_pouch.make_animation(
                path=output_dir,
                fps=fps,
                skip_frames=skip_frames,
                filename=video_filename,
                extra_args=extra_args
            )

            if status_callback:
                status_callback(f"Video generated: {video_filename}")

        except Exception as e:
            if status_callback:
                status_callback(f"Video generation error: {str(e)}")

    def _force_memory_cleanup(self, status_callback: Optional[Callable] = None):
        """Force aggressive memory cleanup."""
        if status_callback:
            status_callback("Performing memory cleanup...")

        gc.collect()

        # Check if another cleanup is needed
        memory_percent, _ = monitor_memory()
        if memory_percent > 75:
            gc.collect()

        # Clean matplotlib modules if loaded
        import sys
        for name in list(sys.modules.keys()):
            if name.startswith('matplotlib'):
                if name in sys.modules:
                    del sys.modules[name]
