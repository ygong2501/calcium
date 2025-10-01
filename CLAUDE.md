# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Calcium Ion Dynamic Simulation System - A Windows-compatible tool for generating simulated cell calcium ion activity image datasets for neural network training. The system simulates calcium signaling dynamics using PDEs and generates realistic microscopy images with various imaging defects.

## Key Commands

### Running the Application
```bash
# Launch GUI interface
python main.py --gui

# Generate simulations via CLI (5 simulations by default)
python main.py --num_simulations 5 --output output/

# Generate with specific parameters
python main.py --num_simulations 10 --pouch_sizes small medium --sim_types "Intercellular waves" --output results/

# Generate with labels (JSON metadata files)
python main.py --generate-labels --num_simulations 5
```

### Dependencies
```bash
# Install all dependencies
pip install -r requirements.txt

# Note: tkinter is part of Python standard library (required for GUI)
```

### SAM2 Model Fine-tuning
The inference module is designed to work with SAM2 segmentation models:
1. Set up SAM2 from https://github.com/facebookresearch/sam2
2. Copy files from the `inference/` directory to SAM2 root
3. Run under WSL2 (Windows Subsystem for Linux 2):
   ```bash
   python processing.py  # Prepares data
   python inference.py   # Runs training
   ```

## Architecture

### Core Simulation Pipeline

1. **Parameter Definition** ([parameters.py](core/parameters.py))
   - `SimulationParameters` manages all simulation parameters
   - Four simulation types: "Single cell spikes", "Intercellular transients", "Intercellular waves", "Fluttering"
   - Each type has different `lower` and `upper` parameters that control calcium wave behavior

2. **Geometry Loading** ([geometry_loader.py](core/geometry_loader.py))
   - Loads pre-computed cell geometry files from `./calcium_simulation/geometry/`
   - Required files: `disc_vertices.npy`, `disc_sizes_laplacian.npy`, `disc_sizes_adj.npy`
   - Supports four pouch sizes: 'xsmall', 'small', 'medium', 'large'

3. **Calcium Dynamics Simulation** ([pouch.py](core/pouch.py))
   - `Pouch` class implements PDE-based calcium signaling model
   - Simulates 4 state variables per cell: cytosolic Ca²⁺, IP₃, ER Ca²⁺, IP₃ receptor inactivation
   - Time step: 0.2 seconds, default simulation: 1 hour (18,000 time steps)
   - Generates 512x512 images by default (configurable via `output_size` parameter)

4. **Defect Application** ([image_processing.py](utils/image_processing.py))
   - Applies realistic imaging artifacts to clean simulation images
   - Three artifact categories:
     - **Background** ([background.py](artifacts/background.py)): fluorescence, spontaneous luminescence, cell fragments
     - **Optical** ([optical.py](artifacts/optical.py)): chromatic aberration, vignetting, radial distortion
     - **Sensor** ([sensor.py](artifacts/sensor.py)): Poisson noise, readout noise, Gaussian noise, dynamic range compression

5. **Mask and Label Generation**
   - **Combined masks**: Binary masks for all active cells at each time step (default)
   - **Labels** ([labeling.py](utils/labeling.py)): Optional JSON metadata files with cell activity info
   - **CSV mapping** ([generate_csv.py](utils/generate_csv.py)): Creates `train.csv` mapping images to masks

### Batch Processing System

The `generate_simulation_batch()` function in [main.py](main.py) orchestrates large-scale dataset generation:

- **Memory Management**: Monitors system memory and triggers garbage collection at configurable thresholds
- **Sequential Processing**: Generates one simulation at a time, with aggressive cleanup between simulations
- **Output Structure**:
  ```
  output_dir/
    images/               # All simulation images
    masks/                # Combined binary masks for active cells
    labels/               # Optional JSON metadata (if --generate-labels)
    train.csv             # Image-mask mapping file
    simulation_results.json  # Batch metadata
  ```
- **Naming Convention**: `{sim_name}_t{timestep:05d}_{batch_run_id}.jpg` for images
- **CSV Format**: Each row maps one image to its corresponding mask file

### GUI System

The GUI ([main_window.py](gui/main_window.py)) provides:
- **Settings Panel** ([settings_panel.py](gui/settings_panel.py)): Configure simulation parameters and defects
- **Preview Panel** ([preview_panel.py](gui/preview_panel.py)): Real-time visualization of simulations
- **Dataset Panel** ([dataset_panel.py](gui/dataset_panel.py)): Batch generation controls
- **Tools Tab**: Cache cleaning utilities ([cache_cleaner.py](gui/cache_cleaner.py))

### Inference Module

Located in `inference/`, designed for SAM2 fine-tuning:
- **processing.py**: Prepares datasets for training (K-fold cross-validation framework)
- **inference.py**: Runs model training and evaluation
- **model_compare.py**: Compares different model configurations
- **visualization.py**: Visualizes segmentation results
- **load_sam2_direct.py**: Loads SAM2 models directly

Note: Inference scripts are meant to run in the SAM2 repository environment under WSL2.

## Important Implementation Details

### Pouch Class Internals

- **Cell Masks**: `cells_mask` attribute is a 2D array mapping each pixel to a cell ID (0 = background)
- **State Storage**: `disc_dynamics` stores all 4 state variables for all cells across all time steps
- **Active Cells**: Determined by calcium concentration threshold at each time step
- **Image Generation**: `generate_image()` method converts calcium concentrations to grayscale intensities, supports edge blur

### Parameter Ranges

For random parameter generation (`generate_random_params()` in [parameters.py](core/parameters.py)):
- Most parameters use uniform distributions within ±20% of defaults
- `frac` parameter varies between 0.001 and 0.03 (controls percentage of spontaneously active cells)
- `D_c_ratio` varies between 0.05 and 0.2 (calcium diffusion coefficient)

### Defect Configuration

Defect configs are dictionaries with boolean flags and intensity parameters. Key pattern:
```python
{
    'background_fluorescence': True/False,
    'background_intensity': float,
    'poisson_noise': True/False,
    # ... etc
}
```

Most defects are disabled by default in `generate_random_defect_config()` (only background defects enabled).

### Output Image Format

- Default: 512x512 JPEG images at 90% quality
- Configurable via `image_size` (format: "WIDTHxHEIGHT") and `jpeg_quality` parameters
- Masks are saved as JPEG for consistency, named with `_mask_combined.jpg` suffix

## Common Workflows

### Adding a New Defect Type

1. Create the defect function in the appropriate artifact module (background/optical/sensor)
2. Add defect config parameters to `generate_random_defect_config()` in [main.py](main.py)
3. Update `apply_all_defects()` in [image_processing.py](utils/image_processing.py) to call your function
4. Add GUI controls in [settings_panel.py](gui/settings_panel.py) if needed

### Modifying Simulation Parameters

1. Update `DEFAULT_PARAMS` or `SIM_TYPE_PARAMS` in [parameters.py](core/parameters.py)
2. Ensure the parameter is used in `Pouch.__init__()` and the PDE implementations in [pouch.py](core/pouch.py)
3. Add GUI controls if the parameter should be user-adjustable

### Changing Output Structure

- Batch output directory structure is controlled by `generate_simulation_batch()` in [main.py](main.py:143)
- File naming is handled in the image/mask saving loop around [main.py:326](main.py:326)
- CSV format is defined in [generate_csv.py](utils/generate_csv.py)

## Mathematical Model Notes

The calcium signaling simulation implements a 4-variable ODE system per cell:
- **Calcium dynamics**: Includes IP₃ receptor-mediated release, leak, and SERCA pump reuptake
- **IP₃ dynamics**: Generated by PLC activation (calcium-dependent), degraded over time
- **Spatial coupling**: Calcium and IP₃ diffuse between adjacent cells (determined by adjacency matrix)
- **Stochastic activation**: Cells randomly activate based on `frac` parameter to initiate waves

Key model characteristic: `V_PLC` parameter varies per cell (uniformly distributed between `lower` and `upper`) to create heterogeneous activation patterns.

## Platform-Specific Notes

- **Windows Compatibility**: All file paths use `os.path.join()` for cross-platform compatibility
- **WSL2 Requirement**: SAM2 inference modules require Linux environment (CUDA dependencies)
- **Memory Management**: Windows has different garbage collection behavior; aggressive cleanup is critical for batch processing
- **GUI**: Uses tkinter (Python standard library on Windows), requires proper window geometry handling
