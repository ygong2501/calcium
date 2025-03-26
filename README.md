# Calcium Ion Dynamic Simulation System

A Windows-compatible calcium ion dynamic simulation system designed to generate simulated cell calcium ion activity image datasets, specifically tailored for neural network training.

## Features

- **Windows Compatibility**: Fully compatible with Windows 11 and Python 3.9
- **Cell Simulation**: Generates calcium ion activity based on mathematical models
- **Video Generation**: Create animations of calcium dynamics over time
- **Defect Simulation**: Adds various realistic imaging defects:
  - Pre-optical defects (spontaneous luminescence, background fluorescence)
  - Optical propagation defects (radial distortion, chromatic aberration, vignetting)
  - Sensor defects (Poisson noise, readout noise, Gaussian noise)
  - Post-processing defects (defocus blur, dynamic range compression)
- **Standardized Output**: All images are generated at 512×512 pixel size
- **Batch Processing**: Generate many simulations with different parameters
- **Automatic Labeling**: Creates labels for cell activity and parameters
- **User-friendly GUI**: Visual interface for adjusting parameters and previewing results

## Mathematical Model

The simulation is based on a calcium signaling model that incorporates both intracellular and intercellular dynamics. The model uses a system of partial differential equations (PDEs) to simulate calcium ($Ca^{2+}$) and inositol trisphosphate (IP₃) dynamics within cells.

### Core Equations

The model incorporates four key state variables:
- $c$: cytosolic calcium concentration
- $p$: IP₃ concentration
- $s$: endoplasmic reticulum (ER) calcium concentration
- $r$: fraction of inactivated IP₃ receptors

#### Calcium Dynamics

$$\frac{dc}{dt} = D_c \nabla^2 c + \left(k_1 \left(\frac{r \cdot c \cdot p}{(k_a + c)(k_p + p)}\right)^3 + k_2\right)(s - c) - \frac{V_{SERCA} \cdot c^2}{c^2 + K_{SERCA}^2}$$

Where:
- $D_c$: Calcium diffusion coefficient
- $k_1, k_2$: Rate constants for calcium release
- $k_a, k_p$: Dissociation constants
- $V_{SERCA}$: Maximum SERCA pump rate
- $K_{SERCA}$: SERCA pump activation constant

#### IP₃ Dynamics

$$\frac{dp}{dt} = D_p \nabla^2 p + V_{PLC} \frac{c^2}{c^2 + K_{PLC}^2} - K_5 \cdot p$$

Where:
- $D_p$: IP₃ diffusion coefficient
- $V_{PLC}$: PLC activation potential (varies by cell)
- $K_{PLC}$: PLC activation constant
- $K_5$: IP₃ degradation rate

#### ER Calcium Concentration

$$s = \frac{c_{tot} - c}{\beta}$$

Where:
- $c_{tot}$: Total calcium concentration
- $\beta$: ER to cytosol volume ratio

#### IP₃ Receptor Inactivation

$$\frac{dr}{dt} = \frac{k_{\tau}^4 + c^4}{\tau_{max} \cdot k_{\tau}^4} \cdot \left(1 - r \cdot \frac{k_i + c}{k_i}\right)$$

Where:
- $\tau_{max}$: Maximum time constant
- $k_{\tau}$: Activation threshold
- $k_i$: Inhibition constant

## Installation

### Prerequisites

- Python 3.9 or later
- Windows 11 (recommended) or Windows 10
- FFmpeg (for animation saving)

### Setup

1. Clone this repository:
   ```
   git clone https://github.com/username/calcium_simulation.git
   cd calcium_simulation
   ```

2. Create a virtual environment (optional but recommended):
   ```
   conda create -n calcium
   conda activate calcium
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Download geometry files for cell structures
   - Download from the original project: https://github.com/MulticellularSystemsLab/MSELab_Calcium_Cartography_2021
   - Extract the geometry files to the `./calcium_simulation/geometry` directory
   - Ensure you have the following files:
     - `disc_vertices.npy`
     - `disc_sizes_laplacian.npy`
     - `disc_sizes_adj.npy`

## Usage

### GUI Mode

1. Launch the GUI:
   ```
   python main.py --gui
   ```

2. Use the interface to:
   - Adjust simulation parameters
   - Configure defects
   - Generate preview images
   - Create videos of calcium dynamics
   - Run batch simulations

### Command-Line Mode

1. Run batch simulation:
   ```
   python main.py --output ./output --num_simulations 10
   ```

2. Parameters:
   - `--output`: Output directory for simulation results
   - `--num_simulations`: Number of simulations to generate
   - `--pouch_sizes`: List of pouch sizes to use (e.g., 'small medium')
   - `--sim_types`: List of simulation types to use
   - `--num_threads`: Number of threads to use for parallel processing
   - `--gui`: Launch GUI interface
   - `--version`: Show program version and exit

3. Examples:
   ```
   # Generate 5 simulations with small and medium pouches only
   python main.py --output ./my_dataset --num_simulations 5 --pouch_sizes small medium
   
   # Generate 10 simulations using only the "Intercellular waves" type
   python main.py --output ./wave_dataset --num_simulations 10 --sim_types "Intercellular waves"
   
   # Generate 20 simulations using 4 processing threads
   python main.py --output ./large_dataset --num_simulations 20 --num_threads 4
   ```

## Batch Processing Workflow

The batch processing system allows the generation of large datasets with varied parameters for training machine learning models. Here's how it works:

1. **Parameter Selection**:
   - The system randomly selects parameters for each simulation from user-defined ranges
   - Each simulation can have different pouch sizes, simulation types, and parameter values
   - All simulations use the same defect configuration specified in the GUI or via presets

2. **Simulation Generation**:
   - For each simulation, the system:
     - Initializes a pouch with randomized parameters
     - Runs the calcium dynamics simulation
     - Captures images at specified time steps
     - Applies the configured defects to each image

3. **Dataset Organization**:
   - Output is structured in folders for easy dataset management:
     - `/train` (70% of images)
     - `/validation` (15% of images)
     - `/test` (15% of images)
   - Each folder contains:
     - Image files (PNG format)
     - JSON label files with detailed simulation parameters
     - Cell masks for segmentation tasks

4. **Multi-threading**:
   - Multiple simulations can run in parallel using the `--num_threads` parameter
   - Each thread handles one complete simulation
   - Progress is tracked and displayed in real-time

5. **Dataset Statistics**:
   - After batch generation completes, a summary JSON file contains:
     - Number of images in each split
     - Parameter distributions
     - Cell activity statistics

## File Structure

```
calcium_simulation/
├── core/                  # Core simulation classes
│   ├── pouch.py           # Main simulation class
│   ├── parameters.py      # Parameter management
│   └── geometry_loader.py # Geometry structure loading
├── artifacts/             # Defect simulation
│   ├── optical.py         # Optical defects
│   ├── sensor.py          # Sensor defects
│   └── background.py      # Background defects
├── utils/                 # Utility functions
│   ├── image_processing.py # Image processing tools
│   ├── dataset.py         # Dataset management
│   └── labeling.py        # Label generation
├── gui/                   # GUI components
│   ├── main_window.py     # Main window
│   ├── settings_panel.py  # Settings panel
│   └── preview_panel.py   # Preview panel
├── geometry/              # Geometry data files
├── presets/               # Preset configuration files
└── main.py                # Program entry point
```

## Parameter Descriptions

### Simulation Types

- **Single cell spikes**: Individual cells showing activity spikes
- **Intercellular transients**: Transient activations between cells
- **Intercellular waves**: Wave-like propagation of calcium signals
- **Fluttering**: Rapid oscillatory behavior of calcium signals

### Key Parameters

- **K_PLC**: PLC activation factor
- **K_5**: IP3 degradation rate
- **k_1**: IP3 receptor rate constant
- **D_p**: IP3 diffusion coefficient
- **D_c_ratio**: Ratio of calcium to IP3 diffusion 
- **frac**: Fraction of initiator cells

### Defect Configuration

- **Background fluorescence**: Uniform or non-uniform background glow
- **Spontaneous luminescence**: Random cell activity not related to actual signaling
- **Radial distortion**: Optical lens distortion effects
- **Chromatic aberration**: Color channel misalignment
- **Vignetting**: Darkening towards image edges
- **Noise types**: Poisson (intensity-dependent), readout (pattern), Gaussian (random)
- **Defocus blur**: Global or partial image blurring

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the calcium signaling simulation work by the Multicellular Systems Lab
- Original Jupyter notebook: Calcium_Signaling_Simulation_2022.ipynb