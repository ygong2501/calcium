# Calcium Ion Dynamic Simulation System

A Windows-compatible calcium ion dynamic simulation system designed to generate simulated cell calcium ion activity image datasets, specifically tailored for neural network training. The system now includes a K-fold cross-validation framework for optimizing loss functions in segmentation models.

## Features

- **Cell Simulation**: Generates calcium ion activity based on mathematical models
- **Defect Simulation**: Adds various realistic imaging defects:
  - Pre-optical defects (spontaneous luminescence, background fluorescence)
  - Optical propagation defects (radial distortion, chromatic aberration, vignetting)
  - Sensor defects (Poisson noise, readout noise, Gaussian noise)
  - Post-processing defects (defocus blur, dynamic range compression)
- **Batch Processing**: Generate many simulations with different parameters
- **User-friendly GUI**: Visual interface for adjusting parameters and previewing results
- **Cell Mask Generation**: Creates individual mask images for each active cell
- **CSV Mapping**: Optional CSV file generation to map between original images and their mask files
  
## Defect Configuration
- **Background fluorescence**: Uniform or non-uniform background glow
- **Spontaneous luminescence**: Random cell activity not related to actual signaling
- **Radial distortion**: Optical lens distortion effects
- **Chromatic aberration**: Color channel misalignment
- **Vignetting**: Darkening towards image edges
- **Noise types**: Poisson (intensity-dependent), readout (pattern), Gaussian (random)
- **Defocus blur**: Global or partial image blurring


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

1. Launch the GUI:
   ```
   python main.py --gui
   ```

2. Use the interface to:
   - Adjust simulation parameters
   - Configure defects
   - Generate preview images
   - Run batch simulations (see below for details on the workflow of batch simulation)
   - Generate cell masks and CSV mappings
   - Or you can use the provided sample dataset: https://drive.google.com/drive/folders/1CQ3P1wDruGD4ldzeL9slpSbnzzCuxwPC?usp=drive_link

3. Batch Options:
   - **Generate Masks**: Creates individual masks for each active cell (enabled by default)
   - **Create CSV File**: Creates a CSV mapping between original images and masks

4. To fine tune the SAM2 models
   - you need SAM2: https://github.com/facebookresearch/sam2
   - After configuring SAM2 as instructed, copy all files in "inference" to the root of your SAM2 folder
   - Run processing.py under your SAM2 environment under Windows Subsystem for Linux 2 followed by inference.py
   - By defualt, the last iteration's model will be save and be used for inferencing.
   
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

<<<<<<< Updated upstream
=======
<<<<<<< HEAD
3. **Dataset Organization**:
   - Output is structured in folders for easy dataset management:
     - `/train` (70% of images)
     - `/validation` (15% of images)
     - `/test` (15% of images)
   - Each folder contains:
     - Image files (JPG format with adjustable quality)
     - JSON label files with detailed simulation parameters
     - Cell masks for segmentation tasks (individual masks per active cell)

4. **Performance Optimization**:
   - Memory monitoring to prevent out-of-memory situations during batch processing
   - Optimized image processing pipeline
   - Progress is tracked and displayed in real-time

5. **Dataset Statistics**:
   - After batch generation completes, a summary JSON file contains:
     - Number of images in each split
     - Parameter distributions
     - Cell activity statistics
     - Mask and CSV file information when applicable

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
│   └── labeling.py        # Label generation and CSV mapping
├── gui/                   # GUI components
│   ├── main_window.py     # Main window
│   ├── settings_panel.py  # Settings panel
│   ├── preview_panel.py   # Preview panel
│   └── dataset_panel.py   # Dataset creation panel
├── geometry/              # Geometry data files
├── presets/               # Preset configuration files
├── output/                # Default output directory
│   └── simulation_batch_X/ # Each batch directory
│       ├── simulation_results.json # Results summary
│       ├── image_mask_mapping.csv  # CSV mapping file (if enabled)
│       └── [sim_name]/    # Individual simulation folders
│           ├── images/    # JPG image files
│           ├── labels/    # JSON label files
│           └── masks/     # Individual mask images
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

### New Features

- **Mask Generation**: Creates individual binary masks for each active cell in the simulation
  - Each cell gets a separate JPG file with "_mask_XXX" suffix where XXX is the cell ID
  - Masks are created using the same dimensions as the original images (512×512)
  - Only generated when "Generate Masks" option is selected in the Batch tab
  
- **CSV Mapping**: Creates a CSV file that maps original images to their corresponding mask files
  - Format: Two columns - "ImageId" and "MaskId"
  - Handles one-to-many mapping (one source image can have multiple mask images)
  - Uses a 10-digit zero-padded format for cell IDs to ensure consistent sorting
  - Provides a clean data format for machine learning training
  - Created when "Create CSV File" option is selected in the Batch tab

## K-fold Cross-validation Framework for Loss Function Selection

We've implemented a K-fold cross-validation framework to determine the optimal weighting between BCE and Dice loss functions (α and β, where β = 1-α) for segmentation models. The implementation consists of:

### 1. Modified Preprocessing

The modified preprocessing pipeline splits data in a 6:2:2 ratio (train:validation:test) and includes boundary-aware point sampling for better segmentation.

```
python finetuning/modified_preprocessing.py
```

### 2. K-fold Cross-validation

A 10-fold cross-validation framework that tests different α values (weight for BCE loss) to determine the optimal loss function.

```
python finetuning/k_fold/main.py --k 10 --alpha_values 0.1 0.3 0.5 0.7 0.9
```

### 3. Combined Loss Function

The loss function is a weighted combination of BCE and Dice loss:

Loss = α * BCE Loss + β * Dice Loss

where β = 1 - α

### 4. Training with Optimal Loss

Once the best α is determined, a final model is trained using the complete training set and evaluated on the test set.

```
python finetuning/modified_training.py --alpha [BEST_ALPHA]
```

=======
>>>>>>> 6b4c72d4b24f776eea8f58f7493473fe1ba6fbcc
>>>>>>> Stashed changes
## License

This project is licensed under the LGPL-2.1 License - see the LICENSE file for details.

## Acknowledgments

- Based on the calcium signaling simulation work by the Multicellular Systems Lab
- Original Jupyter notebook: Calcium_Signaling_Simulation_2022.ipynb
