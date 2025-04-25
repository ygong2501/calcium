# Calcium Ion Dynamic Simulation System

A Windows-compatible calcium ion dynamic simulation system designed to generate simulated cell calcium ion activity image datasets, specifically tailored for neural network training.

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

## License

This project is licensed under the LGPL-2.1 License - see the LICENSE file for details.

## Acknowledgments

- Based on the calcium signaling simulation work by the Multicellular Systems Lab
- Original Jupyter notebook: Calcium_Signaling_Simulation_2022.ipynb
