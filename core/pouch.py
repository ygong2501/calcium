"""
Calcium signaling simulation engine.

This module implements the Pouch class, which simulates calcium ion dynamics
in epithelial tissue using a coupled ODE system. The simulation models:
- IP₃-mediated calcium release from ER stores
- Calcium diffusion between adjacent cells
- IP₃ diffusion and degradation
- SERCA pump-mediated calcium reuptake

The model generates realistic calcium wave patterns for training neural networks
to segment microscopy images.
"""
import os
import warnings
from typing import Tuple, Optional, Dict, List, Any

import numpy as np
import cv2
from skimage.draw import polygon

from .geometry_loader import GeometryLoader


class Pouch:
    """
    Simulates calcium ion dynamics in epithelial tissue.

    This class implements a spatially-extended model of calcium signaling
    where each cell is modeled as a node in a network. The dynamics include:
    - 4 state variables per cell (Ca²⁺, IP₃, ER Ca²⁺, IP₃R inactivation)
    - Diffusive coupling between adjacent cells
    - Stochastic initiation of calcium waves

    Attributes:
        size (str): Geometry size ('xsmall', 'small', 'medium', 'large', 'xlarge').
        n_cells (int): Number of cells in the tissue.
        T (int): Number of simulation time steps.
        dt (float): Time step size in seconds (default: 0.2s).
        disc_dynamics (np.ndarray): State variables over time (n_cells, 4, T).
        output_size (Tuple[int, int]): Image output dimensions (width, height).
    """

    # Simulation constants
    DEFAULT_TIME_STEP = 0.2  # seconds
    DEFAULT_DURATION = 3600  # seconds (1 hour)

    # Required parameter names
    REQUIRED_PARAMS = [
        'D_c_ratio', 'D_p', 'K_5', 'K_PLC', 'K_SERCA', 'V_SERCA',
        'beta', 'c_tot', 'frac', 'k_1', 'k_2', 'k_a', 'k_i',
        'k_p', 'k_tau', 'lower', 'tau_max', 'upper'
    ]

    def __init__(
        self,
        params: Optional[Dict[str, float]] = None,
        size: str = 'medium',
        sim_number: int = 0,
        save: bool = False,
        save_name: str = 'default',
        geometry_dir: Optional[str] = None,
        output_size: Tuple[int, int] = (512, 512),
        jpeg_quality: int = 90
    ):
        """
        Initialize calcium signaling simulation.

        Args:
            params: Dictionary of simulation parameters. If None, uses defaults.
            size: Geometry size ('xsmall', 'small', 'medium', 'large', 'xlarge').
            sim_number: Simulation ID for random seed and reproducibility.
            save: Whether to enable saving outputs (legacy parameter).
            save_name: Base name for saved files.
            geometry_dir: Custom directory for geometry files. If None, uses default.
            output_size: Output image dimensions (width, height) in pixels.
            jpeg_quality: JPEG quality for legacy image saving (0-100).

        Raises:
            ValueError: If required parameters are missing or invalid.
            RuntimeError: If geometry files cannot be loaded.
        """
        # Store configuration
        self.size = size
        self.save_name = save_name
        self.sim_number = sim_number
        self.save = save
        self.output_size = output_size
        self.jpeg_quality = jpeg_quality

        # Set parameters with defaults if not provided
        self.param_dict = params if params is not None else self._get_default_params()
        self._validate_parameters()

        # Load cell geometry from disk
        geometry_loader = GeometryLoader(geometry_dir)
        self.new_vertices, self.adj_matrix, self.laplacian_matrix = \
            geometry_loader.load_geometry(size)

        # Initialize simulation characteristics
        self.n_cells = self.adj_matrix.shape[0]
        self.dt = self.DEFAULT_TIME_STEP
        self.T = int(self.DEFAULT_DURATION / self.dt)  # Total time steps

        # Extract parameters for faster access during simulation
        self._unpack_parameters()

        # Calculate diffusion coefficients
        self.D_c = self.D_c_ratio * self.D_p  # Calcium diffusion
        self.D_p = self.D_p                    # IP₃ diffusion

        # Initialize V_PLC (PLC activation strength) per cell
        # V_PLC varies between 'lower' and 'upper' to create heterogeneous dynamics
        np.random.seed(sim_number)
        self.V_PLC = np.random.uniform(self.lower, self.upper, (self.n_cells, 1))

        # Initialize state storage (n_cells × 4 states × time_steps)
        # States: [0] Ca²⁺, [1] IP₃, [2] ER Ca²⁺, [3] IP₃R inactivation
        self.disc_dynamics = np.zeros((self.n_cells, 4, self.T))

        # Create cell mask for image generation
        self.cells_mask = self._create_cells_mask()

    def _get_default_params(self) -> Dict[str, float]:
        """Get default simulation parameters."""
        return {
            'K_PLC': 0.2, 'K_5': 0.66, 'k_1': 1.11, 'k_a': 0.08,
            'k_p': 0.13, 'k_2': 0.0203, 'V_SERCA': 0.9, 'K_SERCA': 0.1,
            'c_tot': 2.0, 'beta': 0.185, 'k_i': 0.4, 'D_p': 0.005,
            'tau_max': 800.0, 'k_tau': 1.5, 'lower': 0.5, 'upper': 0.7,
            'frac': 0.007680491551459293, 'D_c_ratio': 0.1
        }

    def _validate_parameters(self):
        """
        Validate that all required parameters are present.

        Raises:
            ValueError: If parameters are missing or extra parameters are present.
        """
        provided = set(self.param_dict.keys())
        required = set(self.REQUIRED_PARAMS)

        missing = required - provided
        extra = provided - required

        if missing or extra:
            error_parts = ["Parameter validation failed:"]
            if missing:
                error_parts.append(f"  Missing: {', '.join(sorted(missing))}")
            if extra:
                error_parts.append(f"  Unexpected: {', '.join(sorted(extra))}")
            raise ValueError('\n'.join(error_parts))

    def _unpack_parameters(self):
        """Unpack parameters to instance variables for fast access."""
        self.K_PLC = self.param_dict['K_PLC']
        self.K_5 = self.param_dict['K_5']
        self.k_1 = self.param_dict['k_1']
        self.k_a = self.param_dict['k_a']
        self.k_p = self.param_dict['k_p']
        self.k_2 = self.param_dict['k_2']
        self.V_SERCA = self.param_dict['V_SERCA']
        self.K_SERCA = self.param_dict['K_SERCA']
        self.c_tot = self.param_dict['c_tot']
        self.beta = self.param_dict['beta']
        self.k_i = self.param_dict['k_i']
        self.D_c_ratio = self.param_dict['D_c_ratio']
        self.tau_max = self.param_dict['tau_max']
        self.k_tau = self.param_dict['k_tau']
        self.lower = self.param_dict['lower']
        self.upper = self.param_dict['upper']
        self.frac = self.param_dict['frac']

    def _create_cells_mask(self) -> np.ndarray:
        """
        Create pixel-level cell mask mapping pixels to cell IDs.

        Returns:
            2D array where each pixel contains the ID of the cell it belongs to,
            or 0 for background pixels. Shape: (height, width).
        """
        width, height = self.output_size
        cells_mask = np.zeros((height, width), dtype=np.int32)

        # Get bounding box for all vertices
        all_x = np.concatenate([v[:, 0] for v in self.new_vertices])
        all_y = np.concatenate([v[:, 1] for v in self.new_vertices])
        x_min, x_max = np.min(all_x), np.max(all_x)
        y_min, y_max = np.min(all_y), np.max(all_y)

        # Draw each cell into the mask
        for cell_id, vertices in enumerate(self.new_vertices, start=1):
            # Scale vertices to image coordinates
            scaled_x = ((vertices[:, 0] - x_min) / (x_max - x_min) * (width - 1)).astype(int)
            scaled_y = ((vertices[:, 1] - y_min) / (y_max - y_min) * (height - 1)).astype(int)

            # Get pixels inside this cell polygon
            rr, cc = polygon(scaled_y, scaled_x, shape=(height, width))
            cells_mask[rr, cc] = cell_id

        return cells_mask

    def simulate(self) -> bool:
        """
        Run calcium dynamics simulation.

        Simulates the coupled ODE system for all cells over time using
        explicit Euler integration. The model equations are:

        dCa/dt = Laplacian(Ca)*D_c + J_channel + J_leak - J_SERCA
        dIP3/dt = Laplacian(IP3)*D_p + J_PLC - K_5*IP3
        dER/dt = (c_tot - Ca) / beta
        dR/dt = (inactivation recovery - inactivation)

        Returns:
            True if simulation completed successfully.

        Raises:
            ValueError: If parameters are invalid.
            RuntimeError: If numerical instabilities are detected.
        """
        # Validate parameters before expensive computation
        if self.c_tot <= 0 or self.beta <= 0 or self.D_p <= 0:
            raise ValueError("Invalid simulation parameters: c_tot, beta, and D_p must be positive")

        if not 0 <= self.frac <= 1:
            raise ValueError(f"Invalid fraction of initiator cells: {self.frac}. Must be between 0 and 1")

        try:
            # Set random seed for reproducibility
            np.random.seed(self.sim_number)

            # Initialize state variables at t=0
            # All cells start at resting state
            ca_init = 0.1  # Resting cytosolic Ca²⁺ (µM)
            ipt_init = 0.1  # Resting IP₃ (µM)
            r_init = 1.0   # IP₃R fully available

            self.disc_dynamics[:, 0, 0] = ca_init
            self.disc_dynamics[:, 1, 0] = ipt_init
            self.disc_dynamics[:, 2, 0] = (self.c_tot - ca_init) / self.beta
            self.disc_dynamics[:, 3, 0] = r_init

            # Determine which cells initiate spontaneously
            n_initiators = max(1, int(self.frac * self.n_cells))
            initiator_cells = np.random.choice(self.n_cells, n_initiators, replace=False)

            # Give initiator cells elevated IP₃ to start waves
            self.disc_dynamics[initiator_cells, 1, 0] = 0.5

            # V_PLC array for all cells (used in PLC activation)
            V_PLC = self.V_PLC.flatten()

            # Time integration loop
            epsilon = 1e-10  # Prevent division by zero

            for step in range(1, self.T):
                # Extract current state
                ca = self.disc_dynamics[:, 0, step-1].reshape(-1, 1)
                ipt = self.disc_dynamics[:, 1, step-1].reshape(-1, 1)
                s = self.disc_dynamics[:, 2, step-1].reshape(-1, 1)
                r = self.disc_dynamics[:, 3, step-1].reshape(-1, 1)

                # Calculate Laplacian (diffusion) terms
                ca_laplacian = self.D_c * np.dot(self.laplacian_matrix, ca)
                ipt_laplacian = self.D_p * np.dot(self.laplacian_matrix, ipt)

                # ODE right-hand sides

                # Calcium dynamics
                # J_channel: IP₃R-mediated release from ER
                ip3r_term = (r * ca * ipt) / ((self.k_a + ca + epsilon) * (self.k_p + ipt + epsilon))
                J_channel = (self.k_1 * ip3r_term**3 + self.k_2) * (s - ca)
                # J_SERCA: SERCA pump reuptake
                J_SERCA = self.V_SERCA * ca**2 / (ca**2 + self.K_SERCA**2 + epsilon)

                ca_next = ca + self.dt * (ca_laplacian + J_channel - J_SERCA)

                # IP₃ dynamics
                # J_PLC: Calcium-dependent IP₃ production
                J_PLC = V_PLC.reshape(-1, 1) * ca**2 / (ca**2 + self.K_PLC**2 + epsilon)
                ipt_next = ipt + self.dt * (ipt_laplacian + J_PLC - self.K_5 * ipt)

                # ER calcium (determined by conservation)
                s_next = (self.c_tot - ca_next) / self.beta

                # IP₃ receptor inactivation
                recovery_rate = (self.k_tau**4 + ca**4) / (self.tau_max * self.k_tau**4 + epsilon)
                inactivation = r * (self.k_i + ca) / (self.k_i + epsilon)
                r_next = r + self.dt * recovery_rate * (1 - inactivation)

                # Check for numerical issues
                if (np.isnan(ca_next).any() or np.isnan(ipt_next).any() or
                    np.isnan(s_next).any() or np.isnan(r_next).any()):
                    raise RuntimeError(f"Numerical instability detected at step {step}")

                # Store state (ensure physical constraints)
                self.disc_dynamics[:, 0, step] = np.clip(ca_next.flatten(), 0, None)
                self.disc_dynamics[:, 1, step] = np.clip(ipt_next.flatten(), 0, None)
                self.disc_dynamics[:, 2, step] = np.clip(s_next.flatten(), 0, None)
                self.disc_dynamics[:, 3, step] = np.clip(r_next.flatten(), 0, 1)

            return True

        except Exception as e:
            warnings.warn(f"Simulation error: {str(e)}")
            raise

    def generate_image(
        self,
        time_step: int,
        output_path: Optional[str] = None,
        with_border: bool = False,
        colormap: str = 'gray',
        edge_blur: bool = False,
        blur_kernel_size: int = 3,
        blur_type: str = 'mean'
    ) -> np.ndarray:
        """
        Generate microscopy-like image for a given time step.

        Creates a grayscale image with alpha channel showing calcium activity.
        All cells are displayed with intensity proportional to their calcium
        concentration. Background is fully transparent.

        Args:
            time_step: Time step to visualize.
            output_path: If provided, save image to this path.
            with_border: Whether to draw cell boundaries (default: False).
            colormap: Colormap name (kept for compatibility, always uses grayscale).
            edge_blur: Whether to blur cell edges for smoother appearance.
            blur_kernel_size: Size of blur kernel (pixels).
            blur_type: Blur type ('mean' or 'motion').

        Returns:
            Image array with shape (height, width, 2) for grayscale+alpha.
            Values are uint8, with grayscale in [0, 255] and alpha in [0, 255].

        Raises:
            ValueError: If time_step is out of range.
            RuntimeError: If image generation fails.
        """
        if time_step < 0 or time_step >= self.T:
            raise ValueError(f"Time step must be between 0 and {self.T-1}, got {time_step}")

        try:
            width, height = self.output_size

            # Create 2-channel image (grayscale + alpha)
            img_data = np.zeros((height, width, 2), dtype=np.uint8)

            # Get calcium activity
            calcium_activity = self.disc_dynamics[:, 0, time_step]

            # Normalize activity to grayscale range
            vmin = 0
            vmax = max(np.max(calcium_activity), 0.5)

            def map_to_grayscale(value: float) -> int:
                """Map calcium concentration to grayscale intensity."""
                normalized = np.clip((value - vmin) / (vmax - vmin), 0, 1)
                return int(255 * normalized)

            # Get vertex bounding box
            all_x = np.concatenate([v[:, 0] for v in self.new_vertices])
            all_y = np.concatenate([v[:, 1] for v in self.new_vertices])
            x_min, x_max = np.min(all_x), np.max(all_x)
            y_min, y_max = np.min(all_y), np.max(all_y)

            # Edge mask for optional border drawing
            edges_mask = None
            if with_border or edge_blur:
                edges_mask = np.zeros((height, width), dtype=np.uint8)

            # Draw each cell
            for cell_id, vertices in enumerate(self.new_vertices):
                gray_value = map_to_grayscale(calcium_activity[cell_id])

                # Scale vertices to image coordinates
                scaled_x = ((vertices[:, 0] - x_min) / (x_max - x_min) * (width - 1)).astype(int)
                scaled_y = ((vertices[:, 1] - y_min) / (y_max - y_min) * (height - 1)).astype(int)

                # Create polygon points
                points = np.column_stack([scaled_x, scaled_y])
                points = points.reshape((-1, 1, 2)).astype(np.int32)

                # Fill cell with grayscale + full opacity
                cv2.fillPoly(img_data, [points], color=(gray_value, 255))

                # Track edges if needed
                if edges_mask is not None:
                    cv2.polylines(edges_mask, [points], True, 255, 1)

            # Apply edge blur if requested
            if edge_blur and edges_mask is not None:
                kernel = self._create_blur_kernel(blur_kernel_size, blur_type)
                edges_blurred = cv2.filter2D(edges_mask, -1, kernel)
                edges_dilated = cv2.dilate(edges_blurred, np.ones((3, 3), np.uint8), iterations=1)
                edge_blend_mask = edges_dilated / 255.0

                if not with_border:
                    # Blur cell edges into background
                    img_blurred = cv2.filter2D(img_data, -1, kernel)
                    mask_2ch = np.stack([edge_blend_mask, edge_blend_mask], axis=-1)
                    img_data = (img_data * (1 - mask_2ch) + img_blurred * mask_2ch).astype(np.uint8)
                else:
                    # Darken edges
                    img_data[:, :, 0] = np.where(edges_mask > 0, img_data[:, :, 0] // 2, img_data[:, :, 0])

            # Draw explicit borders if requested
            if with_border and not edge_blur:
                for vertices in self.new_vertices:
                    scaled_x = ((vertices[:, 0] - x_min) / (x_max - x_min) * (width - 1)).astype(int)
                    scaled_y = ((vertices[:, 1] - y_min) / (y_max - y_min) * (height - 1)).astype(int)
                    points = np.column_stack([scaled_x, scaled_y]).reshape((-1, 1, 2)).astype(np.int32)
                    cv2.polylines(img_data[:, :, 0], [points], True, 0, 1)

            # Save if path provided
            if output_path is not None:
                from utils.image_processing import save_image
                save_image(img_data, os.path.dirname(output_path),
                          os.path.basename(output_path), format='png', bit_depth=10)

            return img_data

        except Exception as e:
            raise RuntimeError(f"Error generating image: {str(e)}")

    def _create_blur_kernel(self, size: int, blur_type: str) -> np.ndarray:
        """
        Create convolution kernel for edge blurring.

        Args:
            size: Kernel size (pixels).
            blur_type: Type of blur ('mean' or 'motion').

        Returns:
            Blur kernel as float32 array.
        """
        if blur_type == 'mean':
            return np.ones((size, size), np.float32) / (size * size)
        elif blur_type == 'motion':
            kernel = np.zeros((size, size), np.float32)
            kernel[size // 2, :] = 1.0 / size
            return kernel
        else:
            # Default to mean blur
            return np.ones((size, size), np.float32) / (size * size)

    def get_cell_masks(
        self,
        active_only: bool = False,
        time_step: Optional[int] = None,
        threshold: float = 0.1
    ) -> np.ndarray:
        """
        Get cell mask image.

        Args:
            active_only: If True, return only active cells.
            time_step: Required when active_only=True.
            threshold: Calcium threshold for determining active cells (µM).

        Returns:
            2D array where each pixel contains cell ID (or 0 for background).

        Raises:
            ValueError: If active_only=True but time_step not provided.
        """
        if active_only:
            if time_step is None:
                raise ValueError("time_step must be provided when active_only=True")

            # Create mask with only active cells
            active_cells = self.get_active_cells(time_step, threshold)
            active_mask = np.zeros_like(self.cells_mask)

            for cell_id in active_cells:
                active_mask[self.cells_mask == (cell_id + 1)] = cell_id + 1

            return active_mask
        else:
            return self.cells_mask.copy()

    def get_active_cells(self, time_step: int, threshold: float = 0.1) -> List[int]:
        """
        Get list of active cell indices at a given time step.

        A cell is considered active if its calcium concentration exceeds the threshold.

        Args:
            time_step: Time step to query.
            threshold: Calcium concentration threshold (µM).

        Returns:
            List of active cell indices (0-indexed).
        """
        calcium = self.disc_dynamics[:, 0, time_step]
        active_indices = np.where(calcium > threshold)[0]
        return active_indices.tolist()

    def generate_label_data(self, time_step: int, threshold: float = 0.1) -> Dict[str, Any]:
        """
        Generate comprehensive label data for a time step.

        Args:
            time_step: Time step to generate labels for.
            threshold: Activity threshold for cell detection.

        Returns:
            Dictionary containing:
            - active_cells: List of active cell IDs
            - calcium_values: Calcium concentration per active cell
            - n_active: Number of active cells
            - simulation_info: Metadata about this simulation
        """
        active_cells = self.get_active_cells(time_step, threshold)
        calcium_values = {
            cell_id: float(self.disc_dynamics[cell_id, 0, time_step])
            for cell_id in active_cells
        }

        return {
            'active_cells': active_cells,
            'calcium_values': calcium_values,
            'n_active': len(active_cells),
            'simulation_info': {
                'sim_number': self.sim_number,
                'size': self.size,
                'time_step': time_step,
                'total_cells': self.n_cells
            }
        }

    def __repr__(self) -> str:
        """String representation of Pouch object."""
        return (f"Pouch(size='{self.size}', n_cells={self.n_cells}, "
                f"sim_number={self.sim_number}, T={self.T})")

    def __str__(self) -> str:
        """Detailed string representation."""
        return (f"Calcium Signaling Simulation\n"
                f"  Geometry: {self.size} ({self.n_cells} cells)\n"
                f"  Duration: {self.T * self.dt:.0f}s ({self.T} steps @ {self.dt}s)\n"
                f"  Output: {self.output_size[0]}×{self.output_size[1]} pixels\n"
                f"  Sim ID: {self.sim_number}")
