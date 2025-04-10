"""
Core simulation class for calcium signaling.
"""
import os
import numpy as np
import random
import math
import warnings
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import animation
import peakutils
from scipy.signal import find_peaks
from scipy.signal import peak_widths
from scipy.integrate import simps, solve_ivp
import pandas as pd
import cv2
from skimage.draw import polygon
from skimage.transform import resize

from .geometry_loader import GeometryLoader


class Pouch:
    """
    Class implementing pouch structure and simulating Calcium signaling.
    This is an adaptation of the original Pouch class, modified for Windows compatibility
    and to generate standardized 512x512 images without cell boundaries.
    """
    
    def __init__(self, params=None, size='xsmall', sim_number=0, save=False, save_name='default', 
                 geometry_dir=None, output_size=(512, 512)):
        """
        Initialize the Pouch object.
        
        Args:
            params (dict): Simulation parameters.
            size (str): Size of the pouch to simulate ('xsmall', 'small', 'medium', 'large').
            sim_number (int): Simulation ID for random seed and output naming.
            save (bool): Whether to save outputs.
            save_name (str): Additional name for saved files.
            geometry_dir (str, optional): Directory containing geometry files.
            output_size (tuple): Size of output images (width, height).
        """
        # Create characteristics of the pouch object
        self.size = size
        self.save_name = save_name
        self.sim_number = sim_number
        self.save = save
        self.param_dict = params
        self.output_size = output_size
        
        # If parameters are not set, then use baseline values
        if self.param_dict is None:
            self.param_dict = {
                'K_PLC': 0.2, 'K_5': 0.66, 'k_1': 1.11, 'k_a': 0.08, 
                'k_p': 0.13, 'k_2': 0.0203, 'V_SERCA': 0.9, 'K_SERCA': 0.1,
                'c_tot': 2, 'beta': .185, 'k_i': 0.4, 'D_p': 0.005, 
                'tau_max': 800, 'k_tau': 1.5, 'lower': 0.5, 'upper': 0.7, 
                'frac': 0.007680491551459293, 'D_c_ratio': 0.1
            }
        
        # If a dictionary is given, assure all parameters are provided
        required_params = [
            'D_c_ratio', 'D_p', 'K_5', 'K_PLC', 'K_SERCA', 'V_SERCA', 'beta', 
            'c_tot', 'frac', 'k_1', 'k_2', 'k_a', 'k_i', 'k_p', 'k_tau', 
            'lower', 'tau_max', 'upper'
        ]
        
        if sorted([r for r in self.param_dict]) != sorted(required_params):
            missing = set(required_params) - set(self.param_dict.keys())
            extra = set(self.param_dict.keys()) - set(required_params)
            error_msg = "Improper parameter input, please ensure all parameters are specified."
            if missing:
                error_msg += f"\nMissing parameters: {', '.join(missing)}"
            if extra:
                error_msg += f"\nUnexpected parameters: {', '.join(extra)}"
            raise ValueError(error_msg)
        
        # Load geometry
        geometry_loader = GeometryLoader(geometry_dir)
        self.new_vertices, self.adj_matrix, self.laplacian_matrix = geometry_loader.load_geometry(size)
        
        # Establish characteristics of the pouch for simulations
        self.n_cells = self.adj_matrix.shape[0]  # Number of cells in the pouch
        self.dt = .2  # Time step for ODE approximations
        self.T = int(3600/self.dt)  # Simulation to run for 3600 seconds (1 hour)
        
        # Establish baseline parameter values for the simulation
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
        self.D_p = self.param_dict['D_p']
        self.D_c = self.param_dict['D_c_ratio'] * self.D_p
        self.tau_max = self.param_dict['tau_max']
        self.k_tau = self.param_dict['k_tau']
        self.lower = self.param_dict['lower']
        self.upper = self.param_dict['upper']
        self.frac = self.param_dict['frac']
        
        # Initialize arrays for simulation
        self.disc_dynamics = np.zeros((self.n_cells, 4, self.T))  # Save simulation calcium, IP3, calcium_ER, ratio
        self.VPLC_state = np.zeros((self.n_cells, 1))  # Initialize VPLC array for cells
        
        # Create masks for cells
        self.cells_mask = self._create_cells_mask(output_size)
    
    def _create_cells_mask(self, size):
        """
        Create a mask array indicating which pixel belongs to which cell.
        
        Args:
            size (tuple): Size of the output image (width, height).
            
        Returns:
            numpy.ndarray: Mask indicating cell IDs for each pixel.
        """
        # Create an empty mask with the specified size
        width, height = size
        mask = np.zeros((height, width), dtype=np.int32)
        
        # Find the scaling factors to map cell vertices to the image dimensions
        x_coords = [v[:, 0] for v in self.new_vertices]
        y_coords = [v[:, 1] for v in self.new_vertices]
        
        all_x = np.concatenate(x_coords)
        all_y = np.concatenate(y_coords)
        
        x_min, x_max = np.min(all_x), np.max(all_x)
        y_min, y_max = np.min(all_y), np.max(all_y)
        
        # Create a normalized mask where each cell is filled with its index
        for i, cell in enumerate(self.new_vertices):
            # Normalize and scale vertices to image coordinates
            cell_x = ((cell[:, 0] - x_min) / (x_max - x_min) * (width - 1)).astype(int)
            cell_y = ((cell[:, 1] - y_min) / (y_max - y_min) * (height - 1)).astype(int)
            
            # Create a polygon for this cell
            points = np.vstack([cell_x, cell_y]).T
            
            # Convert points to the expected format for fillPoly
            points = points.reshape((-1, 1, 2)).astype(np.int32)
            
            # Use OpenCV's fillPoly for efficient polygon filling
            # Note: OpenCV expects points as (x, y) but mask is indexed as [y, x]
            cell_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(cell_mask, [points], color=1)
            
            # Add cell ID to the mask
            mask[cell_mask > 0] = i + 1
        
        return mask
    
    def simulate(self):
        """
        Simulate calcium dynamics.
        
        Returns:
            bool: True if simulation completed successfully
            
        Raises:
            ValueError: If simulation parameters are invalid
            RuntimeError: If numerical errors occur during simulation
        """
        # Validate parameters
        if self.c_tot <= 0 or self.beta <= 0 or self.D_p <= 0:
            raise ValueError("Invalid simulation parameters: c_tot, beta, and D_p must be positive")
        
        if self.frac < 0 or self.frac > 1:
            raise ValueError(f"Invalid fraction of initiator cells: {self.frac}. Must be between 0 and 1")
        
        # Set the seed for reproducibility
        np.random.seed(self.sim_number)  
        
        try:
            # Initialize simulation states
            self.disc_dynamics[:, 2, 0] = (self.c_tot - self.disc_dynamics[:, 0, 0]) / self.beta  # ER Calcium
            self.disc_dynamics[:, 3, 0] = np.random.uniform(.5, .7, size=(self.n_cells, 1)).T  # Fraction of inactivated IP3R
            
            # Initialize the values for VPLCs of standby cells
            self.VPLC_state = np.random.uniform(self.lower, self.upper, (self.n_cells, 1))
            
            # Choose which cells are initiator cells
            n_initiator_cells = max(1, int(self.frac * self.n_cells))  # Ensure at least one initiator cell
            stimulated_cell_idxs = np.random.choice(self.n_cells, n_initiator_cells, replace=False)
            self.VPLC_state[stimulated_cell_idxs, 0] = np.random.uniform(1.3, 1.5, len(stimulated_cell_idxs))
            
            V_PLC = self.VPLC_state.reshape((self.n_cells, 1))  # Establish the VPLCs for ODE approximations
            
            # ODE approximation solving
            for step in range(1, self.T):
                # ARRAY REFORMATTING
                ca = self.disc_dynamics[:, 0, step-1].reshape(-1, 1)
                ipt = self.disc_dynamics[:, 1, step-1].reshape(-1, 1)
                s = self.disc_dynamics[:, 2, step-1].reshape(-1, 1)
                r = self.disc_dynamics[:, 3, step-1].reshape(-1, 1)
                ca_laplacian = self.D_c * np.dot(self.laplacian_matrix, ca)
                ipt_laplacian = self.D_p * np.dot(self.laplacian_matrix, ipt)
                
                # Prevent division by zero with small epsilon
                epsilon = 1e-10
                
                # ODE EQUATIONS
                ca_next = (ca + self.dt * (
                    ca_laplacian + 
                    (self.k_1 * (np.divide(np.divide(r * np.multiply(ca, ipt), (self.k_a + ca + epsilon)), 
                                          (self.k_p + ipt + epsilon)))**3 + self.k_2) * 
                    (s - ca) - 
                    self.V_SERCA * (ca**2) / (ca**2 + self.K_SERCA**2 + epsilon)
                )).T
                
                ipt_next = (ipt + self.dt * (
                    ipt_laplacian + 
                    np.multiply(V_PLC, np.divide(ca**2, (ca**2 + self.K_PLC**2 + epsilon))) - 
                    self.K_5 * ipt
                )).T
                
                s_next = ((self.c_tot - ca) / self.beta).T
                
                r_next = (r + self.dt * 
                    ((self.k_tau**4 + ca**4) / (self.tau_max * self.k_tau**4 + epsilon)) * 
                    ((1 - r * (self.k_i + ca) / (self.k_i + epsilon)))
                ).T
                
                # Check for numerical instabilities
                if np.isnan(ca_next).any() or np.isnan(ipt_next).any() or np.isnan(s_next).any() or np.isnan(r_next).any():
                    raise RuntimeError(f"Numerical instability detected at step {step}")
                
                # Store results
                self.disc_dynamics[:, 0, step] = np.clip(ca_next, 0, None)  # Calcium can't be negative
                self.disc_dynamics[:, 1, step] = np.clip(ipt_next, 0, None)  # IP3 can't be negative
                self.disc_dynamics[:, 2, step] = np.clip(s_next, 0, None)  # ER Calcium can't be negative
                self.disc_dynamics[:, 3, step] = np.clip(r_next, 0, 1)  # Fraction must be between 0 and 1
                
            return True
            
        except Exception as e:
            warnings.warn(f"Simulation error: {str(e)}")
            raise
    
    def generate_image(self, time_step, output_path=None, with_border=False, colormap='gray'):
        """
        Generate single-frame image.
        
        Args:
            time_step (int): Time step to generate.
            output_path (str, optional): Output path, no save if None.
            with_border (bool): Whether to display cell boundaries.
            colormap (str): Matplotlib colormap to use.
            
        Returns:
            numpy.ndarray: Image array (512×512 pixels).
            
        Raises:
            ValueError: If time_step is out of range or parameters are invalid
        """
        if time_step < 0 or time_step >= self.T:
            raise ValueError(f"Time step must be between 0 and {self.T-1}")
        
        try:
            # Create a figure with the specified output size
            width, height = self.output_size
            dpi = 100  # Default DPI
            figsize = (width / dpi, height / dpi)
            
            # Create a new figure
            fig = plt.figure(figsize=figsize, dpi=dpi)
            ax = fig.add_subplot(1, 1, 1)
            ax.axis('off')
            fig.patch.set_alpha(0.)
            
            # Setup colormap
            cmap = plt.cm.get_cmap(colormap)
            
            # Get calcium activity for the current time step
            calcium_activity = self.disc_dynamics[:, 0, time_step]
            
            # Ensure low activity appears dark by setting proper normalization
            vmin = 0
            vmax = max(np.max(calcium_activity), 0.5)  # Ensure bright cells are visible
            normalize = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            
            # Draw the cells
            for i, cell in enumerate(self.new_vertices):
                # Get cell activity at this time step
                cell_activity = calcium_activity[i]
                
                # Map to color
                cell_color = colors.to_hex(cmap(normalize(cell_activity)), keep_alpha=False)
                
                # Draw cell polygon
                if with_border:
                    ax.plot(cell[:, 0], cell[:, 1], linewidth=1.0, color='k')
                else:
                    # No need to draw invisible lines
                    pass
                
                # Fill cell with appropriate color
                ax.fill(cell[:, 0], cell[:, 1], cell_color)
            
            # Adjust layout
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
            
            # Save if output path specified
            if output_path is not None:
                output_dir = os.path.dirname(output_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
            
            # Get the image data
            fig.canvas.draw()
            img_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img_data = img_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            # Close the figure to free memory
            plt.close(fig)
            
            # Resize to standard 512x512 if not already that size
            if img_data.shape[0] != 512 or img_data.shape[1] != 512:
                img_data = resize(img_data, (512, 512, 3), anti_aliasing=True)
                if img_data.max() <= 1.0:  # Check if resize returned values in [0,1]
                    img_data = (img_data * 255).astype(np.uint8)
            
            return img_data
            
        except Exception as e:
            plt.close()  # Ensure figure is closed in case of error
            raise RuntimeError(f"Error generating image: {str(e)}")
            
        finally:
            # Clean up any remaining matplotlib resources
            plt.close('all')
    
    def make_animation(self, path=None, fps=10, skip_frames=50):
        """
        Create animation of calcium dynamics.
        
        Args:
            path (str, optional): Path to save the animation.
            fps (int): Frames per second for the animation.
            skip_frames (int): Number of frames to skip between each animation frame.
        """
        # This version is similar to the original but outputs to a standard size
        colormap = plt.cm.gray
        normalize = matplotlib.colors.Normalize(
            vmin=np.min(self.disc_dynamics[:, 0, :]), 
            vmax=max(np.max(self.disc_dynamics[:, 0, :]), 1)
        )
        
        fig = plt.figure(figsize=(10, 10))
        fig.patch.set_alpha(0.)
        ax = fig.add_subplot(1, 1, 1)
        ax.axis('off')
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=normalize)
        sm._A = []
        cbar = fig.colorbar(sm, ax=ax)
        cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=15, fontweight="bold")
        
        # Create patches for each cell
        patches = []
        for cell in self.new_vertices:
            ax.plot(cell[:, 0], cell[:, 1], linewidth=0.0, color='w', alpha=0.0)
            patch = matplotlib.patches.Polygon(cell)
            patches.append(patch)
        
        def time_stamp_gen(n):
            j = 0
            while j < n:  # 0.2 sec interval to 1 hour time lapse
                yield "Elapsed time: " + '{0:02.0f}:{1:02.0f}'.format(*divmod(j * self.dt, 60))
                j += skip_frames
        
        time_stamps = time_stamp_gen(self.T)
        
        def init():
            return [ax.add_patch(p) for p in patches]
        
        def animate(frame, time_stamps):
            for j in range(len(patches)):
                c = colors.to_hex(colormap(normalize(frame[j])), keep_alpha=False)
                patches[j].set_facecolor(c)
            ax.set_title(next(time_stamps), fontsize=20, fontweight="bold")
            return patches
        
        anim = animation.FuncAnimation(
            fig, animate,
            init_func=init,
            frames=self.disc_dynamics[:, 0, ::skip_frames].T,
            fargs=(time_stamps,),
            interval=1000/fps,  # milliseconds between frames
            blit=True
        )
        
        if self.save and path is not None:
            if not os.path.exists(path):
                os.makedirs(path)
            
            output_file = os.path.join(
                path, 
                f"{self.size}Disc_{self.sim_number}_{self.save_name}.mp4"
            )
            anim.save(output_file, writer='ffmpeg', fps=fps)
        
        return anim
    
    def draw_profile(self, path=None, with_border=False):
        """
        Draw the VPLC Profile for the simulation.
        
        Args:
            path (str, optional): Path to save the profile image.
            with_border (bool): Whether to show cell borders.
        """
        # Create a standard-sized output
        width, height = self.output_size
        dpi = 100
        figsize = (width / dpi, height / dpi)
        
        colormap = plt.cm.Blues
        normalize = matplotlib.colors.Normalize(vmin=.0, vmax=1.5)
        
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(1, 1, 1)
        ax.axis('off')
        fig.patch.set_alpha(0.)
        
        for i, cell in enumerate(self.new_vertices):
            if with_border:
                ax.plot(cell[:, 0], cell[:, 1], linewidth=1.0, color='k')
            else:
                ax.plot(cell[:, 0], cell[:, 1], linewidth=0.0, color='w', alpha=0.0)
            
            c = colors.to_hex(colormap(normalize(self.VPLC_state[i]))[0], keep_alpha=False)
            ax.fill(cell[:, 0], cell[:, 1], c)
        
        # Adjust layout
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        
        if self.save and path is not None:
            if not os.path.exists(path):
                os.makedirs(path)
            
            output_file = os.path.join(
                path, 
                f"{self.size}Disc_VPLCProfile_{self.sim_number}_{self.save_name}.png"
            )
            fig.savefig(output_file, dpi=dpi, bbox_inches='tight', pad_inches=0)
        
        # Close the figure to free memory
        plt.close(fig)
    
    def get_cell_masks(self):
        """
        Get cell masks for labeling.
        
        Returns:
            numpy.ndarray: Array where each pixel value corresponds to a cell ID.
        """
        return self.cells_mask.copy()
    
    def get_active_cells(self, time_step, threshold=0.1):
        """
        Get active cells at the specified time step.
        
        Args:
            time_step (int): Time step to check.
            threshold (float): Activity threshold.
            
        Returns:
            list: List of active cell IDs.
        """
        activity = self.disc_dynamics[:, 0, time_step]
        active_cells = np.where(activity > threshold)[0]
        return active_cells.tolist()
    
    def generate_label_data(self, time_step, threshold=0.1):
        """
        Generate label data for the current time step.
        
        Args:
            time_step (int): Time step.
            threshold (float): Activity threshold.
            
        Returns:
            dict: Label data dictionary including active cells and their activity levels.
        """
        # Get active cells
        activity = self.disc_dynamics[:, 0, time_step]
        active_cells = np.where(activity > threshold)[0]
        
        # Create label data
        label_data = {
            'active_cells': active_cells.tolist(),
            'activity_levels': activity[active_cells].tolist(),
            'time_step': time_step,
            'simulation_id': self.sim_number,
            'simulation_type': self.save_name,
            'pouch_size': self.size
        }
        
        return label_data