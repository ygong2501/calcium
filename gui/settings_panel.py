"""
Settings panel for the calcium simulation GUI.
"""
import tkinter as tk
from tkinter import ttk
import numpy as np


class SettingsPanel:
    """Panel for simulation settings."""
    
    def __init__(self, parent, main_window):
        """
        Initialize the settings panel.
        
        Args:
            parent (ttk.Frame): Parent frame.
            main_window (MainWindow): Main window reference.
        """
        self.parent = parent
        self.main_window = main_window
        
        # Create settings frame - parent already has width constraint
        self.frame = ttk.Frame(parent)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.sim_tab = ttk.Frame(self.notebook)
        self.defect_tab = ttk.Frame(self.notebook)
        self.batch_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.sim_tab, text="Simulation")
        self.notebook.add(self.defect_tab, text="Defects")
        self.notebook.add(self.batch_tab, text="Batch")
        
        # Setup tabs
        self._setup_simulation_tab()
        self._setup_defect_tab()
        self._setup_batch_tab()
    
    def _setup_simulation_tab(self):
        """Setup the simulation settings tab."""
        # Create a proper scrollable frame with fixed width
        # Container frame
        container = ttk.Frame(self.sim_tab)
        container.pack(fill=tk.BOTH, expand=True)
        
        # Canvas with scrollbar
        canvas = tk.Canvas(container)
        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=canvas.yview)
        
        # Scrollable frame inside canvas
        scrollable_frame = ttk.Frame(canvas)
        
        # Configure scrolling
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        # Make canvas use scrollbar
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Create window inside canvas containing the scrollable frame
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        # Pack scrollbar first (right side)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Then pack canvas to fill remaining space
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Simulation Type
        ttk.Label(scrollable_frame, text="Simulation Type:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.sim_type_var = tk.StringVar()
        self.sim_type_combo = ttk.Combobox(scrollable_frame, textvariable=self.sim_type_var, width=25)
        self.sim_type_combo['values'] = (
            "Single cell spikes",
            "Intercellular transients",
            "Intercellular waves",
            "Fluttering"
        )
        self.sim_type_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Pouch Size
        ttk.Label(scrollable_frame, text="Pouch Size:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.pouch_size_var = tk.StringVar()
        self.pouch_size_combo = ttk.Combobox(scrollable_frame, textvariable=self.pouch_size_var, width=25)
        self.pouch_size_combo['values'] = ("xsmall", "small", "medium", "large")
        self.pouch_size_combo.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Time Step
        ttk.Label(scrollable_frame, text="Time Step:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.time_step_var = tk.StringVar()
        self.time_step_entry = ttk.Entry(scrollable_frame, textvariable=self.time_step_var, width=10)
        self.time_step_entry.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Separator
        ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL).grid(
            row=3, column=0, columnspan=2, padx=5, pady=10, sticky=tk.EW
        )
        
        # Advanced Parameters Label
        ttk.Label(
            scrollable_frame, text="Advanced Parameters:", font=("", 10, "bold")
        ).grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)
        
        # Parameter entries
        self.param_vars = {}
        param_names = [
            'K_PLC', 'K_5', 'k_1', 'k_a', 'k_p', 'k_2', 'V_SERCA', 'K_SERCA',
            'c_tot', 'beta', 'k_i', 'D_p', 'tau_max', 'k_tau', 'lower', 'upper',
            'frac', 'D_c_ratio'
        ]
        
        for i, param in enumerate(param_names):
            row = i + 5  # Start after the header
            ttk.Label(scrollable_frame, text=f"{param}:").grid(
                row=row, column=0, padx=5, pady=2, sticky=tk.W
            )
            self.param_vars[param] = tk.StringVar()
            ttk.Entry(scrollable_frame, textvariable=self.param_vars[param], width=10).grid(
                row=row, column=1, padx=5, pady=2, sticky=tk.W
            )
    
    def _setup_defect_tab(self):
        """Setup the defect settings tab."""
        # Container frame
        container = ttk.Frame(self.defect_tab)
        container.pack(fill=tk.BOTH, expand=True)
        
        # Canvas with scrollbar
        canvas = tk.Canvas(container)
        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=canvas.yview)
        
        # Scrollable frame inside canvas
        scrollable_frame = ttk.Frame(canvas)
        
        # Configure scrolling
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        # Make canvas use scrollbar
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Create window inside canvas containing the scrollable frame
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        # Pack scrollbar first (right side)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Then pack canvas to fill remaining space
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Background Defects Section
        ttk.Label(
            scrollable_frame, text="Background Defects:", font=("", 10, "bold")
        ).grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)
        
        # Background Fluorescence
        self.bg_fluor_var = tk.BooleanVar()
        ttk.Checkbutton(
            scrollable_frame, text="Background Fluorescence", variable=self.bg_fluor_var
        ).grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        
        ttk.Label(scrollable_frame, text="Intensity:").grid(row=2, column=0, padx=20, pady=2, sticky=tk.W)
        self.bg_intensity_var = tk.StringVar()
        ttk.Entry(scrollable_frame, textvariable=self.bg_intensity_var, width=10).grid(
            row=2, column=1, padx=5, pady=2, sticky=tk.W
        )
        
        self.non_uniform_bg_var = tk.BooleanVar()
        ttk.Checkbutton(
            scrollable_frame, text="Non-uniform Background", variable=self.non_uniform_bg_var
        ).grid(row=3, column=0, padx=20, pady=2, sticky=tk.W)
        
        # Spontaneous Luminescence
        self.spont_lum_var = tk.BooleanVar()
        ttk.Checkbutton(
            scrollable_frame, text="Spontaneous Luminescence", variable=self.spont_lum_var
        ).grid(row=4, column=0, padx=5, pady=2, sticky=tk.W)
        
        ttk.Label(scrollable_frame, text="Min Intensity:").grid(row=5, column=0, padx=20, pady=2, sticky=tk.W)
        self.spont_min_var = tk.StringVar()
        ttk.Entry(scrollable_frame, textvariable=self.spont_min_var, width=10).grid(
            row=5, column=1, padx=5, pady=2, sticky=tk.W
        )
        
        ttk.Label(scrollable_frame, text="Max Intensity:").grid(row=6, column=0, padx=20, pady=2, sticky=tk.W)
        self.spont_max_var = tk.StringVar()
        ttk.Entry(scrollable_frame, textvariable=self.spont_max_var, width=10).grid(
            row=6, column=1, padx=5, pady=2, sticky=tk.W
        )
        
        ttk.Label(scrollable_frame, text="Probability:").grid(row=7, column=0, padx=20, pady=2, sticky=tk.W)
        self.spont_prob_var = tk.StringVar()
        ttk.Entry(scrollable_frame, textvariable=self.spont_prob_var, width=10).grid(
            row=7, column=1, padx=5, pady=2, sticky=tk.W
        )
        
        # Cell Fragments
        self.cell_frag_var = tk.BooleanVar()
        ttk.Checkbutton(
            scrollable_frame, text="Cell Fragments", variable=self.cell_frag_var
        ).grid(row=8, column=0, padx=5, pady=2, sticky=tk.W)
        
        ttk.Label(scrollable_frame, text="Count:").grid(row=9, column=0, padx=20, pady=2, sticky=tk.W)
        self.frag_count_var = tk.StringVar()
        ttk.Entry(scrollable_frame, textvariable=self.frag_count_var, width=10).grid(
            row=9, column=1, padx=5, pady=2, sticky=tk.W
        )
        
        # Separator
        ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL).grid(
            row=10, column=0, columnspan=2, padx=5, pady=10, sticky=tk.EW
        )
        
        # Optical Defects Section
        ttk.Label(
            scrollable_frame, text="Optical Defects:", font=("", 10, "bold")
        ).grid(row=11, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)
        
        # Placeholder for future optical defects
        # Removed radial distortion
        
        # Chromatic Aberration
        self.chromatic_ab_var = tk.BooleanVar()
        ttk.Checkbutton(
            scrollable_frame, text="Chromatic Aberration", variable=self.chromatic_ab_var
        ).grid(row=13, column=0, padx=5, pady=2, sticky=tk.W)
        
        # Vignetting
        self.vignetting_var = tk.BooleanVar()
        ttk.Checkbutton(
            scrollable_frame, text="Vignetting", variable=self.vignetting_var
        ).grid(row=14, column=0, padx=5, pady=2, sticky=tk.W)
        
        ttk.Label(scrollable_frame, text="Strength:").grid(row=15, column=0, padx=20, pady=2, sticky=tk.W)
        self.vignetting_strength_var = tk.StringVar()
        ttk.Entry(scrollable_frame, textvariable=self.vignetting_strength_var, width=10).grid(
            row=15, column=1, padx=5, pady=2, sticky=tk.W
        )
        
        # Separator
        ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL).grid(
            row=16, column=0, columnspan=2, padx=5, pady=10, sticky=tk.EW
        )
        
        # Sensor Defects Section
        ttk.Label(
            scrollable_frame, text="Sensor Defects:", font=("", 10, "bold")
        ).grid(row=17, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)
        
        # Poisson Noise
        self.poisson_noise_var = tk.BooleanVar()
        ttk.Checkbutton(
            scrollable_frame, text="Poisson Noise", variable=self.poisson_noise_var
        ).grid(row=18, column=0, padx=5, pady=2, sticky=tk.W)
        
        # Readout Noise
        self.readout_noise_var = tk.BooleanVar()
        ttk.Checkbutton(
            scrollable_frame, text="Readout Noise", variable=self.readout_noise_var
        ).grid(row=19, column=0, padx=5, pady=2, sticky=tk.W)
        
        # Gaussian Noise
        self.gaussian_noise_var = tk.BooleanVar()
        ttk.Checkbutton(
            scrollable_frame, text="Gaussian Noise", variable=self.gaussian_noise_var
        ).grid(row=20, column=0, padx=5, pady=2, sticky=tk.W)
        
        ttk.Label(scrollable_frame, text="Sigma:").grid(row=21, column=0, padx=20, pady=2, sticky=tk.W)
        self.gaussian_sigma_var = tk.StringVar()
        ttk.Entry(scrollable_frame, textvariable=self.gaussian_sigma_var, width=10).grid(
            row=21, column=1, padx=5, pady=2, sticky=tk.W
        )
        
        # Separator
        ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL).grid(
            row=22, column=0, columnspan=2, padx=5, pady=10, sticky=tk.EW
        )
        
        # Post-processing Defects Section
        ttk.Label(
            scrollable_frame, text="Post-processing Defects:", font=("", 10, "bold")
        ).grid(row=23, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)
        
        # Defocus Blur
        self.defocus_blur_var = tk.BooleanVar()
        ttk.Checkbutton(
            scrollable_frame, text="Defocus Blur", variable=self.defocus_blur_var
        ).grid(row=24, column=0, padx=5, pady=2, sticky=tk.W)
        
        ttk.Label(scrollable_frame, text="Sigma:").grid(row=25, column=0, padx=20, pady=2, sticky=tk.W)
        self.defocus_sigma_var = tk.StringVar()
        ttk.Entry(scrollable_frame, textvariable=self.defocus_sigma_var, width=10).grid(
            row=25, column=1, padx=5, pady=2, sticky=tk.W
        )
        
        # Partial Defocus
        self.partial_defocus_var = tk.BooleanVar()
        ttk.Checkbutton(
            scrollable_frame, text="Partial Defocus", variable=self.partial_defocus_var
        ).grid(row=26, column=0, padx=20, pady=2, sticky=tk.W)
        
        # Brightness/Contrast Adjustment
        self.adjust_bc_var = tk.BooleanVar()
        ttk.Checkbutton(
            scrollable_frame, text="Adjust Brightness/Contrast", variable=self.adjust_bc_var
        ).grid(row=27, column=0, padx=5, pady=2, sticky=tk.W)
        
        # Separator
        ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL).grid(
            row=28, column=0, columnspan=2, padx=5, pady=10, sticky=tk.EW
        )
        
        # Edge Blur Options
        ttk.Label(
            scrollable_frame, text="Edge Blur Options:", font=("", 10, "bold")
        ).grid(row=29, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)
        
        # Edge blur checkbox
        self.edge_blur_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            scrollable_frame, text="Apply Edge Blur", variable=self.edge_blur_var
        ).grid(row=30, column=0, padx=5, pady=2, sticky=tk.W)
        
        # Blur kernel size
        ttk.Label(scrollable_frame, text="Kernel Size:").grid(row=31, column=0, padx=20, pady=2, sticky=tk.W)
        self.blur_kernel_size_var = tk.StringVar(value="3")
        ttk.Combobox(
            scrollable_frame, textvariable=self.blur_kernel_size_var, 
            values=["3", "5", "7", "9"], width=8
        ).grid(row=31, column=1, padx=5, pady=2, sticky=tk.W)
        
        # Blur type
        ttk.Label(scrollable_frame, text="Blur Type:").grid(row=32, column=0, padx=20, pady=2, sticky=tk.W)
        self.blur_type_var = tk.StringVar(value="mean")
        ttk.Combobox(
            scrollable_frame, textvariable=self.blur_type_var, 
            values=["mean", "motion"], width=8
        ).grid(row=32, column=1, padx=5, pady=2, sticky=tk.W)
    
    def _setup_batch_tab(self):
        """Setup the batch generation tab."""
        # Container frame
        container = ttk.Frame(self.batch_tab)
        container.pack(fill=tk.BOTH, expand=True)
        
        # Canvas with scrollbar
        canvas = tk.Canvas(container)
        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=canvas.yview)
        
        # Scrollable frame inside canvas
        scrollable_frame = ttk.Frame(canvas)
        
        # Configure scrolling
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        # Make canvas use scrollbar
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Create window inside canvas containing the scrollable frame
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        # Pack scrollbar first (right side)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Then pack canvas to fill remaining space
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Number of Simulations
        ttk.Label(scrollable_frame, text="Number of Simulations:").grid(
            row=0, column=0, padx=5, pady=5, sticky=tk.W
        )
        self.num_sims_var = tk.StringVar()
        ttk.Entry(scrollable_frame, textvariable=self.num_sims_var, width=10).grid(
            row=0, column=1, padx=5, pady=5, sticky=tk.W
        )
        
        # Pouch Sizes - use more vertical space to avoid crowding
        ttk.Label(scrollable_frame, text="Pouch Sizes:").grid(
            row=1, column=0, padx=5, pady=5, sticky=tk.W
        )
        
        self.pouch_size_vars = {}
        sizes = ["xsmall", "small", "medium", "large"]
        
        # Layout options in a 2x2 grid
        size_row_offset = 2
        for i, size in enumerate(sizes):
            # Calculate row and column (2 per row)
            row = size_row_offset + (i // 2)
            col = i % 2
            
            self.pouch_size_vars[size] = tk.BooleanVar()
            ttk.Checkbutton(
                scrollable_frame, text=size, variable=self.pouch_size_vars[size]
            ).grid(row=row, column=col, padx=5, pady=2, sticky=tk.W)
        
        # Simulation Types - adjust starting row to account for new pouch size layout
        ttk.Label(scrollable_frame, text="Simulation Types:").grid(
            row=4, column=0, padx=5, pady=5, sticky=tk.W
        )
        
        self.sim_type_vars = {}
        sim_types = [
            "Single cell spikes",
            "Intercellular transients",
            "Intercellular waves",
            "Fluttering"
        ]
        
        for i, sim_type in enumerate(sim_types):
            self.sim_type_vars[sim_type] = tk.BooleanVar()
            ttk.Checkbutton(
                scrollable_frame, text=sim_type, variable=self.sim_type_vars[sim_type]
            ).grid(row=i+5, column=0, columnspan=4, padx=5, pady=2, sticky=tk.W)
        
        # Time Steps - adjust row to account for new layout
        ttk.Label(scrollable_frame, text="Time Step Range:").grid(
            row=9, column=0, padx=5, pady=5, sticky=tk.W
        )
        
        ttk.Label(scrollable_frame, text="Start:").grid(
            row=10, column=0, padx=20, pady=2, sticky=tk.W
        )
        self.time_start_var = tk.StringVar()
        ttk.Entry(scrollable_frame, textvariable=self.time_start_var, width=10).grid(
            row=10, column=1, padx=5, pady=2, sticky=tk.W
        )
        
        ttk.Label(scrollable_frame, text="End:").grid(
            row=11, column=0, padx=20, pady=2, sticky=tk.W
        )
        self.time_end_var = tk.StringVar()
        ttk.Entry(scrollable_frame, textvariable=self.time_end_var, width=10).grid(
            row=11, column=1, padx=5, pady=2, sticky=tk.W
        )
        
        ttk.Label(scrollable_frame, text="Step:").grid(
            row=12, column=0, padx=20, pady=2, sticky=tk.W
        )
        self.time_step_size_var = tk.StringVar()
        ttk.Entry(scrollable_frame, textvariable=self.time_step_size_var, width=10).grid(
            row=12, column=1, padx=5, pady=2, sticky=tk.W
        )
        
        # No dataset options here - removed to prevent duplication with Dataset Creation tab
        
        # No edge blur options here (moved to defects tab)
    
    def load_default_values(self):
        """Load default values for all settings."""
        # Simulation defaults
        self.sim_type_var.set("Intercellular waves")
        self.pouch_size_var.set("small")
        self.time_step_var.set("1000")  # 200 seconds into simulation
        
        # Parameter defaults
        default_params = {
            'K_PLC': 0.2, 'K_5': 0.66, 'k_1': 1.11, 'k_a': 0.08,
            'k_p': 0.13, 'k_2': 0.0203, 'V_SERCA': 0.9, 'K_SERCA': 0.1,
            'c_tot': 2, 'beta': .185, 'k_i': 0.4, 'D_p': 0.005,
            'tau_max': 800, 'k_tau': 1.5, 'lower': 0.4, 'upper': 0.8,
            'frac': 0.007680491551459293, 'D_c_ratio': 0.1
        }
        
        for param, value in default_params.items():
            if param in self.param_vars:
                self.param_vars[param].set(str(value))
        
        # Defect defaults
        self.bg_fluor_var.set(True)
        self.bg_intensity_var.set("0.1")
        self.non_uniform_bg_var.set(True)
        
        self.spont_lum_var.set(True)
        self.spont_min_var.set("0.05")
        self.spont_max_var.set("0.15")
        self.spont_prob_var.set("0.2")
        
        self.cell_frag_var.set(True)
        self.frag_count_var.set("5")
        
        # Radial distortion option removed
        self.chromatic_ab_var.set(False)
        self.vignetting_var.set(False)
        self.vignetting_strength_var.set("0.5")
        
        self.poisson_noise_var.set(False)
        self.readout_noise_var.set(False)
        self.gaussian_noise_var.set(False)
        self.gaussian_sigma_var.set("0.1")
        
        self.defocus_blur_var.set(False)
        self.defocus_sigma_var.set("3.0")
        self.partial_defocus_var.set(False)
        self.adjust_bc_var.set(False)
        
        # Edge blur defaults
        self.edge_blur_var.set(False)
        self.blur_kernel_size_var.set("3")
        self.blur_type_var.set("mean")
        
        # Batch defaults
        self.num_sims_var.set("5")
        
        for size in self.pouch_size_vars:
            self.pouch_size_vars[size].set(True)
            
        for sim_type in self.sim_type_vars:
            self.sim_type_vars[sim_type].set(True)
        
        self.time_start_var.set("0")
        self.time_end_var.set("18000")
        self.time_step_size_var.set("200")
        
        # Dataset defaults removed
        
        # Edge blur defaults
        self.edge_blur_var.set(False)
        self.blur_kernel_size_var.set("3")
        self.blur_type_var.set("mean")
    
    def get_simulation_parameters(self):
        """
        Get simulation parameters from form.
        
        Returns:
            dict: Simulation parameters.
        """
        params = {}
        
        # Basic parameters
        params['sim_type'] = self.sim_type_var.get()
        params['pouch_size'] = self.pouch_size_var.get()
        params['time_step'] = int(self.time_step_var.get())
        
        # Advanced parameters
        for param, var in self.param_vars.items():
            try:
                value = float(var.get())
                params[param] = value
            except ValueError:
                # Use default if conversion fails
                pass
        
        return params
    
    def get_defect_configuration(self):
        """
        Get defect configuration from form.
        
        Returns:
            dict: Defect configuration.
        """
        config = {}
        
        # Background defects
        config['background_fluorescence'] = self.bg_fluor_var.get()
        config['background_intensity'] = float(self.bg_intensity_var.get())
        config['non_uniform_background'] = self.non_uniform_bg_var.get()
        
        config['spontaneous_luminescence'] = self.spont_lum_var.get()
        config['spontaneous_min'] = float(self.spont_min_var.get())
        config['spontaneous_max'] = float(self.spont_max_var.get())
        config['spontaneous_probability'] = float(self.spont_prob_var.get())
        
        config['cell_fragments'] = self.cell_frag_var.get()
        config['fragment_count'] = int(self.frag_count_var.get())
        
        # Optical defects
        config['radial_distortion'] = False # Radial distortion option removed
        config['chromatic_aberration'] = self.chromatic_ab_var.get()
        config['vignetting'] = self.vignetting_var.get()
        config['vignetting_strength'] = float(self.vignetting_strength_var.get())
        
        # Sensor defects
        config['poisson_noise'] = self.poisson_noise_var.get()
        config['readout_noise'] = self.readout_noise_var.get()
        config['gaussian_noise'] = self.gaussian_noise_var.get()
        config['gaussian_sigma'] = float(self.gaussian_sigma_var.get())
        
        # Post-processing defects
        config['defocus_blur'] = self.defocus_blur_var.get()
        config['defocus_sigma'] = float(self.defocus_sigma_var.get())
        config['partial_defocus'] = self.partial_defocus_var.get()
        config['adjust_brightness_contrast'] = self.adjust_bc_var.get()
        
        # Edge blur options (now in defects tab)
        config['edge_blur'] = self.edge_blur_var.get()
        config['blur_kernel_size'] = int(self.blur_kernel_size_var.get())
        config['blur_type'] = self.blur_type_var.get()
        
        return config
    
    def get_batch_parameters(self):
        """
        Get batch generation parameters from form.
        
        Returns:
            dict: Batch parameters.
        """
        params = {}
        
        # Number of simulations
        params['num_simulations'] = int(self.num_sims_var.get())
        
        # Pouch sizes
        pouch_sizes = []
        for size, var in self.pouch_size_vars.items():
            if var.get():
                pouch_sizes.append(size)
        params['pouch_sizes'] = pouch_sizes
        
        # Simulation types
        sim_types = []
        for sim_type, var in self.sim_type_vars.items():
            if var.get():
                sim_types.append(sim_type)
        params['sim_types'] = sim_types
        
        # Time steps
        time_start = int(self.time_start_var.get())
        time_end = int(self.time_end_var.get())
        time_step = int(self.time_step_size_var.get())
        
        params['time_steps'] = list(range(time_start, time_end, time_step))
        
        # Dataset options removed - now handled entirely in Dataset Creation tab
        
        # Edge blur options are now retrieved from defect configuration
        # to ensure consistency between preview and batch generation
        defect_config = self.get_defect_configuration()
        params['edge_blur'] = defect_config.get('edge_blur', False)
        params['blur_kernel_size'] = defect_config.get('blur_kernel_size', 3)
        params['blur_type'] = defect_config.get('blur_type', 'mean')
        
        return params
    
    def set_simulation_parameters(self, params):
        """
        Set simulation parameters in the form.
        
        Args:
            params (dict): Simulation parameters.
        """
        # Basic parameters
        if 'sim_type' in params:
            self.sim_type_var.set(params['sim_type'])
        if 'pouch_size' in params:
            self.pouch_size_var.set(params['pouch_size'])
        if 'time_step' in params:
            self.time_step_var.set(str(params['time_step']))
        
        # Advanced parameters
        for param, var in self.param_vars.items():
            if param in params:
                var.set(str(params[param]))
    
    def set_defect_configuration(self, config):
        """
        Set defect configuration in the form.
        
        Args:
            config (dict): Defect configuration.
        """
        # Background defects
        if 'background_fluorescence' in config:
            self.bg_fluor_var.set(config['background_fluorescence'])
        if 'background_intensity' in config:
            self.bg_intensity_var.set(str(config['background_intensity']))
        if 'non_uniform_background' in config:
            self.non_uniform_bg_var.set(config['non_uniform_background'])
        
        if 'spontaneous_luminescence' in config:
            self.spont_lum_var.set(config['spontaneous_luminescence'])
        if 'spontaneous_min' in config:
            self.spont_min_var.set(str(config['spontaneous_min']))
        if 'spontaneous_max' in config:
            self.spont_max_var.set(str(config['spontaneous_max']))
        if 'spontaneous_probability' in config:
            self.spont_prob_var.set(str(config['spontaneous_probability']))
        
        if 'cell_fragments' in config:
            self.cell_frag_var.set(config['cell_fragments'])
        if 'fragment_count' in config:
            self.frag_count_var.set(str(config['fragment_count']))
        
        # Optical defects
        if 'radial_distortion' in config:
            self.radial_dist_var.set(config['radial_distortion'])
        if 'chromatic_aberration' in config:
            self.chromatic_ab_var.set(config['chromatic_aberration'])
        if 'vignetting' in config:
            self.vignetting_var.set(config['vignetting'])
        if 'vignetting_strength' in config:
            self.vignetting_strength_var.set(str(config['vignetting_strength']))
        
        # Sensor defects
        if 'poisson_noise' in config:
            self.poisson_noise_var.set(config['poisson_noise'])
        if 'readout_noise' in config:
            self.readout_noise_var.set(config['readout_noise'])
        if 'gaussian_noise' in config:
            self.gaussian_noise_var.set(config['gaussian_noise'])
        if 'gaussian_sigma' in config:
            self.gaussian_sigma_var.set(str(config['gaussian_sigma']))
        
        # Post-processing defects
        if 'defocus_blur' in config:
            self.defocus_blur_var.set(config['defocus_blur'])
        if 'defocus_sigma' in config:
            self.defocus_sigma_var.set(str(config['defocus_sigma']))
        if 'partial_defocus' in config:
            self.partial_defocus_var.set(config['partial_defocus'])
        if 'adjust_brightness_contrast' in config:
            self.adjust_bc_var.set(config['adjust_brightness_contrast'])
            
        # Edge blur settings
        if 'edge_blur' in config:
            self.edge_blur_var.set(config['edge_blur'])
        if 'blur_kernel_size' in config:
            self.blur_kernel_size_var.set(str(config['blur_kernel_size']))
        if 'blur_type' in config:
            self.blur_type_var.set(config['blur_type'])