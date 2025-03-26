"""
Main window for the calcium simulation GUI.
"""
import os
import sys
import json
import threading
import warnings
import numpy as np

# Check if tkinter is available
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from PIL import Image, ImageTk
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    warnings.warn("tkinter not available. GUI functionality will be limited.", ImportWarning)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calcium_simulation.core.parameters import SimulationParameters
from calcium_simulation.core.pouch import Pouch
from calcium_simulation.utils.image_processing import apply_all_defects

# Only import GUI components if tkinter is available
if TKINTER_AVAILABLE:
    from .settings_panel import SettingsPanel
    from .preview_panel import PreviewPanel


class MainWindow:
    """Main window for the calcium simulation GUI."""
    
    def __init__(self, root):
        """
        Initialize the main window.
        
        Args:
            root (tk.Tk): Tkinter root window.
        """
        self.root = root
        self.root.title("Calcium Simulation System")
        self.root.geometry("1200x800")
        
        # Set icon (if available)
        icon_path = os.path.join(os.path.dirname(__file__), "../assets/icon.ico")
        if os.path.exists(icon_path):
            self.root.iconbitmap(icon_path)
        
        # Initialize simulation state
        self.simulation_running = False
        self.current_pouch = None
        self.current_image = None
        self.current_defect_config = {}
        
        # Create the main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create left panel (settings)
        self.settings_panel = SettingsPanel(self.main_frame, self)
        self.settings_panel.frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Create right panel (preview)
        self.preview_panel = PreviewPanel(self.main_frame, self)
        self.preview_panel.frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create bottom action bar
        self.action_bar = ttk.Frame(self.root)
        self.action_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=10)
        
        # Add action buttons
        self.generate_btn = ttk.Button(
            self.action_bar, text="Generate Preview", 
            command=self.generate_preview
        )
        self.generate_btn.pack(side=tk.LEFT, padx=5)
        
        self.batch_btn = ttk.Button(
            self.action_bar, text="Batch Generate", 
            command=self.batch_generate
        )
        self.batch_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_btn = ttk.Button(
            self.action_bar, text="Save Current Image", 
            command=self.save_current_image
        )
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_preset_btn = ttk.Button(
            self.action_bar, text="Save Preset", 
            command=self.save_preset
        )
        self.save_preset_btn.pack(side=tk.LEFT, padx=5)
        
        self.load_preset_btn = ttk.Button(
            self.action_bar, text="Load Preset", 
            command=self.load_preset
        )
        self.load_preset_btn.pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(
            self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W
        )
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=10)
        
        # Initialize with default values
        self.settings_panel.load_default_values()
    
    def generate_preview(self):
        """Generate a preview image with current settings."""
        if self.simulation_running:
            messagebox.showwarning("Warning", "Simulation already running")
            return
        
        # Update status
        self.status_var.set("Generating simulation...")
        self.simulation_running = True
        
        # Disable buttons
        self.generate_btn.config(state=tk.DISABLED)
        self.batch_btn.config(state=tk.DISABLED)
        
        # Get parameters from settings panel
        sim_params = self.settings_panel.get_simulation_parameters()
        defect_config = self.settings_panel.get_defect_configuration()
        
        # Start simulation in a separate thread
        threading.Thread(target=self._run_simulation, 
                        args=(sim_params, defect_config),
                        daemon=True).start()
    
    def _run_simulation(self, sim_params, defect_config):
        """
        Run the simulation in a background thread.
        
        Args:
            sim_params (dict): Simulation parameters.
            defect_config (dict): Defect configuration.
        """
        try:
            # Create parameter object
            params = SimulationParameters(
                sim_type=sim_params.get('sim_type', 'Intercellular waves')
            )
            
            # Create pouch with parameters
            pouch = Pouch(
                params=params.get_params_dict(),
                size=sim_params.get('pouch_size', 'small'),
                sim_number=np.random.randint(0, 10000),
                save=False,
                save_name='GUI_Preview',
                output_size=(512, 512)
            )
            
            # Run simulation
            pouch.simulate()
            
            # Store pouch for later use
            self.current_pouch = pouch
            
            # Generate image at specified time step
            time_step = int(sim_params.get('time_step', 0))
            if time_step >= pouch.T:
                time_step = pouch.T - 1
            
            # Generate clean image
            clean_image = pouch.generate_image(time_step, with_border=False)
            
            # Apply defects
            processed_image = apply_all_defects(
                clean_image, 
                pouch.get_cell_masks(), 
                defect_config
            )
            
            # Store defect config
            self.current_defect_config = defect_config
            
            # Update preview panel with image
            self.current_image = processed_image
            self.root.after(0, self._update_preview)
            
        except Exception as e:
            # Handle errors
            error_msg = f"Error in simulation: {str(e)}"
            self.root.after(0, lambda: self.status_var.set(error_msg))
            self.root.after(0, lambda: messagebox.showerror("Simulation Error", error_msg))
        
        # Enable buttons
        self.root.after(0, lambda: self.generate_btn.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.batch_btn.config(state=tk.NORMAL))
        
        # Update status
        self.root.after(0, lambda: self.status_var.set("Simulation complete"))
        self.simulation_running = False
    
    def _update_preview(self):
        """Update the preview panel with the current image."""
        if self.current_image is not None:
            self.preview_panel.set_image(self.current_image)
    
    def batch_generate(self):
        """Start batch generation process."""
        if self.simulation_running:
            messagebox.showwarning("Warning", "Simulation already running")
            return
        
        # Get batch parameters
        batch_params = self.settings_panel.get_batch_parameters()
        
        # Ask for output directory
        output_dir = filedialog.askdirectory(
            title="Select Output Directory for Batch Generation"
        )
        
        if not output_dir:
            return  # User cancelled
        
        # Update status
        self.status_var.set("Starting batch generation...")
        self.simulation_running = True
        
        # Disable buttons
        self.generate_btn.config(state=tk.DISABLED)
        self.batch_btn.config(state=tk.DISABLED)
        
        # Start batch generation in a separate thread
        threading.Thread(target=self._run_batch_generation, 
                        args=(batch_params, output_dir),
                        daemon=True).start()
    
    def _run_batch_generation(self, batch_params, output_dir):
        """
        Run batch generation in a background thread.
        
        Args:
            batch_params (dict): Batch generation parameters.
            output_dir (str): Output directory.
        """
        try:
            from calcium_simulation.main import generate_simulation_batch
            
            # Create a progress update function
            def progress_callback(current, total):
                self.root.after(0, lambda: self.status_var.set(
                    f"Generating simulation {current}/{total}..."
                ))
            
            # Get parameters
            num_simulations = int(batch_params.get('num_simulations', 5))
            
            # Run batch generation with progress callback
            results = generate_simulation_batch(
                num_simulations=num_simulations,
                output_dir=output_dir,
                pouch_sizes=batch_params.get('pouch_sizes', None),
                sim_types=batch_params.get('sim_types', None),
                time_steps=batch_params.get('time_steps', None),
                progress_callback=progress_callback
            )
            
            # Show success message
            stats = results.get('dataset_stats', {})
            train_count = stats.get('num_images', {}).get('train', 0)
            val_count = stats.get('num_images', {}).get('val', 0)
            test_count = stats.get('num_images', {}).get('test', 0)
            total_images = train_count + val_count + test_count
            
            success_msg = (
                f"Batch generation complete.\n"
                f"Generated {num_simulations} simulations with {total_images} total images.\n"
                f"Train: {train_count}, Validation: {val_count}, Test: {test_count}"
            )
            self.root.after(0, lambda: messagebox.showinfo("Batch Complete", success_msg))
            
        except Exception as e:
            # Handle errors
            error_msg = f"Error in batch generation: {str(e)}"
            self.root.after(0, lambda: self.status_var.set(error_msg))
            self.root.after(0, lambda: messagebox.showerror("Batch Error", error_msg))
        
        # Enable buttons
        self.root.after(0, lambda: self.generate_btn.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.batch_btn.config(state=tk.NORMAL))
        
        # Update status
        self.root.after(0, lambda: self.status_var.set("Batch generation complete"))
        self.simulation_running = False
    
    def save_current_image(self):
        """Save the current preview image."""
        if self.current_image is None:
            messagebox.showwarning("Warning", "No image to save")
            return
        
        # Ask for save path
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        
        if not file_path:
            return  # User cancelled
        
        try:
            # Convert to PIL Image and save
            if isinstance(self.current_image, np.ndarray):
                img = Image.fromarray(self.current_image)
                img.save(file_path)
                self.status_var.set(f"Image saved to {file_path}")
            else:
                messagebox.showerror("Error", "Invalid image format")
        except Exception as e:
            messagebox.showerror("Save Error", f"Error saving image: {str(e)}")
    
    def save_preset(self):
        """Save current settings as a preset."""
        # Get current settings
        sim_params = self.settings_panel.get_simulation_parameters()
        defect_config = self.settings_panel.get_defect_configuration()
        
        preset_data = {
            'simulation_parameters': sim_params,
            'defect_configuration': defect_config
        }
        
        # Ask for save path
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not file_path:
            return  # User cancelled
        
        try:
            with open(file_path, 'w') as f:
                json.dump(preset_data, f, indent=4)
            
            self.status_var.set(f"Preset saved to {file_path}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Error saving preset: {str(e)}")
    
    def load_preset(self):
        """Load settings from a preset file."""
        # Ask for preset file
        file_path = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not file_path:
            return  # User cancelled
        
        try:
            with open(file_path, 'r') as f:
                preset_data = json.load(f)
            
            # Apply settings
            sim_params = preset_data.get('simulation_parameters', {})
            defect_config = preset_data.get('defect_configuration', {})
            
            self.settings_panel.set_simulation_parameters(sim_params)
            self.settings_panel.set_defect_configuration(defect_config)
            
            self.status_var.set(f"Preset loaded from {file_path}")
        except Exception as e:
            messagebox.showerror("Load Error", f"Error loading preset: {str(e)}")


def launch_gui():
    """
    Launch the GUI application.
    
    Returns:
        bool: True if GUI was started successfully, False otherwise
    """
    if not TKINTER_AVAILABLE:
        print("ERROR: tkinter is not available. Cannot launch GUI.")
        print("Please install tkinter package appropriate for your Python version and platform.")
        return False
    
    try:
        root = tk.Tk()
        app = MainWindow(root)
        root.mainloop()
        return True
    except Exception as e:
        print(f"Error launching GUI: {str(e)}")
        return False


if __name__ == "__main__":
    launch_gui()