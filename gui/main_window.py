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

from core.parameters import SimulationParameters
from core.pouch import Pouch
from utils.image_processing import apply_all_defects

# Only import GUI components if tkinter is available
if TKINTER_AVAILABLE:
    from .settings_panel import SettingsPanel
    from .preview_panel import PreviewPanel
    from .dataset_panel import DatasetPanel


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
        self.root.geometry("1400x850")  # Increased size to accommodate panels better
        
        # Set icon (if available)
        icon_path = os.path.join(os.path.dirname(__file__), "../assets/icon.ico")
        if os.path.exists(icon_path):
            self.root.iconbitmap(icon_path)
        
        # Initialize simulation state
        self.simulation_running = False
        self.current_pouch = None
        self.current_image = None
        self.current_defect_config = {}
        
        # Create main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create simulation tab
        self.simulation_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.simulation_tab, text="Simulation")
        
        # Create dataset tab
        self.dataset_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.dataset_tab, text="Dataset Creation")
        
        # Setup simulation tab with fixed width for left panel
        self.main_frame = ttk.Frame(self.simulation_tab)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a PanedWindow to allow user to adjust the split
        self.paned_window = ttk.PanedWindow(self.main_frame, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)
        
        # Left panel container with fixed width
        self.left_frame = ttk.Frame(self.paned_window, width=300)
        self.left_frame.pack_propagate(False)  # Prevent shrinking
        
        # Create left panel (settings)
        self.settings_panel = SettingsPanel(self.left_frame, self)
        self.settings_panel.frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Right panel for preview
        self.right_frame = ttk.Frame(self.paned_window)
        
        # Create right panel (preview)
        self.preview_panel = PreviewPanel(self.right_frame, self)
        self.preview_panel.frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add both frames to the paned window
        self.paned_window.add(self.left_frame, weight=1)
        self.paned_window.add(self.right_frame, weight=3)
        
        # Setup dataset tab
        self.dataset_panel = DatasetPanel(self.dataset_tab, self)
        self.dataset_panel.frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create bottom action bar for simulation tab
        self.action_bar = ttk.Frame(self.simulation_tab)
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
        
        self.generate_video_btn = ttk.Button(
            self.action_bar, text="Generate Video", 
            command=self.generate_video
        )
        self.generate_video_btn.pack(side=tk.LEFT, padx=5)
        
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
            
            # Get a simulation name based on type and random number
            sim_type_short = sim_params.get('sim_type', 'Waves').replace(' ', '_')
            sim_name = f"{sim_type_short}_{np.random.randint(0, 10000)}"
            
            # Create pouch with parameters
            pouch = Pouch(
                params=params.get_params_dict(),
                size=sim_params.get('pouch_size', 'small'),
                sim_number=np.random.randint(0, 10000),
                save=False,
                save_name=sim_name,
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
            
            # Generate clean image with edge blur if enabled
            edge_blur = defect_config.get('edge_blur', False)
            blur_kernel_size = defect_config.get('blur_kernel_size', 3)
            blur_type = defect_config.get('blur_type', 'mean')
            
            clean_image = pouch.generate_image(
                time_step, 
                with_border=False,
                edge_blur=edge_blur,
                blur_kernel_size=blur_kernel_size,
                blur_type=blur_type
            )
            
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
            from main import generate_simulation_batch
            
            # Create a progress update function
            def progress_callback(current, total):
                self.root.after(0, lambda: self.status_var.set(
                    f"Generating simulation {current}/{total}..."
                ))
            
            # Get parameters
            num_simulations = int(batch_params.get('num_simulations', 5))
            
            # Get edge blur settings
            edge_blur = batch_params.get('edge_blur', False)
            blur_kernel_size = int(batch_params.get('blur_kernel_size', 3))
            blur_type = batch_params.get('blur_type', 'mean')
            
            # Get dataset creation settings from the dataset panel
            # Use default values if dataset panel isn't available
            create_dataset = True  # Always create dataset
            
            try:
                # Access dataset panel values for split ratios if available
                train_ratio = self.dataset_panel.train_var.get()
                val_ratio = self.dataset_panel.val_var.get()
                test_ratio = self.dataset_panel.test_var.get()
            except (AttributeError, tk.TclError):
                # Use defaults if dataset panel not available or value retrieval fails
                train_ratio = 0.7
                val_ratio = 0.15
                test_ratio = 0.15
            
            # Get defect settings from the settings panel
            defect_config = self.settings_panel.get_defect_configuration()
            defect_configs = [defect_config] * num_simulations if defect_config else None
            
            # Run batch generation with progress callback
            results = generate_simulation_batch(
                num_simulations=num_simulations,
                output_dir=output_dir,
                pouch_sizes=batch_params.get('pouch_sizes', None),
                sim_types=batch_params.get('sim_types', None),
                time_steps=batch_params.get('time_steps', None),
                defect_configs=defect_configs,
                progress_callback=progress_callback,
                create_dataset=create_dataset,
                dataset_split_ratios=(train_ratio, val_ratio, test_ratio),
                edge_blur=edge_blur,
                blur_kernel_size=blur_kernel_size,
                blur_type=blur_type,
                memory_threshold=80  # Set memory threshold for garbage collection
            )
            
            # Build success message
            batch_dir = results.get('output_dir', output_dir)
            total_simulations = len(results.get('simulations', []))
            
            # Show dataset information since we always create dataset now
            stats = results.get('dataset_stats', {})
            train_count = stats.get('num_images', {}).get('train', 0)
            val_count = stats.get('num_images', {}).get('val', 0)
            test_count = stats.get('num_images', {}).get('test', 0)
            total_images = train_count + val_count + test_count
            
            success_msg = (
                f"Batch generation complete.\n"
                f"Generated {total_simulations} simulations with {total_images} total images.\n"
                f"Train: {train_count}, Validation: {val_count}, Test: {test_count}\n"
                f"Output directory: {batch_dir}"
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
            
    def generate_video(self):
        """Generate an animation from the current simulation."""
        if self.current_pouch is None:
            messagebox.showwarning("Warning", "No simulation to generate video from. Please generate a preview first.")
            return
        
        # Ask for save path
        file_path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
        )
        
        if not file_path:
            return  # User cancelled
        
        # Update status
        self.status_var.set("Generating video...")
        self.simulation_running = True
        
        # Disable buttons during video generation
        self.generate_btn.config(state=tk.DISABLED)
        self.batch_btn.config(state=tk.DISABLED)
        self.generate_video_btn.config(state=tk.DISABLED)
        
        # Start video generation in a separate thread
        threading.Thread(target=self._run_video_generation, 
                        args=(file_path,),
                        daemon=True).start()
    
    def _run_video_generation(self, output_path):
        """
        Run video generation in a background thread.
        
        Args:
            output_path (str): Path to save the video.
        """
        try:
            # Get parent directory
            output_dir = os.path.dirname(output_path)
            if not output_dir:
                output_dir = '.'
                
            # Check if output directory exists
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Generate animation using current pouch
            fps = 10  # frames per second
            skip_frames = 50  # skip frames to make video length reasonable
            
            self.root.after(0, lambda: self.status_var.set("Creating animation..."))
            
            # Generate and save animation
            anim = self.current_pouch.make_animation(path=output_dir, fps=fps, skip_frames=skip_frames)
            
            self.root.after(0, lambda: self.status_var.set("Video generation complete"))
            self.root.after(0, lambda: messagebox.showinfo("Video Complete", 
                                                 f"Video saved to {output_path}"))
            
        except Exception as e:
            # Handle errors
            error_msg = f"Error generating video: {str(e)}"
            self.root.after(0, lambda: self.status_var.set(error_msg))
            self.root.after(0, lambda: messagebox.showerror("Video Error", error_msg))
        
        # Enable buttons
        self.root.after(0, lambda: self.generate_btn.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.batch_btn.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.generate_video_btn.config(state=tk.NORMAL))
        
        # Update status
        self.simulation_running = False


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