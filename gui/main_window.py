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
    # Dataset panel import removed


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
        self.root.geometry("1400x1000")  # Increased height to ensure all elements are visible
        
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
        
        # Dataset tab removed
        
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
        
        # Add both frames to the paned window with better proportions
        self.paned_window.add(self.left_frame, weight=1)
        self.paned_window.add(self.right_frame, weight=4)  # Increased right panel weight for better visibility
        
        # Dataset panel removed
        
        # Create bottom action bar for simulation tab - more compact to save space
        self.action_bar = ttk.Frame(self.simulation_tab)
        self.action_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=5)
        
        # Create a more compact layout for action buttons using frames
        buttons_frame = ttk.Frame(self.action_bar)
        buttons_frame.pack(fill=tk.X, padx=5, pady=2)
        
        # Add action buttons in a more compact grid layout
        # Row 1
        self.generate_btn = ttk.Button(
            buttons_frame, text="Generate Preview", 
            command=self.generate_preview
        )
        self.generate_btn.grid(row=0, column=0, padx=3, pady=2, sticky=tk.W)
        
        self.batch_btn = ttk.Button(
            buttons_frame, text="Batch Generate", 
            command=self.batch_generate
        )
        self.batch_btn.grid(row=0, column=1, padx=3, pady=2, sticky=tk.W)
        
        self.save_btn = ttk.Button(
            buttons_frame, text="Save Current Image", 
            command=self.save_current_image
        )
        self.save_btn.grid(row=0, column=2, padx=3, pady=2, sticky=tk.W)
        
        # Row 2
        self.save_preset_btn = ttk.Button(
            buttons_frame, text="Save Preset", 
            command=self.save_preset
        )
        self.save_preset_btn.grid(row=1, column=0, padx=3, pady=2, sticky=tk.W)
        
        self.load_preset_btn = ttk.Button(
            buttons_frame, text="Load Preset", 
            command=self.load_preset
        )
        self.load_preset_btn.grid(row=1, column=1, padx=3, pady=2, sticky=tk.W)
        
        # Status bar removed from main window and moved to preview panel
        
        # Initialize with default values
        self.settings_panel.load_default_values()
    
    def generate_preview(self):
        """Generate a preview image with current settings."""
        if self.simulation_running:
            messagebox.showwarning("Warning", "Simulation already running")
            return
        
        # Update status in preview panel
        self.preview_panel.status_var.set("Generating simulation...")
        self.simulation_running = True
        
        # Disable buttons
        self.generate_btn.config(state=tk.DISABLED)
        self.batch_btn.config(state=tk.DISABLED)
        
        # Get parameters from settings panel
        sim_params = self.settings_panel.get_simulation_parameters()
        defect_config = self.settings_panel.get_defect_configuration()
        
        # Get batch parameters for image options
        batch_params = self.settings_panel.get_batch_parameters()
        
        # Start simulation in a separate thread
        threading.Thread(target=self._run_simulation, 
                        args=(sim_params, defect_config, batch_params),
                        daemon=True).start()
    
    def _run_simulation(self, sim_params, defect_config, batch_params=None):
        """
        Run the simulation in a background thread.
        
        Args:
            sim_params (dict): Simulation parameters.
            defect_config (dict): Defect configuration.
            batch_params (dict, optional): Batch parameters for image options.
        """
        try:
            # Create parameter object
            params = SimulationParameters(
                sim_type=sim_params.get('sim_type', 'Intercellular waves')
            )
            
            # Get a simulation name based on type and random number
            sim_type_short = sim_params.get('sim_type', 'Waves').replace(' ', '_')
            sim_name = f"{sim_type_short}_{np.random.randint(0, 10000)}"
            
            # Set output size from batch parameters if available
            output_size = (512, 512)  # Default size
            if batch_params:
                # Parse image size from string (e.g., "512x512" -> (512, 512))
                image_size_str = batch_params.get('image_size', '512x512')
                try:
                    width, height = map(int, image_size_str.split('x'))
                    output_size = (width, height)
                    print(f"Preview: Using custom image size: {width}x{height}")
                except (ValueError, AttributeError):
                    # Default to 512x512 if parsing fails
                    print(f"Preview: Failed to parse image size '{image_size_str}', using default 512x512")
                    pass
            
            # Create pouch with parameters, including JPEG quality from batch options
            jpeg_quality = batch_params.get('jpeg_quality', 90) if batch_params else 90
            
            pouch = Pouch(
                params=params.get_params_dict(),
                size=sim_params.get('pouch_size', 'small'),
                sim_number=np.random.randint(0, 10000),
                save=False,
                save_name=sim_name,
                output_size=output_size,
                jpeg_quality=jpeg_quality
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
            self.root.after(0, lambda: self.preview_panel.status_var.set(error_msg))
            self.root.after(0, lambda: messagebox.showerror("Simulation Error", error_msg))
        
        # Enable buttons
        self.root.after(0, lambda: self.generate_btn.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.batch_btn.config(state=tk.NORMAL))
        
        # Update status
        self.root.after(0, lambda: self.preview_panel.status_var.set("Simulation complete"))
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
        
        # Ask for output directory, defaulting to 'output'
        output_dir = filedialog.askdirectory(
            title="Select Output Directory for Batch Generation",
            initialdir="output"
        )
        
        if not output_dir:
            # Use default output directory if user cancels
            output_dir = "output"
            # Ensure directory exists
            os.makedirs(output_dir, exist_ok=True)
        
        # Update status
        self.preview_panel.status_var.set("Starting batch generation...")
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
            from utils.labeling import create_csv_mapping
            
            # Create a progress update function
            def progress_callback(current, total):
                self.root.after(0, lambda: self.preview_panel.status_var.set(
                    f"Generating simulation {current}/{total}..."
                ))
            
            # Get parameters
            num_simulations = int(batch_params.get('num_simulations', 5))
            
            # Get edge blur settings
            edge_blur = batch_params.get('edge_blur', False)
            blur_kernel_size = int(batch_params.get('blur_kernel_size', 3))
            blur_type = batch_params.get('blur_type', 'mean')
            
            # Get mask options
            generate_masks = batch_params.get('generate_masks', True)
            create_csv = batch_params.get('create_csv', False)
            
            # Get basic stats creation settings
            # Always create stats for visualization and monitoring
            create_stats = True  # Generate basic statistics
            
            # Default dataset split ratios - dataset panel removed
            train_ratio = 0.7
            val_ratio = 0.15
            test_ratio = 0.15
            
            # Get defect settings from the settings panel
            defect_config = self.settings_panel.get_defect_configuration()
            defect_configs = [defect_config] * num_simulations if defect_config else None
            
            # Parse image size from string (e.g., "512x512" -> (512, 512))
            image_size_str = batch_params.get('image_size', '512x512')
            try:
                width, height = map(int, image_size_str.split('x'))
                output_size = (width, height)
                print(f"Batch: Using custom image size: {width}x{height}")
            except (ValueError, AttributeError):
                # Default to 512x512 if parsing fails
                output_size = (512, 512)
                print(f"Batch: Failed to parse image size '{image_size_str}', using default 512x512")
                
            # Get JPEG quality setting
            jpeg_quality = batch_params.get('jpeg_quality', 90)
            
            # Run batch generation with progress callback
            results = generate_simulation_batch(
                num_simulations=num_simulations,
                output_dir=output_dir,
                pouch_sizes=batch_params.get('pouch_sizes', None),
                sim_types=batch_params.get('sim_types', None),
                time_steps=batch_params.get('time_steps', None),
                defect_configs=defect_configs,
                progress_callback=progress_callback,
                create_stats=create_stats,
                edge_blur=edge_blur,
                blur_kernel_size=blur_kernel_size,
                blur_type=blur_type,
                generate_masks=generate_masks,
                image_size=output_size,
                jpeg_quality=jpeg_quality,
                memory_threshold=80,  # Set memory threshold for garbage collection
                save_pouch=batch_params.get('enable_video', False)  # Save pouch object if video generation is enabled
            )
            
            # Create CSV mapping if requested
            csv_path = None
            if create_csv:
                self.root.after(0, lambda: self.preview_panel.status_var.set("Creating CSV file for image-mask mapping..."))
                
                # Collect all image and mask files from simulations
                all_images = []
                all_masks = []
                for sim_info in results.get('simulations', []):
                    # Add image files
                    img_dir = sim_info.get('image_dir', '')
                    if os.path.exists(img_dir):
                        img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')]
                        all_images.extend(img_files)
                    
                    # Add mask files if generate_masks was enabled
                    if generate_masks:
                        mask_dir = sim_info.get('mask_dir', '')
                        if os.path.exists(mask_dir):
                            mask_files = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if (f.endswith('.jpg') or f.endswith('.png')) and '_mask_' in f]
                            all_masks.extend(mask_files)
                
                # Create CSV mapping file in the batch directory
                if all_images and all_masks:
                    batch_dir = results.get('output_dir', output_dir)
                    csv_path = create_csv_mapping(all_images, all_masks, batch_dir)
                    self.root.after(0, lambda: self.preview_panel.status_var.set(f"CSV mapping created: {os.path.basename(csv_path)}"))
            
            # Process video generation if enabled
            if batch_params.get('enable_video', False) and results.get('simulations'):
                # 检查内存使用
                from main import monitor_memory
                memory_percent, memory_mb = monitor_memory()
                if memory_percent > 80:  # 高于80%时提醒用户
                    self.root.after(0, lambda: messagebox.showwarning(
                        "Memory Warning", 
                        f"Memory usage is high ({memory_percent:.1f}%). Video generation may be slow or fail."
                    ))
                    
                # Get the last simulation info
                last_sim = results.get('simulations', [])[-1]
                last_pouch = last_sim.get('pouch')
                
                if last_pouch:
                    self.root.after(0, lambda: self.preview_panel.status_var.set("Generating video from last simulation..."))
                    
                    # Configure video options
                    video_options = {
                        'format': batch_params.get('video_format', 'mp4'),
                        'fps': batch_params.get('video_fps', 10),
                        'skip_frames': batch_params.get('video_skip_frames', 0),
                        'quality': batch_params.get('video_quality', 23)
                    }
                    
                    # Generate the video filename
                    video_filename = f"simulation_video.{video_options['format']}"
                    video_path = os.path.join(output_dir, video_filename)
                    
                    try:
                        # Generate video using the last simulation's pouch
                        extra_args = ['-codec:v', 'h264', '-crf', str(video_options['quality']), '-pix_fmt', 'yuv420p']
                        last_pouch.make_animation(
                            path=output_dir,
                            fps=video_options['fps'],
                            skip_frames=video_options['skip_frames'],
                            filename=video_filename,
                            extra_args=extra_args
                        )
                        self.root.after(0, lambda: self.preview_panel.status_var.set("Video generation complete"))
                    except Exception as e:
                        error_msg = f"Error generating video: {str(e)}"
                        self.root.after(0, lambda: self.preview_panel.status_var.set(error_msg))
                        self.root.after(0, lambda: messagebox.showerror("Video Error", error_msg))
            
            # Build success message
            batch_dir = results.get('output_dir', output_dir)
            total_simulations = len(results.get('simulations', []))
            
            # Count actual images from simulation results
            total_images = 0
            for sim_info in results.get('simulations', []):
                total_images += sim_info.get('image_count', 0)
            
            # Count actual images from simulation results
            total_images = 0
            for sim_info in results.get('simulations', []):
                total_images += sim_info.get('image_count', 0)
            
            success_msg = (
                f"Batch generation complete.\n"
                f"Generated {total_simulations} simulations with {total_images} total images.\n"
            )
            
            if generate_masks:
                mask_count = sum(len([f for f in os.listdir(sim_info.get('mask_dir', '')) 
                               if os.path.isfile(os.path.join(sim_info.get('mask_dir', ''), f)) and '_mask_' in f])
                           for sim_info in results.get('simulations', []))
                success_msg += f"Generated {mask_count} mask files.\n"
            
            if csv_path:
                success_msg += f"CSV mapping file created at: {os.path.basename(csv_path)}\n"
                
            if batch_params.get('enable_video', False):
                video_filename = f"simulation_video.{batch_params.get('video_format', 'mp4')}"
                success_msg += f"Generated video: {video_filename}\n"
            
            success_msg += f"Output directory: {batch_dir}"
            
            self.root.after(0, lambda: messagebox.showinfo("Batch Complete", success_msg))
            
        except Exception as e:
            # Handle errors
            error_msg = f"Error in batch generation: {str(e)}"
            self.root.after(0, lambda: self.preview_panel.status_var.set(error_msg))
            self.root.after(0, lambda: messagebox.showerror("Batch Error", error_msg))
        
        # Enable buttons
        self.root.after(0, lambda: self.generate_btn.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.batch_btn.config(state=tk.NORMAL))
        
        # Update status
        self.root.after(0, lambda: self.preview_panel.status_var.set("Batch generation complete"))
        self.simulation_running = False
    
    def save_current_image(self):
        """Save the current preview image."""
        if self.current_image is None:
            messagebox.showwarning("Warning", "No image to save")
            return
        
        # Ask for save path, starting in the output directory
        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")],
            initialdir="output"
        )
        
        if not file_path:
            return  # User cancelled
        
        try:
            # Get batch parameters to retrieve quality setting
            batch_params = self.settings_panel.get_batch_parameters()
            jpeg_quality = batch_params.get('jpeg_quality', 90)
            
            # Save using utility function
            from utils.image_processing import save_image
            
            # Determine format from extension
            format = os.path.splitext(file_path)[1].lower().replace('.', '')
            if format not in ['jpg', 'jpeg', 'png']:
                format = 'jpg'  # Default to jpg
                
            # Save image
            output_dir = os.path.dirname(file_path)
            filename = os.path.basename(file_path)
            # Get the current output size from batch parameters
            batch_params = self.settings_panel.get_batch_parameters()
            image_size_str = batch_params.get('image_size', '512x512')
            try:
                width, height = map(int, image_size_str.split('x'))
                output_size = (width, height)
            except (ValueError, AttributeError):
                output_size = (512, 512)
                
            save_image(self.current_image, output_dir, filename, format=format, quality=jpeg_quality, target_size=output_size)
            
            self.preview_panel.status_var.set(f"Image saved to {file_path}")
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
        
        # Ask for save path, starting in the presets directory
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir="presets"
        )
        
        if not file_path:
            return  # User cancelled
        
        try:
            with open(file_path, 'w') as f:
                json.dump(preset_data, f, indent=4)
            
            self.preview_panel.status_var.set(f"Preset saved to {file_path}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Error saving preset: {str(e)}")
    
    def load_preset(self):
        """Load settings from a preset file."""
        # Ask for preset file, starting in the presets directory
        file_path = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir="presets"
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
            
            self.preview_panel.status_var.set(f"Preset loaded from {file_path}")
        except Exception as e:
            messagebox.showerror("Load Error", f"Error loading preset: {str(e)}")
            
    def generate_video(self):
        """Generate an animation from the current simulation."""
        if self.current_pouch is None:
            messagebox.showwarning("Warning", "No simulation to generate video from. Please generate a preview first.")
            return
        
        # Create a dialog for video settings
        self.video_settings_dialog = tk.Toplevel(self.root)
        self.video_settings_dialog.title("Video Generation Settings")
        self.video_settings_dialog.geometry("400x350")
        self.video_settings_dialog.transient(self.root)  # Make dialog modal
        self.video_settings_dialog.grab_set()
        
        # Center the dialog on the screen
        # First update to make sure dimensions are updated
        self.video_settings_dialog.update_idletasks()
        
        # Get main window position and dimensions
        root_x = self.root.winfo_rootx()
        root_y = self.root.winfo_rooty()
        root_width = self.root.winfo_width()
        root_height = self.root.winfo_height()
        
        # Get dialog dimensions
        dialog_width = self.video_settings_dialog.winfo_width()
        dialog_height = self.video_settings_dialog.winfo_height()
        
        # Calculate position for center of main window
        x = root_x + (root_width - dialog_width) // 2
        y = root_y + (root_height - dialog_height) // 2
        
        # Set position
        self.video_settings_dialog.geometry(f"+{x}+{y}")
        
        # Video format selection
        ttk.Label(self.video_settings_dialog, text="Video Format:", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=10, pady=(15, 5))
        self.video_format_var = tk.StringVar(value="mp4")
        format_frame = ttk.Frame(self.video_settings_dialog)
        format_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Radiobutton(format_frame, text="MP4", variable=self.video_format_var, value="mp4").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(format_frame, text="AVI", variable=self.video_format_var, value="avi").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(format_frame, text="MOV", variable=self.video_format_var, value="mov").pack(side=tk.LEFT, padx=5)
        
        # Frame rate settings
        ttk.Label(self.video_settings_dialog, text="Frame Rate (FPS):", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=10, pady=(15, 5))
        self.fps_var = tk.StringVar(value="10")
        fps_frame = ttk.Frame(self.video_settings_dialog)
        fps_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Combobox(fps_frame, textvariable=self.fps_var, values=["5", "10", "12", "15", "24", "30"], width=10).pack(side=tk.LEFT)
        
        # Frame skip settings
        ttk.Label(self.video_settings_dialog, text="Frame Skip (higher = shorter video):", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=10, pady=(15, 5))
        self.skip_frames_var = tk.StringVar(value="50")
        skip_frame = ttk.Frame(self.video_settings_dialog)
        skip_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Combobox(skip_frame, textvariable=self.skip_frames_var, values=["10", "25", "50", "100", "200"], width=10).pack(side=tk.LEFT)
        
        # Quality settings
        ttk.Label(self.video_settings_dialog, text="Quality:", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=10, pady=(15, 5))
        self.quality_var = tk.IntVar(value=23)  # Default CRF value
        quality_frame = ttk.Frame(self.video_settings_dialog)
        quality_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Quality radio buttons with descriptions
        ttk.Radiobutton(
            quality_frame, 
            text="High Quality (larger file)", 
            variable=self.quality_var, 
            value=18
        ).pack(anchor=tk.W, padx=5, pady=2)
        
        ttk.Radiobutton(
            quality_frame, 
            text="Medium Quality (default)", 
            variable=self.quality_var, 
            value=23
        ).pack(anchor=tk.W, padx=5, pady=2)
        
        ttk.Radiobutton(
            quality_frame, 
            text="Lower Quality (smaller file)", 
            variable=self.quality_var, 
            value=26
        ).pack(anchor=tk.W, padx=5, pady=2)
        
        # Button frame
        button_frame = ttk.Frame(self.video_settings_dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=(20, 10))
        
        # Cancel button
        ttk.Button(button_frame, text="Cancel", command=self.video_settings_dialog.destroy).pack(side=tk.LEFT, padx=5)
        
        # Generate button
        ttk.Button(button_frame, text="Generate Video", command=self._on_generate_video_confirmed).pack(side=tk.RIGHT, padx=5)
    
    def _on_generate_video_confirmed(self):
        """Handle video generation after settings are confirmed."""
        # Get video settings
        video_format = self.video_format_var.get()
        fps = int(self.fps_var.get())
        skip_frames = int(self.skip_frames_var.get())
        quality = self.quality_var.get()
        
        # Close the settings dialog
        self.video_settings_dialog.destroy()
        
        # Ask for save path, starting in the output directory
        file_path = filedialog.asksaveasfilename(
            defaultextension=f".{video_format}",
            filetypes=[(f"{video_format.upper()} files", f"*.{video_format}"), ("All files", "*.*")],
            initialdir="output"
        )
        
        if not file_path:
            return  # User cancelled
        
        # Update status
        self.preview_panel.status_var.set("Generating video...")
        self.simulation_running = True
        
        # Disable buttons during video generation
        self.generate_btn.config(state=tk.DISABLED)
        self.batch_btn.config(state=tk.DISABLED)
        
        # Start video generation in a separate thread
        video_options = {
            'format': video_format,
            'fps': fps,
            'skip_frames': skip_frames,
            'quality': quality
        }
        threading.Thread(target=self._run_video_generation, 
                        args=(file_path, video_options),
                        daemon=True).start()
    
    def _run_video_generation(self, output_path, video_options=None):
        """
        Run video generation in a background thread.
        
        Args:
            output_path (str): Path to save the video.
            video_options (dict): Dictionary of video generation options:
                - format (str): Video format (mp4, avi, mov)
                - fps (int): Frames per second
                - skip_frames (int): Number of frames to skip
                - quality (int): Quality setting (0-51, lower is better)
        """
        try:
            # Set default options if none provided
            if video_options is None:
                video_options = {
                    'format': 'mp4',
                    'fps': 10,
                    'skip_frames': 50,
                    'quality': 23
                }
                
            # Get parent directory
            output_dir = os.path.dirname(output_path)
            if not output_dir:
                output_dir = '.'
                
            # Check if output directory exists
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Extract video settings
            fps = video_options.get('fps', 10)
            skip_frames = video_options.get('skip_frames', 50)
            quality = video_options.get('quality', 23)
            
            self.root.after(0, lambda: self.preview_panel.status_var.set("Creating animation..."))
            
            # Configure FFmpeg codec based on format
            codec = 'h264'  # Default to h264 for all formats
            
            # Set up extra FFmpeg arguments based on quality
            extra_args = ['-codec:v', codec]
            
            # Add quality settings (CRF - Constant Rate Factor)
            # Lower CRF = higher quality, higher file size
            extra_args.extend(['-crf', str(quality)])
            
            # Set pixel format for better compatibility
            extra_args.extend(['-pix_fmt', 'yuv420p'])
            
            # Generate and save animation with the specified options
            anim = self.current_pouch.make_animation(
                path=output_dir, 
                fps=fps, 
                skip_frames=skip_frames,
                filename=os.path.basename(output_path),
                extra_args=extra_args
            )
            
            self.root.after(0, lambda: self.preview_panel.status_var.set("Video generation complete"))
            self.root.after(0, lambda: messagebox.showinfo("Video Complete", 
                                                 f"Video saved to {output_path}"))
            
        except Exception as e:
            # Handle errors
            error_msg = f"Error generating video: {str(e)}"
            self.root.after(0, lambda: self.preview_panel.status_var.set(error_msg))
            self.root.after(0, lambda: messagebox.showerror("Video Error", error_msg))
        
        # Enable buttons
        self.root.after(0, lambda: self.generate_btn.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.batch_btn.config(state=tk.NORMAL))
        
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