"""
Preview panel for the calcium simulation GUI.
"""
import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageTk


class PreviewPanel:
    """Panel for image preview."""
    
    def __init__(self, parent, main_window):
        """
        Initialize the preview panel.
        
        Args:
            parent (ttk.Frame): Parent frame.
            main_window (MainWindow): Main window reference.
        """
        self.parent = parent
        self.main_window = main_window
        
        # Create preview frame
        self.frame = ttk.Frame(parent)
        
        # Create heading
        ttk.Label(
            self.frame, text="Preview", font=("", 14, "bold")
        ).pack(side=tk.TOP, pady=10)
        
        # Create canvas for image preview
        self.canvas = tk.Canvas(self.frame, width=512, height=512, bg='black')
        self.canvas.pack(side=tk.TOP, padx=10, pady=10)
        
        # Create info panel
        self.info_frame = ttk.Frame(self.frame)
        self.info_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        # Image info
        ttk.Label(self.info_frame, text="Image Info:", font=("", 10, "bold")).grid(
            row=0, column=0, padx=5, pady=5, sticky=tk.W
        )
        
        self.image_size_var = tk.StringVar()
        self.image_size_var.set("Size: -")
        ttk.Label(self.info_frame, textvariable=self.image_size_var).grid(
            row=1, column=0, padx=20, pady=2, sticky=tk.W
        )
        
        # Cell info
        ttk.Label(self.info_frame, text="Cell Info:", font=("", 10, "bold")).grid(
            row=2, column=0, padx=5, pady=5, sticky=tk.W
        )
        
        self.active_cells_var = tk.StringVar()
        self.active_cells_var.set("Active Cells: -")
        ttk.Label(self.info_frame, textvariable=self.active_cells_var).grid(
            row=3, column=0, padx=20, pady=2, sticky=tk.W
        )
        
        self.total_cells_var = tk.StringVar()
        self.total_cells_var.set("Total Cells: -")
        ttk.Label(self.info_frame, textvariable=self.total_cells_var).grid(
            row=4, column=0, padx=20, pady=2, sticky=tk.W
        )
        
        # Image adjustments
        ttk.Label(self.info_frame, text="Adjustments:", font=("", 10, "bold")).grid(
            row=0, column=1, padx=5, pady=5, sticky=tk.W
        )
        
        ttk.Label(self.info_frame, text="Brightness:").grid(
            row=1, column=1, padx=20, pady=2, sticky=tk.W
        )
        self.brightness_scale = ttk.Scale(
            self.info_frame, from_=-50, to=50, value=0,
            command=self._on_brightness_change
        )
        self.brightness_scale.grid(row=1, column=2, padx=5, pady=2, sticky=tk.W)
        
        ttk.Label(self.info_frame, text="Contrast:").grid(
            row=2, column=1, padx=20, pady=2, sticky=tk.W
        )
        self.contrast_scale = ttk.Scale(
            self.info_frame, from_=0.5, to=2.0, value=1.0,
            command=self._on_contrast_change
        )
        self.contrast_scale.grid(row=2, column=2, padx=5, pady=2, sticky=tk.W)
        
        # Reset button
        self.reset_btn = ttk.Button(
            self.info_frame, text="Reset Adjustments", 
            command=self._reset_adjustments
        )
        self.reset_btn.grid(row=3, column=1, columnspan=2, padx=5, pady=5)
        
        # Store image data
        self.original_image = None
        self.current_image = None
        self.current_tk_image = None
        self.brightness_value = 0
        self.contrast_value = 1.0
    
    def set_image(self, image_data):
        """
        Set the preview image.
        
        Args:
            image_data (numpy.ndarray): Image data array.
        """
        if image_data is None:
            return
        
        # Store original image
        self.original_image = image_data.copy()
        self.current_image = image_data.copy()
        
        # Reset adjustments
        self._reset_adjustments()
        
        # Update image info
        self._update_image_info()
        
        # Display image
        self._display_image()
    
    def _display_image(self):
        """Display the current image on the canvas."""
        if self.current_image is None:
            return
        
        # Convert image to PIL Image
        if isinstance(self.current_image, np.ndarray):
            pil_image = Image.fromarray(self.current_image.astype(np.uint8))
        else:
            pil_image = self.current_image
        
        # Convert to Tkinter PhotoImage
        self.current_tk_image = ImageTk.PhotoImage(pil_image)
        
        # Clear canvas and display new image
        self.canvas.delete("all")
        self.canvas.create_image(
            256, 256, image=self.current_tk_image
        )
    
    def _update_image_info(self):
        """Update image information display."""
        if self.original_image is None:
            return
        
        # Update image size
        height, width = self.original_image.shape[:2]
        self.image_size_var.set(f"Size: {width}×{height}")
        
        # Update cell info if available
        if hasattr(self.main_window, 'current_pouch') and self.main_window.current_pouch is not None:
            pouch = self.main_window.current_pouch
            time_step = self.main_window.settings_panel.get_simulation_parameters().get('time_step', 0)
            
            if time_step >= pouch.T:
                time_step = pouch.T - 1
                
            active_cells = pouch.get_active_cells(time_step)
            total_cells = pouch.n_cells
            
            self.active_cells_var.set(f"Active Cells: {len(active_cells)}")
            self.total_cells_var.set(f"Total Cells: {total_cells}")
    
    def _on_brightness_change(self, value):
        """
        Handle brightness slider change.
        
        Args:
            value: Slider value.
        """
        try:
            self.brightness_value = float(value)
            self._apply_adjustments()
        except ValueError:
            pass
    
    def _on_contrast_change(self, value):
        """
        Handle contrast slider change.
        
        Args:
            value: Slider value.
        """
        try:
            self.contrast_value = float(value)
            self._apply_adjustments()
        except ValueError:
            pass
    
    def _apply_adjustments(self):
        """Apply brightness and contrast adjustments to the image."""
        if self.original_image is None:
            return
        
        # Apply adjustments
        adjusted = self._adjust_brightness_contrast(
            self.original_image,
            self.brightness_value / 100,  # Scale to -0.5 to 0.5
            self.contrast_value
        )
        
        self.current_image = adjusted
        self._display_image()
    
    def _adjust_brightness_contrast(self, image, brightness, contrast):
        """
        Adjust image brightness and contrast.
        
        Args:
            image (numpy.ndarray): Input image.
            brightness (float): Brightness adjustment [-0.5, 0.5].
            contrast (float): Contrast adjustment [0.5, 2.0].
        
        Returns:
            numpy.ndarray: Adjusted image.
        """
        # Convert to float for processing
        img_float = image.astype(np.float32)
        
        # Normalize to [0, 1] if needed
        if img_float.max() > 1.0:
            img_float = img_float / 255.0
        
        # Apply contrast adjustment
        img_float = (img_float - 0.5) * contrast + 0.5
        
        # Apply brightness adjustment
        img_float += brightness
        
        # Clip values to valid range
        img_float = np.clip(img_float, 0, 1.0)
        
        # Return to original range
        if image.max() > 1.0:
            img_float = img_float * 255.0
        
        return img_float.astype(np.uint8)
    
    def _reset_adjustments(self):
        """Reset brightness and contrast adjustments."""
        self.brightness_scale.set(0)
        self.contrast_scale.set(1.0)
        self.brightness_value = 0
        self.contrast_value = 1.0
        
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self._display_image()