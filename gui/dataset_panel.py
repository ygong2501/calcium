"""
Dataset creation panel for the calcium simulation GUI.
This is a stub implementation to maintain backward compatibility.
The actual dataset creation functionality has been removed.
"""
import tkinter as tk
from tkinter import ttk


class DatasetPanel:
    """Panel for CSV mapping creation (stub implementation)."""
    
    def __init__(self, parent, main_window):
        """
        Initialize the dataset panel with default values only.
        
        Args:
            parent (tk.Frame): Parent frame.
            main_window (MainWindow): Main window reference.
        """
        self.parent = parent
        self.main_window = main_window
        
        # Create empty frame
        self.frame = ttk.Frame(parent)
        
        # Default split ratios for batch generation
        self.train_var = tk.DoubleVar(value=0.7)
        self.val_var = tk.DoubleVar(value=0.15)
        self.test_var = tk.DoubleVar(value=0.15)