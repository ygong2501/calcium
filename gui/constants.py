"""
GUI constants and configuration values.

This module centralizes all magic numbers, dimensions, and configuration
values used throughout the GUI to improve maintainability.
"""
from typing import Dict, List, Tuple

# ============================================================================
# Window Dimensions (pixels)
# ============================================================================
WINDOW_WIDTH: int = 1400
WINDOW_HEIGHT: int = 1000
WINDOW_MIN_WIDTH: int = 1200
WINDOW_MIN_HEIGHT: int = 800

# ============================================================================
# Panel Layout
# ============================================================================
SETTINGS_PANEL_WIDTH: int = 300
PREVIEW_PANEL_WEIGHT: int = 4  # Relative weight in paned window
SETTINGS_PANEL_WEIGHT: int = 1

# Padding
PANEL_PADDING: int = 10
FRAME_PADDING: int = 5
BUTTON_PADDING: int = 3

# ============================================================================
# GUI Update Intervals (milliseconds unless specified)
# ============================================================================
UPDATE_INTERVAL_MS: int = 50  # GUIUpdater batch update interval (50ms = 20 FPS)
GUI_UPDATE_INTERVAL_MS: int = 200  # Minimum time between GUI refreshes (ms)
GUI_UPDATE_INTERVAL_SEC: float = 0.2  # Same as above in seconds
RESOURCE_CHECK_INTERVAL_MS: int = 3000  # System resource monitoring interval
GUI_SCHEDULE_DELAY_MS: int = 10  # Delay for scheduling GUI updates

# ============================================================================
# Memory Management Thresholds (percentage)
# ============================================================================
MEMORY_CLEANUP_THRESHOLD: int = 85  # Auto trigger garbage collection
MEMORY_THRESHOLD_DEFAULT: int = 70  # Default for small/medium pouches
MEMORY_THRESHOLD_LARGE: int = 60  # Conservative threshold for large pouches
MEMORY_WARNING_THRESHOLD: int = 80  # Show warning to user
MEMORY_THRESHOLD_RETRY: int = 75  # Retry cleanup if still above this

# ============================================================================
# Image Quality Settings
# ============================================================================
DEFAULT_JPEG_QUALITY: int = 90
DEFAULT_IMAGE_SIZE: Tuple[int, int] = (512, 512)
DEFAULT_OUTPUT_SIZE: Tuple[int, int] = (512, 512)  # Alias for compatibility
DEFAULT_IMAGE_SIZE_STR: str = "512x512"

# ============================================================================
# Video Generation Settings
# ============================================================================
# CRF (Constant Rate Factor) - lower values = higher quality
VIDEO_QUALITY_HIGH: int = 18
VIDEO_QUALITY_MEDIUM: int = 23
VIDEO_QUALITY_LOW: int = 26
DEFAULT_VIDEO_QUALITY: int = VIDEO_QUALITY_MEDIUM

# Frame rates
VIDEO_FPS_OPTIONS: List[str] = ["5", "10", "12", "15", "24", "30"]
DEFAULT_VIDEO_FPS: int = 10

# Frame skip values
VIDEO_SKIP_FRAMES_OPTIONS: List[str] = ["10", "25", "50", "100", "200"]
DEFAULT_VIDEO_SKIP_FRAMES: int = 50

# Supported formats
VIDEO_FORMATS: List[str] = ["mp4", "avi", "mov"]
DEFAULT_VIDEO_FORMAT: str = "mp4"

# Video codec
VIDEO_CODEC_H264: str = "h264"
VIDEO_PIXEL_FORMAT: str = "yuv420p"

# ============================================================================
# File Paths
# ============================================================================
DEFAULT_OUTPUT_DIR: str = "output"
DEFAULT_PRESETS_DIR: str = "presets"
ICON_RELATIVE_PATH: str = "../assets/icon.ico"

# ============================================================================
# Batch Generation
# ============================================================================
DEFAULT_NUM_SIMULATIONS: int = 5
DEFAULT_NUM_BATCHES: int = 1
DEFAULT_SIMS_PER_BATCH: int = 10

# Progress tracking
BATCH_CLEANUP_DELAY_SEC: float = 0.5  # Delay between batch cleanup cycles

# ============================================================================
# Dialog Dimensions
# ============================================================================
VIDEO_DIALOG_WIDTH: int = 400
VIDEO_DIALOG_HEIGHT: int = 350

# ============================================================================
# Font Configurations
# ============================================================================
FONT_FAMILY: str = "Arial"
FONT_SIZE_NORMAL: int = 10
FONT_SIZE_HEADING: int = 16
FONT_WEIGHT_NORMAL: str = "normal"
FONT_WEIGHT_BOLD: str = "bold"

FONT_NORMAL: Tuple[str, int] = (FONT_FAMILY, FONT_SIZE_NORMAL)
FONT_BOLD: Tuple[str, int, str] = (FONT_FAMILY, FONT_SIZE_NORMAL, FONT_WEIGHT_BOLD)
FONT_HEADING: Tuple[str, int, str] = (FONT_FAMILY, FONT_SIZE_HEADING, FONT_WEIGHT_BOLD)

# ============================================================================
# Edge Blur Settings
# ============================================================================
DEFAULT_BLUR_KERNEL_SIZE: int = 3
BLUR_TYPES: List[str] = ["mean", "motion"]
DEFAULT_BLUR_TYPE: str = "mean"

# ============================================================================
# Simulation Types
# ============================================================================
SIMULATION_TYPES: List[str] = [
    "Single cell spikes",
    "Intercellular transients",
    "Intercellular waves",
    "Fluttering"
]
DEFAULT_SIMULATION_TYPE: str = "Intercellular waves"

# ============================================================================
# Pouch Sizes
# ============================================================================
POUCH_SIZES: List[str] = ["xsmall", "small", "medium", "large", "xlarge"]
DEFAULT_POUCH_SIZE: str = "small"

# ============================================================================
# Status Messages
# ============================================================================
STATUS_READY: str = "Ready"
STATUS_GENERATING: str = "Generating simulation..."
STATUS_COMPLETE: str = "Simulation complete"
STATUS_BATCH_STARTING: str = "Starting batch generation..."
STATUS_BATCH_COMPLETE: str = "Batch generation complete"
STATUS_BATCH_CANCELLED: str = "Batch generation cancelled"
STATUS_CREATING_CSV: str = "Creating CSV mapping..."
STATUS_GENERATING_VIDEO: str = "Generating video..."
STATUS_CLEANUP: str = "Performing memory cleanup..."

# ============================================================================
# Error Messages
# ============================================================================
ERROR_NO_IMAGE: str = "No image to save"
ERROR_SIMULATION_RUNNING: str = "Simulation already running"
ERROR_NO_SIMULATION: str = "No simulation to generate video from. Please generate a preview first."
ERROR_TKINTER_UNAVAILABLE: str = "tkinter is not available. Cannot launch GUI."

# ============================================================================
# Button Labels
# ============================================================================
BTN_GENERATE_PREVIEW: str = "Generate Preview"
BTN_BATCH_GENERATE: str = "Batch Generate"
BTN_SAVE_IMAGE: str = "Save Current Image"
BTN_SAVE_PRESET: str = "Save Preset"
BTN_LOAD_PRESET: str = "Load Preset"
BTN_CANCEL_BATCH: str = "Cancel Batch"
BTN_GENERATE_VIDEO: str = "Generate Video"
BTN_CLEAN_CACHE: str = "Clean Python Cache Files"

# ============================================================================
# Tab Labels
# ============================================================================
TAB_SIMULATION: str = "Simulation"
TAB_TOOLS: str = "Tools"

# ============================================================================
# File Extensions
# ============================================================================
IMAGE_EXTENSIONS: List[Tuple[str, str]] = [
    ("PNG files", "*.png"),
    ("JPEG files", "*.jpg"),
    ("All files", "*.*")
]

PRESET_EXTENSIONS: List[Tuple[str, str]] = [
    ("JSON files", "*.json"),
    ("All files", "*.*")
]

DEFAULT_IMAGE_EXT: str = ".png"
DEFAULT_PRESET_EXT: str = ".json"
