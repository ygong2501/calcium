"""
Geometry structure loading for calcium simulation.

This module provides the GeometryLoader class for loading pre-computed
cell geometry data (vertices, adjacency matrices, Laplacian matrices) from
disk. The geometry files define the spatial structure of cell arrangements
for different pouch sizes.
"""
import os
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np


class GeometryLoader:
    """
    Handles loading of cell geometry structures for simulations.

    The geometry files contain pre-computed cell arrangements in different sizes:
    - disc_vertices.npy: Cell boundary vertices
    - disc_sizes_adj.npy: Cell adjacency matrices (which cells are neighbors)
    - disc_sizes_laplacian.npy: Graph Laplacian matrices for diffusion

    Attributes:
        geometry_dir (str): Path to directory containing geometry files.
        available_sizes (List[str]): List of available geometry sizes.
    """

    # Standard geometry file names
    VERTICES_FILE = 'disc_vertices.npy'
    ADJACENCY_FILE = 'disc_sizes_adj.npy'
    LAPLACIAN_FILE = 'disc_sizes_laplacian.npy'

    # Default sizes if files are not available
    DEFAULT_SIZES = ['xsmall', 'small', 'medium', 'large', 'xlarge']

    def __init__(self, geometry_dir: Optional[str] = None):
        """
        Initialize the geometry loader.

        Args:
            geometry_dir: Directory containing geometry files.
                If None, uses './geometry' relative to package root.
        """
        if geometry_dir is None:
            # Use default location relative to package root
            package_dir = Path(__file__).parent.parent
            self.geometry_dir = str(package_dir / 'geometry')
        else:
            self.geometry_dir = geometry_dir

        # Create geometry directory if it doesn't exist
        os.makedirs(self.geometry_dir, exist_ok=True)

        # Discover available sizes
        self.available_sizes = self._discover_available_sizes()

    def _discover_available_sizes(self) -> List[str]:
        """
        Discover available geometry sizes from disk.

        Returns:
            List of available size names.
        """
        vertices_path = os.path.join(self.geometry_dir, self.VERTICES_FILE)

        if not os.path.exists(vertices_path):
            return self.DEFAULT_SIZES.copy()

        try:
            # Load vertices file to extract available sizes
            vertices_dict = np.load(vertices_path, allow_pickle=True).item()
            return sorted(vertices_dict.keys())
        except Exception as e:
            print(f"Warning: Could not load geometry sizes from {vertices_path}: {e}")
            return self.DEFAULT_SIZES.copy()

    def load_geometry(self, size: str = 'medium') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load geometry data for a specific pouch size.

        Args:
            size: Geometry size to load (e.g., 'xsmall', 'small', 'medium', 'large', 'xlarge').

        Returns:
            Tuple of (vertices, adjacency_matrix, laplacian_matrix):
            - vertices: List of cell boundary vertices (n_cells,)
            - adjacency_matrix: Cell adjacency matrix (n_cells, n_cells)
            - laplacian_matrix: Graph Laplacian for diffusion (n_cells, n_cells)

        Raises:
            ValueError: If requested size is not available.
            FileNotFoundError: If geometry files are missing.
            RuntimeError: If geometry files are corrupted or invalid.
        """
        # Validate size
        if size not in self.available_sizes:
            available_str = ', '.join(self.available_sizes) if self.available_sizes else "none"
            raise ValueError(
                f"Geometry size '{size}' not available. "
                f"Available sizes: {available_str}. "
                f"Please download geometry files from the original repository."
            )

        # Construct file paths
        vertices_path = os.path.join(self.geometry_dir, self.VERTICES_FILE)
        adjacency_path = os.path.join(self.geometry_dir, self.ADJACENCY_FILE)
        laplacian_path = os.path.join(self.geometry_dir, self.LAPLACIAN_FILE)

        # Check files exist
        for path, name in [(vertices_path, 'vertices'),
                          (adjacency_path, 'adjacency'),
                          (laplacian_path, 'Laplacian')]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Geometry file not found: {path}. "
                    f"Please download geometry files from the original repository."
                )

        try:
            # Load all geometry components
            vertices_dict = np.load(vertices_path, allow_pickle=True).item()
            adjacency_dict = np.load(adjacency_path, allow_pickle=True).item()
            laplacian_dict = np.load(laplacian_path, allow_pickle=True).item()

            # Extract requested size
            vertices = vertices_dict[size]
            adjacency_matrix = adjacency_dict[size]
            laplacian_matrix = laplacian_dict[size]

            # Validate dimensions match
            n_cells = len(vertices)
            if adjacency_matrix.shape != (n_cells, n_cells):
                raise RuntimeError(
                    f"Adjacency matrix shape {adjacency_matrix.shape} "
                    f"does not match number of cells ({n_cells})"
                )
            if laplacian_matrix.shape != (n_cells, n_cells):
                raise RuntimeError(
                    f"Laplacian matrix shape {laplacian_matrix.shape} "
                    f"does not match number of cells ({n_cells})"
                )

            return vertices, adjacency_matrix, laplacian_matrix

        except KeyError:
            raise RuntimeError(
                f"Geometry size '{size}' exists in file list but could not be loaded. "
                f"Geometry files may be corrupted."
            )
        except Exception as e:
            raise RuntimeError(f"Error loading geometry files: {e}")

    def get_available_sizes(self) -> List[str]:
        """
        Get list of available geometry sizes.

        Returns:
            Copy of available sizes list.
        """
        return self.available_sizes.copy()

    def get_geometry_info(self, size: str) -> dict:
        """
        Get information about a specific geometry size without loading full data.

        Args:
            size: Geometry size to query.

        Returns:
            Dictionary containing geometry metadata.

        Raises:
            ValueError: If size is not available.
        """
        if size not in self.available_sizes:
            raise ValueError(f"Size '{size}' not available")

        try:
            vertices_path = os.path.join(self.geometry_dir, self.VERTICES_FILE)
            vertices_dict = np.load(vertices_path, allow_pickle=True).item()
            vertices = vertices_dict[size]

            return {
                'size': size,
                'n_cells': len(vertices),
                'geometry_dir': self.geometry_dir
            }
        except Exception as e:
            return {
                'size': size,
                'error': str(e)
            }
