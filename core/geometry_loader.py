"""
Geometry structure loading for calcium simulation.
"""
import os
import numpy as np
from pathlib import Path


class GeometryLoader:
    """Handles loading of geometry structures for simulations."""
    
    def __init__(self, geometry_dir=None):
        """
        Initialize the geometry loader.
        
        Args:
            geometry_dir (str, optional): Directory containing geometry files.
                If None, uses the default 'geometry' directory in the package.
        """
        if geometry_dir is None:
            # Use default location relative to this file
            package_dir = Path(__file__).parent.parent
            self.geometry_dir = os.path.join(package_dir, 'geometry')
        else:
            self.geometry_dir = geometry_dir
            
        # Check if geometry directory exists
        if not os.path.exists(self.geometry_dir):
            os.makedirs(self.geometry_dir, exist_ok=True)
            
        self.available_sizes = self._get_available_sizes()
    
    def _get_available_sizes(self):
        """
        Get list of available geometry sizes.
        
        Returns:
            list: List of available sizes.
        """
        # Check what geometry files are available
        sizes = []
        try:
            # Try to load the vertices file to extract sizes
            vertices_file = os.path.join(self.geometry_dir, 'disc_vertices.npy')
            if os.path.exists(vertices_file):
                vertices = np.load(vertices_file, allow_pickle=True).item()
                sizes = list(vertices.keys())
        except Exception as e:
            print(f"Warning: Could not load geometry sizes: {e}")
            # Default sizes if no files available
            sizes = ['xsmall', 'small', 'medium', 'large']
            
        return sizes
    
    def load_geometry(self, size='medium'):
        """
        Load geometry for a specific size.
        
        Args:
            size (str): Size of the geometry to load.
                Options: 'xsmall', 'small', 'medium', 'large'
                
        Returns:
            tuple: (vertices, adjacency_matrix, laplacian_matrix)
        """
        # Validate size
        if size not in self.available_sizes and os.path.exists(self.geometry_dir):
            available = ', '.join(self.available_sizes) if self.available_sizes else "none"
            raise ValueError(f"Size '{size}' not available. Available sizes: {available}")
        
        try:
            # Load vertices
            vertices_file = os.path.join(self.geometry_dir, 'disc_vertices.npy')
            vertices = np.load(vertices_file, allow_pickle=True).item()[size]
            
            # Load adjacency matrix
            adj_file = os.path.join(self.geometry_dir, 'disc_sizes_adj.npy')
            adjacency_matrix = np.load(adj_file, allow_pickle=True).item()[size]
            
            # Load laplacian matrix
            laplacian_file = os.path.join(self.geometry_dir, 'disc_sizes_laplacian.npy')
            laplacian_matrix = np.load(laplacian_file, allow_pickle=True).item()[size]
            
            return vertices, adjacency_matrix, laplacian_matrix
            
        except Exception as e:
            raise RuntimeError(f"Error loading geometry files: {e}")
    
    def extract_geometry_from_notebook(self, notebook_path, output_dir=None):
        """
        Extract geometry data from original notebook.
        
        This is a utility function to extract and save geometry data
        from the original notebook for Windows compatibility.
        
        Args:
            notebook_path (str): Path to the original notebook.
            output_dir (str, optional): Directory to save extracted data.
                If None, uses the default geometry directory.
                
        Returns:
            bool: True if extraction succeeded, False otherwise.
        """
        # Implementation would depend on how the data is stored in the notebook
        # This is a placeholder for the extraction logic
        raise NotImplementedError("Geometry extraction not implemented")
        
    def get_available_sizes(self):
        """
        Get list of available geometry sizes.
        
        Returns:
            list: List of available sizes.
        """
        return self.available_sizes.copy()