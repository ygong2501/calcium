"""
Parameter management module for calcium simulation.
"""
import json
import os
import numpy as np
from pathlib import Path


class SimulationParameters:
    """Manages parameters for calcium simulations."""
    
    DEFAULT_PARAMS = {
        'K_PLC': 0.2, 
        'K_5': 0.66, 
        'k_1': 1.11, 
        'k_a': 0.08,
        'k_p': 0.13, 
        'k_2': 0.0203, 
        'V_SERCA': 0.9, 
        'K_SERCA': 0.1,
        'c_tot': 2, 
        'beta': .185, 
        'k_i': 0.4, 
        'D_p': 0.005,
        'tau_max': 800, 
        'k_tau': 1.5, 
        'frac': 0.007680491551459293, 
        'D_c_ratio': 0.1
    }
    
    # Simulation type parameters
    SIM_TYPE_PARAMS = {
        "Single cell spikes": {
            "lower": 0.1,
            "upper": 0.5
        },
        "Intercellular transients": {
            "lower": 0.25,
            "upper": 0.6
        },
        "Intercellular waves": {
            "lower": 0.4,
            "upper": 0.8
        },
        "Fluttering": {
            "lower": 1.4,
            "upper": 1.5
        }
    }
    
    def __init__(self, sim_type=None, custom_params=None):
        """
        Initialize simulation parameters.
        
        Args:
            sim_type (str, optional): Type of simulation to run. 
                Options: "Single cell spikes", "Intercellular transients", 
                "Intercellular waves", "Fluttering"
            custom_params (dict, optional): Custom parameters to override defaults.
        """
        self.params = self.DEFAULT_PARAMS.copy()
        
        # Set simulation type parameters
        if sim_type is None:
            sim_type = "Intercellular waves"  # Default
        
        if sim_type in self.SIM_TYPE_PARAMS:
            self.params.update(self.SIM_TYPE_PARAMS[sim_type])
        else:
            # Default to Intercellular waves if type not recognized
            self.params.update(self.SIM_TYPE_PARAMS["Intercellular waves"])
        
        # Override with custom parameters if provided
        if custom_params:
            self.params.update(custom_params)
            
        self.sim_type = sim_type
    
    def save_to_file(self, filepath):
        """
        Save parameters to a JSON file.
        
        Args:
            filepath (str): Path to save the parameters.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump({
                'parameters': self.params,
                'simulation_type': self.sim_type
            }, f, indent=4)
    
    @classmethod
    def load_from_file(cls, filepath):
        """
        Load parameters from a JSON file.
        
        Args:
            filepath (str): Path to the parameter file.
            
        Returns:
            SimulationParameters: Instance with loaded parameters.
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        instance = cls(sim_type=data.get('simulation_type'))
        instance.params.update(data.get('parameters', {}))
        return instance
    
    def get_params_dict(self):
        """Get parameters as a dictionary."""
        return self.params.copy()
    
    def generate_random_params(self, seed=None):
        """
        Generate random parameter variations based on simulation type.
        
        Args:
            seed (int, optional): Random seed for reproducibility.
            
        Returns:
            dict: Dictionary of randomized parameters.
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Start with current parameters
        rand_params = self.params.copy()
        
        # Add random variations based on simulation type
        if self.sim_type == "Single cell spikes":
            rand_params['K_PLC'] = self.params['K_PLC'] * np.random.uniform(0.8, 1.2)
            rand_params['k_1'] = self.params['k_1'] * np.random.uniform(0.9, 1.1)
            
        elif self.sim_type == "Intercellular transients":
            rand_params['D_p'] = self.params['D_p'] * np.random.uniform(0.8, 1.2)
            rand_params['D_c_ratio'] = self.params['D_c_ratio'] * np.random.uniform(0.9, 1.1)
            
        elif self.sim_type == "Intercellular waves":
            rand_params['frac'] = self.params['frac'] * np.random.uniform(0.8, 1.2)
            rand_params['D_p'] = self.params['D_p'] * np.random.uniform(0.9, 1.1)
            
        elif self.sim_type == "Fluttering":
            rand_params['k_tau'] = self.params['k_tau'] * np.random.uniform(0.9, 1.1)
            rand_params['tau_max'] = self.params['tau_max'] * np.random.uniform(0.95, 1.05)
        
        # Randomize lower and upper bounds slightly for all types
        rand_params['lower'] = self.params['lower'] * np.random.uniform(0.95, 1.05)
        rand_params['upper'] = self.params['upper'] * np.random.uniform(0.95, 1.05)
        
        return rand_params