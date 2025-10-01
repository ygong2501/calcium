"""
Parameter management for calcium ion dynamics simulation.

This module provides the SimulationParameters class for managing all parameters
required for calcium signaling simulations, including:
- Biochemical reaction rate constants
- Diffusion coefficients
- Initial conditions
- Simulation type-specific parameters
"""
import json
import os
from typing import Dict, Optional, Any
from pathlib import Path

import numpy as np


class SimulationParameters:
    """
    Manages parameters for calcium ion dynamics simulations.

    The simulation models calcium signaling using a 4-variable ODE system per cell:
    1. Cytosolic calcium concentration (Ca²⁺)
    2. IP₃ (inositol 1,4,5-trisphosphate) concentration
    3. ER (endoplasmic reticulum) calcium store
    4. IP₃ receptor inactivation state

    Attributes:
        params (Dict): Current parameter values.
        sim_type (str): Simulation type (determines wave propagation characteristics).
    """

    # Default biochemical parameters
    DEFAULT_PARAMS = {
        # IP₃ production and degradation
        'K_PLC': 0.2,       # PLC activation threshold (µM)
        'K_5': 0.66,        # IP₃ degradation rate (1/s)

        # IP₃ receptor dynamics
        'k_1': 1.11,        # IP₃R channel opening rate (1/s)
        'k_a': 0.08,        # Ca²⁺ activation threshold for IP₃R (µM)
        'k_p': 0.13,        # IP₃ binding threshold for IP₃R (µM)
        'k_2': 0.0203,      # Leak flux rate (µM/s)

        # SERCA pump (ER Ca²⁺ uptake)
        'V_SERCA': 0.9,     # Maximum SERCA pump rate (µM/s)
        'K_SERCA': 0.1,     # SERCA half-saturation constant (µM)

        # Calcium buffering
        'c_tot': 2.0,       # Total calcium concentration (µM)
        'beta': 0.185,      # Cytosolic buffering ratio

        # IP₃ receptor inactivation
        'k_i': 0.4,         # IP₃R inactivation threshold (µM)
        'tau_max': 800.0,   # Maximum inactivation time constant (s)
        'k_tau': 1.5,       # Ca²⁺ threshold for inactivation (µM)

        # Diffusion
        'D_p': 0.005,       # IP₃ diffusion coefficient (µm²/s)
        'D_c_ratio': 0.1,   # Ca²⁺ diffusion as fraction of IP₃ diffusion

        # Stochastic activation
        'frac': 0.007680491551459293,  # Fraction of spontaneously active cells
    }

    # Simulation type-specific parameters
    # These control V_PLC distribution (PLC activation strength per cell)
    SIM_TYPE_PARAMS = {
        "Single cell spikes": {
            "lower": 0.1,    # Min V_PLC for single-cell activity
            "upper": 0.5,    # Max V_PLC (isolated spikes, no propagation)
            "description": "Isolated calcium spikes in individual cells"
        },
        "Intercellular transients": {
            "lower": 0.25,   # Min V_PLC for local wave propagation
            "upper": 0.6,    # Max V_PLC (short-range waves)
            "description": "Local calcium waves between adjacent cells"
        },
        "Intercellular waves": {
            "lower": 0.4,    # Min V_PLC for sustained wave propagation
            "upper": 0.8,    # Max V_PLC (long-range waves)
            "description": "Propagating calcium waves across tissue"
        },
        "Fluttering": {
            "lower": 1.4,    # Min V_PLC for oscillatory behavior
            "upper": 1.5,    # Max V_PLC (synchronized oscillations)
            "description": "Synchronized oscillatory calcium dynamics"
        }
    }

    def __init__(self, sim_type: Optional[str] = None, custom_params: Optional[Dict[str, float]] = None):
        """
        Initialize simulation parameters.

        Args:
            sim_type: Type of calcium dynamics to simulate.
                Options: "Single cell spikes", "Intercellular transients",
                         "Intercellular waves", "Fluttering"
                Default: "Intercellular waves"
            custom_params: Custom parameter values to override defaults.

        Example:
            >>> params = SimulationParameters("Intercellular waves")
            >>> params = SimulationParameters(custom_params={'D_p': 0.01})
        """
        # Initialize with default parameters
        self.params = self.DEFAULT_PARAMS.copy()

        # Set simulation type
        if sim_type is None:
            sim_type = "Intercellular waves"  # Default to most common type

        self.sim_type = sim_type

        # Apply simulation type-specific parameters
        if sim_type in self.SIM_TYPE_PARAMS:
            type_params = {k: v for k, v in self.SIM_TYPE_PARAMS[sim_type].items()
                          if k != 'description'}
            self.params.update(type_params)
        else:
            # Fallback to Intercellular waves if type not recognized
            print(f"Warning: Unknown simulation type '{sim_type}'. Using 'Intercellular waves'.")
            type_params = {k: v for k, v in self.SIM_TYPE_PARAMS["Intercellular waves"].items()
                          if k != 'description'}
            self.params.update(type_params)
            self.sim_type = "Intercellular waves"

        # Apply custom parameter overrides
        if custom_params:
            self.params.update(custom_params)

    def validate(self) -> bool:
        """
        Validate parameter values are physically meaningful.

        Returns:
            True if all parameters are valid.

        Raises:
            ValueError: If any parameter is invalid.
        """
        # Check for negative values where they don't make sense
        positive_params = ['K_PLC', 'K_5', 'k_1', 'k_a', 'k_p', 'k_2',
                          'V_SERCA', 'K_SERCA', 'c_tot', 'beta', 'k_i',
                          'tau_max', 'k_tau', 'D_p', 'D_c_ratio']

        for param in positive_params:
            if param in self.params and self.params[param] <= 0:
                raise ValueError(f"Parameter '{param}' must be positive, got {self.params[param]}")

        # Check fraction is between 0 and 1
        if 'frac' in self.params:
            if not 0 <= self.params['frac'] <= 1:
                raise ValueError(f"Parameter 'frac' must be in [0, 1], got {self.params['frac']}")

        # Check lower < upper
        if 'lower' in self.params and 'upper' in self.params:
            if self.params['lower'] >= self.params['upper']:
                raise ValueError(
                    f"'lower' ({self.params['lower']}) must be less than "
                    f"'upper' ({self.params['upper']})"
                )

        return True

    def save_to_file(self, filepath: str):
        """
        Save parameters to a JSON file.

        Args:
            filepath: Path to save the parameters (will create parent directories).
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'simulation_type': self.sim_type,
            'parameters': self.params,
            'description': self.SIM_TYPE_PARAMS.get(self.sim_type, {}).get('description', '')
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'SimulationParameters':
        """
        Load parameters from a JSON file.

        Args:
            filepath: Path to the parameter file.

        Returns:
            SimulationParameters instance with loaded parameters.

        Raises:
            FileNotFoundError: If file doesn't exist.
            json.JSONDecodeError: If file is not valid JSON.
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        sim_type = data.get('simulation_type', 'Intercellular waves')
        params = data.get('parameters', {})

        instance = cls(sim_type=sim_type)
        instance.params.update(params)
        instance.validate()  # Ensure loaded parameters are valid

        return instance

    def get_params_dict(self) -> Dict[str, float]:
        """
        Get parameters as a dictionary.

        Returns:
            Copy of current parameter dictionary.
        """
        return self.params.copy()

    def generate_random_params(self, seed: Optional[int] = None, variation: float = 0.2) -> Dict[str, float]:
        """
        Generate randomized parameter set for data augmentation.

        This creates parameter variations to increase dataset diversity while
        maintaining biological plausibility.

        Args:
            seed: Random seed for reproducibility.
            variation: Maximum fractional variation (default: 0.2 = ±20%).

        Returns:
            Dictionary of randomized parameters.

        Example:
            >>> params = SimulationParameters("Intercellular waves")
            >>> rand_params = params.generate_random_params(seed=42, variation=0.15)
        """
        if seed is not None:
            np.random.seed(seed)

        # Start with current parameters
        rand_params = self.params.copy()

        # Apply type-specific randomization
        if self.sim_type == "Single cell spikes":
            # Vary activation threshold and channel dynamics
            rand_params['K_PLC'] *= np.random.uniform(1-variation, 1+variation)
            rand_params['k_1'] *= np.random.uniform(1-variation/2, 1+variation/2)

        elif self.sim_type == "Intercellular transients":
            # Vary diffusion to affect wave propagation range
            rand_params['D_p'] *= np.random.uniform(1-variation, 1+variation)
            rand_params['D_c_ratio'] *= np.random.uniform(1-variation/2, 1+variation/2)

        elif self.sim_type == "Intercellular waves":
            # Vary initiation frequency and diffusion
            rand_params['frac'] = np.clip(
                rand_params['frac'] * np.random.uniform(1-variation, 1+variation),
                0.001, 0.03  # Keep within reasonable bounds
            )
            rand_params['D_p'] *= np.random.uniform(1-variation/2, 1+variation/2)

        elif self.sim_type == "Fluttering":
            # Vary oscillation time constants
            rand_params['k_tau'] *= np.random.uniform(1-variation/2, 1+variation/2)
            rand_params['tau_max'] *= np.random.uniform(1-variation/4, 1+variation/4)

        # Randomize V_PLC distribution bounds slightly for all types
        if 'lower' in rand_params:
            rand_params['lower'] *= np.random.uniform(1-variation/4, 1+variation/4)
        if 'upper' in rand_params:
            rand_params['upper'] *= np.random.uniform(1-variation/4, 1+variation/4)

        # Ensure lower < upper after randomization
        if 'lower' in rand_params and 'upper' in rand_params:
            if rand_params['lower'] >= rand_params['upper']:
                rand_params['lower'], rand_params['upper'] = \
                    rand_params['upper'] * 0.9, rand_params['lower'] * 1.1

        return rand_params

    @classmethod
    def get_available_sim_types(cls) -> list:
        """
        Get list of available simulation types.

        Returns:
            List of simulation type names.
        """
        return list(cls.SIM_TYPE_PARAMS.keys())

    def __repr__(self) -> str:
        """String representation of parameters."""
        return f"SimulationParameters(sim_type='{self.sim_type}', n_params={len(self.params)})"

    def __str__(self) -> str:
        """Detailed string representation."""
        desc = self.SIM_TYPE_PARAMS.get(self.sim_type, {}).get('description', '')
        lines = [
            f"Simulation Type: {self.sim_type}",
            f"Description: {desc}",
            f"Parameters ({len(self.params)}):"
        ]
        for key, value in sorted(self.params.items()):
            lines.append(f"  {key:15s} = {value}")
        return '\n'.join(lines)
