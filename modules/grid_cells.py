"""
Grid Cell Encoder Implementation for NeuroGenesis

This module implements a basic grid cell encoder for spatial representation
and navigation, inspired by hippocampal grid cells.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List, Optional
import math


class GridCellEncoder:
    """
    A basic grid cell encoder that creates hexagonal grid patterns for spatial representation.
    
    This implementation includes:
    - Multiple grid scales for hierarchical representation
    - Hexagonal grid patterns
    - Position encoding and decoding
    - Grid cell activation patterns
    """
    
    def __init__(self, 
                 grid_scales: List[float] = None,
                 num_orientations: int = 6,
                 field_size: float = 1.0,
                 noise_level: float = 0.01):
        """
        Initialize the grid cell encoder.
        
        Args:
            grid_scales: List of grid scales (spatial frequencies)
            num_orientations: Number of grid orientations
            field_size: Size of individual grid fields
            noise_level: Noise level for realistic activation patterns
        """
        if grid_scales is None:
            # Default grid scales following geometric progression
            grid_scales = [1.0, 1.4, 2.0, 2.8, 4.0, 5.7]
        
        self.grid_scales = grid_scales
        self.num_orientations = num_orientations
        self.field_size = field_size
        self.noise_level = noise_level
        
        # Generate grid orientations
        self.orientations = np.linspace(0, 2 * np.pi, num_orientations, endpoint=False)
        
        # Calculate total number of grid cells
        self.num_grid_cells = len(grid_scales) * num_orientations
        
        # Initialize grid cell parameters
        self._initialize_grid_parameters()
    
    def _initialize_grid_parameters(self):
        """Initialize grid cell parameters for each scale and orientation."""
        self.grid_parameters = []
        
        for scale in self.grid_scales:
            for orientation in self.orientations:
                # Create basis vectors for hexagonal grid
                basis1 = np.array([np.cos(orientation), np.sin(orientation)])
                basis2 = np.array([np.cos(orientation + np.pi/3), np.sin(orientation + np.pi/3)])
                
                # Scale the basis vectors
                basis1 *= scale
                basis2 *= scale
                
                self.grid_parameters.append({
                    'scale': scale,
                    'orientation': orientation,
                    'basis1': basis1,
                    'basis2': basis2,
                    'phase': np.random.uniform(0, 2*np.pi, 2)  # Random phase offset
                })
    
    def encode_position(self, position: np.ndarray) -> np.ndarray:
        """
        Encode a 2D position into grid cell activations.
        
        Args:
            position: 2D position [x, y]
            
        Returns:
            Array of grid cell activations
        """
        if position.shape != (2,):
            raise ValueError("Position must be a 2D array [x, y]")
        
        activations = np.zeros(self.num_grid_cells)
        
        for i, params in enumerate(self.grid_parameters):
            # Project position onto grid basis
            proj1 = np.dot(position, params['basis1']) + params['phase'][0]
            proj2 = np.dot(position, params['basis2']) + params['phase'][1]
            
            # Compute hexagonal grid activation
            activation = self._hexagonal_activation(proj1, proj2, params['scale'])
            
            # Add noise for realism
            activation += np.random.normal(0, self.noise_level)
            
            # Apply sigmoid activation function
            activation = 1.0 / (1.0 + np.exp(-activation))
            
            activations[i] = activation
        
        return activations
    
    def _hexagonal_activation(self, proj1: float, proj2: float, scale: float) -> float:
        """
        Compute hexagonal grid activation pattern.
        
        Args:
            proj1: Projection onto first basis vector
            proj2: Projection onto second basis vector
            scale: Grid scale
            
        Returns:
            Hexagonal activation value
        """
        # Convert to hexagonal coordinates
        hex_x = proj1 / scale
        hex_y = proj2 / scale
        
        # Compute hexagonal distance from grid points
        q = (2/3 * hex_x)
        r = (-1/3 * hex_x + np.sqrt(3)/3 * hex_y)
        s = -(q + r)
        
        # Round to nearest hexagon center
        q_round = round(q)
        r_round = round(r)
        s_round = round(s)
        
        q_diff = abs(q_round - q)
        r_diff = abs(r_round - r)
        s_diff = abs(s_round - s)
        
        if q_diff > r_diff and q_diff > s_diff:
            q_round = -(r_round + s_round)
        elif r_diff > s_diff:
            r_round = -(q_round + s_round)
        else:
            s_round = -(q_round + r_round)
        
        # Compute distance from nearest grid point
        distance = np.sqrt((q - q_round)**2 + (r - r_round)**2 + (s - s_round)**2)
        
        # Gaussian activation centered on grid points
        sigma = self.field_size / (2 * scale)
        activation = np.exp(-(distance**2) / (2 * sigma**2))
        
        return activation
    
    def decode_position(self, activations: np.ndarray, 
                       initial_guess: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Decode grid cell activations back to a position estimate.
        
        Args:
            activations: Array of grid cell activations
            initial_guess: Initial position guess for optimization
            
        Returns:
            Estimated 2D position
        """
        if len(activations) != self.num_grid_cells:
            raise ValueError(f"Expected {self.num_grid_cells} activations")
        
        if initial_guess is None:
            initial_guess = np.array([0.0, 0.0])
        
        # Simple gradient descent to find position that maximizes activation
        position = initial_guess.copy()
        learning_rate = 0.01
        max_iterations = 100
        
        for _ in range(max_iterations):
            # Compute gradient
            gradient = np.zeros(2)
            
            for i, params in enumerate(self.grid_parameters):
                # Compute activation gradient for this grid cell
                proj1 = np.dot(position, params['basis1']) + params['phase'][0]
                proj2 = np.dot(position, params['basis2']) + params['phase'][1]
                
                # Simplified gradient computation
                target_activation = activations[i]
                current_activation = self._hexagonal_activation(proj1, proj2, params['scale'])
                
                # Gradient contribution
                diff = target_activation - current_activation
                gradient += diff * (params['basis1'] + params['basis2'])
            
            # Update position
            position += learning_rate * gradient
            
            # Optional: add momentum or adaptive learning rate here
        
        return position
    
    def get_grid_pattern(self, scale_idx: int = 0, 
                        x_range: Tuple[float, float] = (-5, 5),
                        y_range: Tuple[float, float] = (-5, 5),
                        resolution: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a grid pattern visualization for a specific scale.
        
        Args:
            scale_idx: Index of the grid scale to visualize
            x_range: X-axis range for visualization
            y_range: Y-axis range for visualization
            resolution: Resolution of the visualization grid
            
        Returns:
            Tuple of (X, Y, Z) arrays for plotting
        """
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        
        Z = np.zeros_like(X)
        
        for i in range(resolution):
            for j in range(resolution):
                position = np.array([X[i, j], Y[i, j]])
                activations = self.encode_position(position)
                
                # Sum activations for the specified scale
                start_idx = scale_idx * self.num_orientations
                end_idx = start_idx + self.num_orientations
                Z[i, j] = np.sum(activations[start_idx:end_idx])
        
        return X, Y, Z
    
    def get_activation_statistics(self) -> dict:
        """
        Get statistics about grid cell activations.
        
        Returns:
            Dictionary containing activation statistics
        """
        # Test activations across a range of positions
        test_positions = []
        for x in np.linspace(-5, 5, 20):
            for y in np.linspace(-5, 5, 20):
                test_positions.append(np.array([x, y]))
        
        all_activations = []
        for pos in test_positions:
            activations = self.encode_position(pos)
            all_activations.append(activations)
        
        all_activations = np.array(all_activations)
        
        return {
            'mean_activation': np.mean(all_activations),
            'std_activation': np.std(all_activations),
            'max_activation': np.max(all_activations),
            'min_activation': np.min(all_activations),
            'sparsity': np.mean(all_activations < 0.1),  # Fraction of low activations
            'num_grid_cells': self.num_grid_cells,
            'grid_scales': self.grid_scales
        }


class HierarchicalGridEncoder:
    """
    A hierarchical grid encoder that combines multiple grid scales for robust spatial representation.
    """
    
    def __init__(self, 
                 base_scale: float = 1.0,
                 scale_ratio: float = 1.4,
                 num_scales: int = 6,
                 num_orientations: int = 6):
        """
        Initialize hierarchical grid encoder.
        
        Args:
            base_scale: Base grid scale
            scale_ratio: Ratio between consecutive scales
            num_scales: Number of grid scales
            num_orientations: Number of orientations per scale
        """
        self.base_scale = base_scale
        self.scale_ratio = scale_ratio
        self.num_scales = num_scales
        self.num_orientations = num_orientations
        
        # Generate geometric progression of scales
        grid_scales = [base_scale * (scale_ratio ** i) for i in range(num_scales)]
        
        # Create individual grid encoders for each scale
        self.grid_encoders = []
        for scale in grid_scales:
            encoder = GridCellEncoder(
                grid_scales=[scale],
                num_orientations=num_orientations
            )
            self.grid_encoders.append(encoder)
        
        self.total_cells = sum(encoder.num_grid_cells for encoder in self.grid_encoders)
    
    def encode_position(self, position: np.ndarray) -> np.ndarray:
        """
        Encode position using all grid scales.
        
        Args:
            position: 2D position [x, y]
            
        Returns:
            Concatenated activations from all scales
        """
        all_activations = []
        for encoder in self.grid_encoders:
            activations = encoder.encode_position(position)
            all_activations.extend(activations)
        
        return np.array(all_activations)
    
    def decode_position(self, activations: np.ndarray) -> np.ndarray:
        """
        Decode position using hierarchical information.
        
        Args:
            activations: Concatenated activations from all scales
            
        Returns:
            Estimated 2D position
        """
        if len(activations) != self.total_cells:
            raise ValueError(f"Expected {self.total_cells} activations")
        
        # Weighted combination of estimates from each scale
        position_estimates = []
        weights = []
        
        start_idx = 0
        for i, encoder in enumerate(self.grid_encoders):
            end_idx = start_idx + encoder.num_grid_cells
            scale_activations = activations[start_idx:end_idx]
            
            # Estimate position from this scale
            estimate = encoder.decode_position(scale_activations)
            position_estimates.append(estimate)
            
            # Weight by activation strength
            weight = np.mean(scale_activations)
            weights.append(weight)
            
            start_idx = end_idx
        
        # Weighted average of position estimates
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize
        
        final_position = np.zeros(2)
        for estimate, weight in zip(position_estimates, weights):
            final_position += weight * estimate
        
        return final_position 