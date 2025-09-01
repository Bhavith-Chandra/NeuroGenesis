"""
Unit tests for the grid_cells module.

Tests cover:
- GridCellEncoder basic functionality
- HierarchicalGridEncoder operations
- Position encoding and decoding
- Grid pattern generation
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.grid_cells import GridCellEncoder, HierarchicalGridEncoder


class TestGridCellEncoder(unittest.TestCase):
    """Test cases for GridCellEncoder class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.encoder = GridCellEncoder(
            grid_scales=[1.0, 1.4, 2.0],
            num_orientations=6,
            field_size=0.5,
            noise_level=0.01
        )
    
    def test_initialization(self):
        """Test encoder initialization."""
        self.assertEqual(len(self.encoder.grid_scales), 3)
        self.assertEqual(self.encoder.num_orientations, 6)
        self.assertEqual(self.encoder.field_size, 0.5)
        self.assertEqual(self.encoder.noise_level, 0.01)
        self.assertEqual(self.encoder.num_grid_cells, 18)  # 3 scales * 6 orientations
        self.assertEqual(len(self.encoder.grid_parameters), 18)
    
    def test_default_initialization(self):
        """Test encoder with default parameters."""
        default_encoder = GridCellEncoder()
        self.assertEqual(len(default_encoder.grid_scales), 6)
        self.assertEqual(default_encoder.num_orientations, 6)
        self.assertEqual(default_encoder.num_grid_cells, 36)
    
    def test_position_encoding(self):
        """Test position encoding functionality."""
        position = np.array([1.5, 2.0])
        activations = self.encoder.encode_position(position)
        
        # Check output shape
        self.assertEqual(len(activations), self.encoder.num_grid_cells)
        self.assertIsInstance(activations, np.ndarray)
        
        # Check activation values are reasonable
        for activation in activations:
            self.assertGreaterEqual(activation, 0.0)
            self.assertLessEqual(activation, 1.0)
    
    def test_position_encoding_wrong_shape(self):
        """Test position encoding with wrong input shape."""
        position = np.array([1.0])  # Wrong shape
        
        with self.assertRaises(ValueError):
            self.encoder.encode_position(position)
    
    def test_hexagonal_activation(self):
        """Test hexagonal activation computation."""
        # Test at grid point (should give high activation)
        proj1, proj2 = 0.0, 0.0
        scale = 1.0
        activation = self.encoder._hexagonal_activation(proj1, proj2, scale)
        
        self.assertGreater(activation, 0.0)
        self.assertLessEqual(activation, 1.0)
        
        # Test away from grid point (should give lower activation)
        proj1, proj2 = 1.0, 1.0
        activation2 = self.encoder._hexagonal_activation(proj1, proj2, scale)
        
        self.assertGreaterEqual(activation2, 0.0)
        self.assertLessEqual(activation2, 1.0)
    
    def test_position_decoding(self):
        """Test position decoding functionality."""
        # Create a test position
        original_position = np.array([1.0, -1.5])
        activations = self.encoder.encode_position(original_position)
        
        # Decode position
        estimated_position = self.encoder.decode_position(activations)
        
        # Check output shape
        self.assertEqual(estimated_position.shape, (2,))
        self.assertIsInstance(estimated_position, np.ndarray)
        
        # Check position is reasonable
        for coord in estimated_position:
            self.assertIsInstance(coord, (int, float, np.number))
    
    def test_position_decoding_wrong_size(self):
        """Test position decoding with wrong activation size."""
        activations = np.array([0.1, 0.2, 0.3])  # Wrong size
        
        with self.assertRaises(ValueError):
            self.encoder.decode_position(activations)
    
    def test_grid_pattern_generation(self):
        """Test grid pattern visualization generation."""
        X, Y, Z = self.encoder.get_grid_pattern(
            scale_idx=0,
            x_range=(-2, 2),
            y_range=(-2, 2),
            resolution=20
        )
        
        # Check output shapes
        self.assertEqual(X.shape, (20, 20))
        self.assertEqual(Y.shape, (20, 20))
        self.assertEqual(Z.shape, (20, 20))
        
        # Check Z values are reasonable
        self.assertTrue(np.all(Z >= 0))
        self.assertTrue(np.all(Z <= 1))
    
    def test_activation_statistics(self):
        """Test activation statistics computation."""
        stats = self.encoder.get_activation_statistics()
        
        required_keys = ['mean_activation', 'std_activation', 'max_activation', 
                        'min_activation', 'sparsity', 'num_grid_cells', 'grid_scales']
        
        for key in required_keys:
            self.assertIn(key, stats)
            self.assertIsNotNone(stats[key])
        
        # Check statistics are reasonable
        self.assertGreaterEqual(stats['mean_activation'], 0.0)
        self.assertLessEqual(stats['mean_activation'], 1.0)
        self.assertGreaterEqual(stats['sparsity'], 0.0)
        self.assertLessEqual(stats['sparsity'], 1.0)


class TestHierarchicalGridEncoder(unittest.TestCase):
    """Test cases for HierarchicalGridEncoder class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.hierarchical_encoder = HierarchicalGridEncoder(
            base_scale=0.5,
            scale_ratio=1.4,
            num_scales=4,
            num_orientations=6
        )
    
    def test_initialization(self):
        """Test hierarchical encoder initialization."""
        self.assertEqual(self.hierarchical_encoder.base_scale, 0.5)
        self.assertEqual(self.hierarchical_encoder.scale_ratio, 1.4)
        self.assertEqual(self.hierarchical_encoder.num_scales, 4)
        self.assertEqual(self.hierarchical_encoder.num_orientations, 6)
        self.assertEqual(len(self.hierarchical_encoder.grid_encoders), 4)
        
        # Check total cells calculation
        expected_total = 4 * 6  # num_scales * num_orientations
        self.assertEqual(self.hierarchical_encoder.total_cells, expected_total)
    
    def test_position_encoding(self):
        """Test hierarchical position encoding."""
        position = np.array([1.0, -1.5])
        activations = self.hierarchical_encoder.encode_position(position)
        
        # Check output shape
        self.assertEqual(len(activations), self.hierarchical_encoder.total_cells)
        self.assertIsInstance(activations, np.ndarray)
        
        # Check activation values are reasonable
        for activation in activations:
            self.assertGreaterEqual(activation, 0.0)
            self.assertLessEqual(activation, 1.0)
    
    def test_position_decoding(self):
        """Test hierarchical position decoding."""
        # Create a test position
        original_position = np.array([0.5, 1.0])
        activations = self.hierarchical_encoder.encode_position(original_position)
        
        # Decode position
        estimated_position = self.hierarchical_encoder.decode_position(activations)
        
        # Check output shape
        self.assertEqual(estimated_position.shape, (2,))
        self.assertIsInstance(estimated_position, np.ndarray)
        
        # Check position is reasonable
        for coord in estimated_position:
            self.assertIsInstance(coord, (int, float, np.number))
    
    def test_position_decoding_wrong_size(self):
        """Test hierarchical position decoding with wrong activation size."""
        activations = np.array([0.1, 0.2, 0.3])  # Wrong size
        
        with self.assertRaises(ValueError):
            self.hierarchical_encoder.decode_position(activations)


class TestIntegration(unittest.TestCase):
    """Integration tests for grid cell functionality."""
    
    def test_encoding_decoding_accuracy(self):
        """Test encoding and decoding accuracy."""
        encoder = GridCellEncoder(grid_scales=[1.0, 1.4])
        
        # Test multiple positions
        test_positions = [
            np.array([0.0, 0.0]),
            np.array([1.0, 1.0]),
            np.array([-1.0, 0.5]),
            np.array([2.0, -1.5])
        ]
        
        errors = []
        for position in test_positions:
            activations = encoder.encode_position(position)
            estimate = encoder.decode_position(activations)
            error = np.linalg.norm(position - estimate)
            errors.append(error)
        
        # Check errors are reasonable (should be small)
        mean_error = np.mean(errors)
        self.assertLess(mean_error, 1.0)  # Should be less than 1 unit
        
        # Check all errors are finite
        for error in errors:
            self.assertTrue(np.isfinite(error))
    
    def test_hierarchical_vs_single_scale(self):
        """Test hierarchical vs single scale encoding."""
        # Single scale encoder
        single_encoder = GridCellEncoder(grid_scales=[1.0])
        
        # Hierarchical encoder with same base scale
        hierarchical_encoder = HierarchicalGridEncoder(
            base_scale=1.0,
            num_scales=1
        )
        
        # Test position
        position = np.array([1.0, 1.0])
        
        # Encode with both
        single_activations = single_encoder.encode_position(position)
        hierarchical_activations = hierarchical_encoder.encode_position(position)
        
        # Should have same number of activations
        self.assertEqual(len(single_activations), len(hierarchical_activations))
        
        # Activations should be similar (not identical due to different implementations)
        correlation = np.corrcoef(single_activations, hierarchical_activations)[0, 1]
        self.assertGreater(correlation, 0.5)  # Should be reasonably correlated
    
    def test_grid_scale_progression(self):
        """Test geometric progression of grid scales."""
        hierarchical_encoder = HierarchicalGridEncoder(
            base_scale=1.0,
            scale_ratio=2.0,
            num_scales=4
        )
        
        # Check scale progression
        expected_scales = [1.0, 2.0, 4.0, 8.0]
        
        for i, encoder in enumerate(hierarchical_encoder.grid_encoders):
            actual_scale = encoder.grid_scales[0]  # Single scale per encoder
            self.assertAlmostEqual(actual_scale, expected_scales[i], places=6)
    
    def test_noise_effect(self):
        """Test effect of noise on encoding."""
        # Create encoders with different noise levels
        low_noise_encoder = GridCellEncoder(noise_level=0.001)
        high_noise_encoder = GridCellEncoder(noise_level=0.1)
        
        position = np.array([1.0, 1.0])
        
        # Encode multiple times
        low_noise_activations = []
        high_noise_activations = []
        
        for _ in range(10):
            low_noise_activations.append(low_noise_encoder.encode_position(position))
            high_noise_activations.append(high_noise_encoder.encode_position(position))
        
        # Convert to arrays
        low_noise_activations = np.array(low_noise_activations)
        high_noise_activations = np.array(high_noise_activations)
        
        # High noise should have higher variance
        low_noise_variance = np.var(low_noise_activations, axis=0)
        high_noise_variance = np.var(high_noise_activations, axis=0)
        
        self.assertGreater(np.mean(high_noise_variance), np.mean(low_noise_variance))


if __name__ == '__main__':
    unittest.main() 