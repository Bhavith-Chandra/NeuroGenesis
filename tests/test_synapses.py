"""
Unit tests for the synapses module.

Tests cover:
- MetaplasticSynapse basic functionality
- SynapticNetwork operations
- Learning and adaptation mechanisms
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.synapses import MetaplasticSynapse, SynapticNetwork


class TestMetaplasticSynapse(unittest.TestCase):
    """Test cases for MetaplasticSynapse class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.synapse = MetaplasticSynapse(
            initial_weight=0.1,
            learning_rate=0.01,
            metaplasticity_rate=0.001
        )
    
    def test_initialization(self):
        """Test synapse initialization."""
        self.assertEqual(self.synapse.weight, 0.1)
        self.assertEqual(self.synapse.base_learning_rate, 0.01)
        self.assertEqual(self.synapse.metaplasticity_rate, 0.001)
        self.assertEqual(self.synapse.activation_history, 0.0)
        self.assertEqual(self.synapse.metaplasticity_state, 0.0)
        self.assertEqual(self.synapse.total_activations, 0)
    
    def test_activation(self):
        """Test synapse activation."""
        input_signal = 1.0
        output = self.synapse.activate(input_signal)
        
        # Check output is weight * input
        expected_output = self.synapse.weight * input_signal
        self.assertAlmostEqual(output, expected_output, places=6)
        
        # Check activation history updated
        self.assertGreater(self.synapse.activation_history, 0)
        self.assertEqual(self.synapse.total_activations, 1)
    
    def test_weight_update(self):
        """Test weight update mechanism."""
        initial_weight = self.synapse.weight
        error_signal = 0.5
        input_signal = 1.0
        
        weight_change = self.synapse.update_weight(error_signal, input_signal)
        
        # Check weight changed
        self.assertNotEqual(self.synapse.weight, initial_weight)
        
        # Check weight change is reasonable
        self.assertIsInstance(weight_change, float)
        self.assertGreater(abs(weight_change), 0)
    
    def test_adaptive_learning_rate(self):
        """Test adaptive learning rate computation."""
        # Test initial learning rate
        initial_lr = self.synapse._compute_adaptive_learning_rate()
        self.assertAlmostEqual(initial_lr, self.synapse.base_learning_rate, places=6)
        
        # Activate synapse to change history
        self.synapse.activate(1.0)
        new_lr = self.synapse._compute_adaptive_learning_rate()
        
        # Learning rate should be different after activation
        self.assertNotEqual(new_lr, initial_lr)
    
    def test_metaplasticity_state_update(self):
        """Test metaplasticity state updates."""
        initial_state = self.synapse.metaplasticity_state
        
        # Update weight to trigger metaplasticity
        self.synapse.update_weight(0.5, 1.0)
        
        # Metaplasticity state should change
        self.assertNotEqual(self.synapse.metaplasticity_state, initial_state)
        
        # State should be bounded
        self.assertGreaterEqual(self.synapse.metaplasticity_state, -1.0)
        self.assertLessEqual(self.synapse.metaplasticity_state, 1.0)
    
    def test_get_state(self):
        """Test state retrieval."""
        state = self.synapse.get_state()
        
        required_keys = ['weight', 'activation_history', 'metaplasticity_state', 
                        'total_activations', 'current_learning_rate']
        
        for key in required_keys:
            self.assertIn(key, state)
            self.assertIsNotNone(state[key])
    
    def test_reset(self):
        """Test synapse reset functionality."""
        # Modify synapse state
        self.synapse.activate(1.0)
        self.synapse.update_weight(0.5, 1.0)
        
        # Reset
        self.synapse.reset()
        
        # Check reset to initial state
        self.assertEqual(self.synapse.weight, 0.1)
        self.assertEqual(self.synapse.activation_history, 0.0)
        self.assertEqual(self.synapse.metaplasticity_state, 0.0)
        self.assertEqual(self.synapse.total_activations, 0)


class TestSynapticNetwork(unittest.TestCase):
    """Test cases for SynapticNetwork class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.network = SynapticNetwork(
            num_synapses=5,
            initial_weight=0.0,
            learning_rate=0.01,
            metaplasticity_rate=0.001
        )
    
    def test_initialization(self):
        """Test network initialization."""
        self.assertEqual(self.network.num_synapses, 5)
        self.assertEqual(len(self.network.synapses), 5)
        
        # Check all synapses are properly initialized
        for synapse in self.network.synapses:
            self.assertIsInstance(synapse, MetaplasticSynapse)
            self.assertEqual(synapse.weight, 0.0)
    
    def test_forward_pass(self):
        """Test network forward pass."""
        input_signals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        output_signals = self.network.forward(input_signals)
        
        # Check output shape
        self.assertEqual(len(output_signals), 5)
        self.assertIsInstance(output_signals, np.ndarray)
        
        # Check outputs are reasonable
        for output in output_signals:
            self.assertIsInstance(output, (int, float, np.number))
    
    def test_forward_pass_wrong_size(self):
        """Test forward pass with wrong input size."""
        input_signals = np.array([1.0, 2.0, 3.0])  # Wrong size
        
        with self.assertRaises(ValueError):
            self.network.forward(input_signals)
    
    def test_weight_update(self):
        """Test network weight updates."""
        input_signals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        error_signals = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        weight_changes = self.network.update_weights(error_signals, input_signals)
        
        # Check weight changes shape
        self.assertEqual(len(weight_changes), 5)
        self.assertIsInstance(weight_changes, np.ndarray)
        
        # Check all synapses were updated
        for synapse in self.network.synapses:
            self.assertNotEqual(synapse.weight, 0.0)
    
    def test_weight_update_wrong_size(self):
        """Test weight update with wrong signal sizes."""
        input_signals = np.array([1.0, 2.0, 3.0])
        error_signals = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        with self.assertRaises(ValueError):
            self.network.update_weights(error_signals, input_signals)
    
    def test_get_network_state(self):
        """Test network state retrieval."""
        state = self.network.get_network_state()
        
        required_keys = ['synapse_states', 'average_weight', 'weight_std', 'total_activations']
        
        for key in required_keys:
            self.assertIn(key, state)
            self.assertIsNotNone(state[key])
        
        # Check synapse states
        self.assertEqual(len(state['synapse_states']), 5)
        
        # Check statistics
        self.assertIsInstance(state['average_weight'], float)
        self.assertIsInstance(state['weight_std'], float)
        self.assertIsInstance(state['total_activations'], int)


class TestIntegration(unittest.TestCase):
    """Integration tests for synapse and network functionality."""
    
    def test_learning_sequence(self):
        """Test complete learning sequence."""
        # Create network
        network = SynapticNetwork(num_synapses=3)
        
        # Training sequence
        for i in range(10):
            input_signals = np.random.normal(0, 1, 3)
            target_output = np.sum(input_signals) * 0.1
            
            # Forward pass
            actual_output = network.forward(input_signals)
            actual_sum = np.sum(actual_output)
            
            # Compute error
            error = target_output - actual_sum
            error_signals = np.full(3, error / 3)
            
            # Update weights
            network.update_weights(error_signals, input_signals)
        
        # Check network learned something
        state = network.get_network_state()
        self.assertNotEqual(state['average_weight'], 0.0)
        self.assertGreater(state['total_activations'], 0)
    
    def test_metaplasticity_evolution(self):
        """Test metaplasticity evolution over time."""
        synapse = MetaplasticSynapse()
        
        # Track metaplasticity state over time
        states = []
        for i in range(20):
            synapse.activate(1.0)
            synapse.update_weight(0.1, 1.0)
            states.append(synapse.metaplasticity_state)
        
        # Check metaplasticity state evolved
        self.assertNotEqual(states[0], states[-1])
        
        # Check state remains bounded
        for state in states:
            self.assertGreaterEqual(state, -1.0)
            self.assertLessEqual(state, 1.0)


if __name__ == '__main__':
    unittest.main() 