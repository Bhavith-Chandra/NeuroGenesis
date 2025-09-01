"""
Metaplastic Synapse Implementation for NeuroGenesis

This module implements a metaplastic synapse that can adapt its learning rate
based on its activation history and current state.
"""

import numpy as np
from typing import Optional, Tuple


class MetaplasticSynapse:
    """
    A metaplastic synapse that adapts its learning rate based on activation history.
    
    This implementation includes:
    - Weight storage and updates
    - Metaplasticity parameters
    - Activation history tracking
    - Adaptive learning rate computation
    """
    
    def __init__(self, 
                 initial_weight: float = 0.0,
                 learning_rate: float = 0.01,
                 metaplasticity_rate: float = 0.001,
                 activation_threshold: float = 0.5,
                 decay_rate: float = 0.95):
        """
        Initialize a metaplastic synapse.
        
        Args:
            initial_weight: Starting synaptic weight
            learning_rate: Base learning rate for weight updates
            metaplasticity_rate: Rate of metaplasticity adaptation
            activation_threshold: Threshold for considering synapse active
            decay_rate: Decay rate for activation history
        """
        self.weight = initial_weight
        self.base_learning_rate = learning_rate
        self.metaplasticity_rate = metaplasticity_rate
        self.activation_threshold = activation_threshold
        self.decay_rate = decay_rate
        
        # Metaplasticity state
        self.activation_history = 0.0
        self.metaplasticity_state = 0.0
        
        # Statistics tracking
        self.total_activations = 0
        self.weight_history = [initial_weight]
        self.learning_rate_history = [learning_rate]
    
    def activate(self, input_signal: float) -> float:
        """
        Activate the synapse with an input signal.
        
        Args:
            input_signal: Input signal strength
            
        Returns:
            Output signal (weight * input_signal)
        """
        output = self.weight * input_signal
        
        # Update activation history
        if abs(input_signal) > self.activation_threshold:
            self.activation_history = (self.decay_rate * self.activation_history + 
                                     (1 - self.decay_rate) * abs(input_signal))
            self.total_activations += 1
        
        return output
    
    def update_weight(self, error_signal: float, input_signal: float) -> float:
        """
        Update the synaptic weight based on error signal and input.
        
        Args:
            error_signal: Error signal for learning
            input_signal: Input signal that caused the error
            
        Returns:
            The weight change applied
        """
        # Compute adaptive learning rate based on metaplasticity
        adaptive_lr = self._compute_adaptive_learning_rate()
        
        # Standard Hebbian-like weight update
        weight_change = adaptive_lr * error_signal * input_signal
        
        # Apply weight update
        self.weight += weight_change
        
        # Update metaplasticity state
        self._update_metaplasticity_state(error_signal, input_signal)
        
        # Track history
        self.weight_history.append(self.weight)
        self.learning_rate_history.append(adaptive_lr)
        
        return weight_change
    
    def _compute_adaptive_learning_rate(self) -> float:
        """
        Compute adaptive learning rate based on metaplasticity state.
        
        Returns:
            Adaptive learning rate
        """
        # Metaplasticity modulates learning rate
        metaplasticity_factor = 1.0 + self.metaplasticity_state
        
        # Activation history also influences learning rate
        history_factor = 1.0 / (1.0 + self.activation_history)
        
        return self.base_learning_rate * metaplasticity_factor * history_factor
    
    def _update_metaplasticity_state(self, error_signal: float, input_signal: float):
        """
        Update the metaplasticity state based on recent activity.
        
        Args:
            error_signal: Current error signal
            input_signal: Current input signal
        """
        # Metaplasticity adapts based on error magnitude and input strength
        error_magnitude = abs(error_signal)
        input_magnitude = abs(input_signal)
        
        # Update metaplasticity state
        self.metaplasticity_state += (self.metaplasticity_rate * 
                                     error_magnitude * input_magnitude * 
                                     np.sign(error_signal * input_signal))
        
        # Clamp metaplasticity state to reasonable bounds
        self.metaplasticity_state = np.clip(self.metaplasticity_state, -1.0, 1.0)
    
    def get_state(self) -> dict:
        """
        Get current synapse state for monitoring.
        
        Returns:
            Dictionary containing current state information
        """
        return {
            'weight': self.weight,
            'activation_history': self.activation_history,
            'metaplasticity_state': self.metaplasticity_state,
            'total_activations': self.total_activations,
            'current_learning_rate': self._compute_adaptive_learning_rate()
        }
    
    def reset(self, new_weight: Optional[float] = None):
        """
        Reset the synapse to initial state.
        
        Args:
            new_weight: Optional new weight value (uses initial_weight if None)
        """
        if new_weight is not None:
            self.weight = new_weight
        else:
            self.weight = self.weight_history[0]
        
        self.activation_history = 0.0
        self.metaplasticity_state = 0.0
        self.total_activations = 0
        self.weight_history = [self.weight]
        self.learning_rate_history = [self.base_learning_rate]


class SynapticNetwork:
    """
    A network of metaplastic synapses for more complex learning scenarios.
    """
    
    def __init__(self, num_synapses: int, **synapse_kwargs):
        """
        Initialize a network of metaplastic synapses.
        
        Args:
            num_synapses: Number of synapses in the network
            **synapse_kwargs: Arguments passed to MetaplasticSynapse constructor
        """
        self.synapses = [MetaplasticSynapse(**synapse_kwargs) for _ in range(num_synapses)]
        self.num_synapses = num_synapses
    
    def forward(self, input_signals: np.ndarray) -> np.ndarray:
        """
        Forward pass through the synaptic network.
        
        Args:
            input_signals: Array of input signals
            
        Returns:
            Array of output signals
        """
        if len(input_signals) != self.num_synapses:
            raise ValueError(f"Expected {self.num_synapses} input signals, got {len(input_signals)}")
        
        return np.array([synapse.activate(signal) for synapse, signal in zip(self.synapses, input_signals)])
    
    def update_weights(self, error_signals: np.ndarray, input_signals: np.ndarray) -> np.ndarray:
        """
        Update all synaptic weights.
        
        Args:
            error_signals: Array of error signals
            input_signals: Array of input signals
            
        Returns:
            Array of weight changes
        """
        if len(error_signals) != self.num_synapses or len(input_signals) != self.num_synapses:
            raise ValueError("Error and input signal arrays must match number of synapses")
        
        return np.array([synapse.update_weight(error, input_signal) 
                        for synapse, error, input_signal in zip(self.synapses, error_signals, input_signals)])
    
    def get_network_state(self) -> dict:
        """
        Get state of all synapses in the network.
        
        Returns:
            Dictionary containing network state information
        """
        return {
            'synapse_states': [synapse.get_state() for synapse in self.synapses],
            'average_weight': np.mean([s.weight for s in self.synapses]),
            'weight_std': np.std([s.weight for s in self.synapses]),
            'total_activations': sum(s.total_activations for s in self.synapses)
        } 