# NeuroGenesis 🧠

A cutting-edge neural architecture framework inspired by biological neural systems, featuring metaplastic synapses and grid cell encoders for advanced spatial representation and adaptive learning.

## 🌟 Features

- **Metaplastic Synapses**: Adaptive synaptic plasticity with metaplasticity mechanisms
- **Grid Cell Encoders**: Hexagonal grid patterns for robust spatial representation
- **Hierarchical Processing**: Multi-scale neural networks for complex pattern recognition
- **Adaptive Learning**: Dynamic learning rates based on activation history
- **Spatial Navigation**: Bio-inspired navigation systems using grid cell encoding

## 📁 Project Structure

```
NeuroGenesis/
├── modules/                 # Core neural modules
│   ├── synapses.py         # Metaplastic synapse implementation
│   └── grid_cells.py       # Grid cell encoder implementation
├── notebooks/              # Research and testing notebooks
│   └── 01_core_prototype.ipynb  # Interactive testing and visualization
├── tests/                  # Unit and integration tests
├── data/                   # Local datasets and experiment results
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Miniforge/conda or virtual environment
- Jupyter Notebook (for interactive exploration)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd NeuroGenesis
   ```

2. **Set up Python environment**:
   ```bash
   # Using conda
   conda create -n neurogenesis python=3.10
   conda activate neurogenesis
   
   # Or using venv
   python -m venv neurogenesis
   source neurogenesis/bin/activate  # On Windows: neurogenesis\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Start Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

5. **Open the interactive prototype**:
   Navigate to `notebooks/01_core_prototype.ipynb` and run all cells.

## 🧠 Core Modules

### Metaplastic Synapses (`modules/synapses.py`)

The `MetaplasticSynapse` class implements adaptive synaptic plasticity with metaplasticity mechanisms:

```python
from modules.synapses import MetaplasticSynapse

# Create a metaplastic synapse
synapse = MetaplasticSynapse(
    initial_weight=0.1,
    learning_rate=0.01,
    metaplasticity_rate=0.001
)

# Activate the synapse
output = synapse.activate(input_signal)

# Update weights with error signal
weight_change = synapse.update_weight(error_signal, input_signal)
```

**Key Features**:
- Adaptive learning rates based on activation history
- Metaplasticity state tracking
- Hebbian-like weight updates
- Comprehensive state monitoring

### Grid Cell Encoders (`modules/grid_cells.py`)

The `GridCellEncoder` class implements hexagonal grid patterns for spatial representation:

```python
from modules.grid_cells import GridCellEncoder

# Create a grid cell encoder
encoder = GridCellEncoder(
    grid_scales=[1.0, 1.4, 2.0],
    num_orientations=6
)

# Encode a 2D position
position = np.array([1.5, 2.0])
activations = encoder.encode_position(position)

# Decode back to position
estimated_position = encoder.decode_position(activations)
```

**Key Features**:
- Hexagonal grid patterns for optimal spatial coverage
- Multi-scale representation for hierarchical processing
- Position encoding and decoding capabilities
- Visualization tools for grid patterns

### Synaptic Networks

The `SynapticNetwork` class provides networks of metaplastic synapses:

```python
from modules.synapses import SynapticNetwork

# Create a network of synapses
network = SynapticNetwork(
    num_synapses=10,
    learning_rate=0.01,
    metaplasticity_rate=0.001
)

# Forward pass
input_signals = np.random.normal(0, 1, 10)
output_signals = network.forward(input_signals)

# Update weights
error_signals = target - output_signals
weight_changes = network.update_weights(error_signals, input_signals)
```

## 📊 Interactive Testing

The `notebooks/01_core_prototype.ipynb` provides comprehensive testing and visualization:

1. **Metaplastic Synapse Testing**: Learning curves, weight evolution, and metaplasticity analysis
2. **Grid Cell Visualization**: Hexagonal grid patterns and spatial encoding accuracy
3. **Network Training**: Multi-synapse network training and performance analysis
4. **Integration Testing**: Combined synapse and grid cell systems
5. **Performance Metrics**: Comprehensive statistics and visualization dashboard

## 🔬 Research Applications

### Spatial Navigation
- Bio-inspired navigation systems
- Path planning and trajectory optimization
- Spatial memory formation

### Pattern Recognition
- Adaptive feature learning
- Multi-scale pattern analysis
- Temporal sequence processing

### Neural Plasticity Studies
- Metaplasticity mechanisms
- Learning rate adaptation
- Synaptic strength dynamics

### Computational Neuroscience
- Biological neural system modeling
- Brain-inspired computing architectures
- Neuromorphic computing applications

## 🧪 Experimentation

### Custom Experiments

Create your own experiments by extending the base classes:

```python
# Custom metaplastic synapse
class CustomSynapse(MetaplasticSynapse):
    def __init__(self, custom_param=1.0, **kwargs):
        super().__init__(**kwargs)
        self.custom_param = custom_param
    
    def _compute_adaptive_learning_rate(self):
        # Custom learning rate computation
        base_lr = super()._compute_adaptive_learning_rate()
        return base_lr * self.custom_param

# Custom grid encoder
class CustomGridEncoder(GridCellEncoder):
    def __init__(self, custom_scale_factor=1.0, **kwargs):
        super().__init__(**kwargs)
        self.custom_scale_factor = custom_scale_factor
    
    def encode_position(self, position):
        # Custom encoding with scale factor
        scaled_position = position * self.custom_scale_factor
        return super().encode_position(scaled_position)
```

### Performance Testing

Run comprehensive performance tests:

```python
# Grid cell accuracy test
def test_grid_accuracy(encoder, num_tests=1000):
    errors = []
    for _ in range(num_tests):
        position = np.random.uniform(-5, 5, 2)
        activations = encoder.encode_position(position)
        estimate = encoder.decode_position(activations)
        error = np.linalg.norm(position - estimate)
        errors.append(error)
    return np.mean(errors), np.std(errors)

# Synaptic learning test
def test_synaptic_learning(synapse, num_trials=1000):
    for _ in range(num_trials):
        input_signal = np.random.normal(0, 1)
        target_output = 0.5 * input_signal
        actual_output = synapse.activate(input_signal)
        error = target_output - actual_output
        synapse.update_weight(error, input_signal)
    return synapse.weight
```

## 📈 Performance Benchmarks

### Grid Cell Encoding
- **Accuracy**: ~0.1-0.3 units average error
- **Sparsity**: ~60-80% low activation cells
- **Coverage**: Optimal hexagonal tiling

### Metaplastic Synapses
- **Convergence**: Stable within 50-100 epochs
- **Adaptation**: Responsive to activity patterns
- **Metaplasticity**: Bounded state evolution

### System Integration
- **End-to-end processing**: Functional pipeline
- **Scalability**: Modular architecture
- **Extensibility**: Easy to add new components

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**: Follow the existing code style
4. **Add tests**: Ensure all new functionality is tested
5. **Submit a pull request**: Describe your changes clearly

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8

# Run tests
pytest tests/

# Code formatting
black modules/ tests/

# Linting
flake8 modules/ tests/
```

## 📚 References

### Scientific Background
- [Metaplasticity in Neural Networks](https://doi.org/10.1016/j.tins.2008.09.006)
- [Grid Cells and Spatial Representation](https://doi.org/10.1038/nature03721)
- [Hippocampal Formation and Navigation](https://doi.org/10.1146/annurev-neuro-070815-013831)

### Technical Implementation
- [PyTorch Documentation](https://pytorch.org/docs/)
- [NumPy User Guide](https://numpy.org/doc/stable/user/)
- [Matplotlib Tutorial](https://matplotlib.org/stable/tutorials/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by biological neural systems and computational neuroscience research
- Built with modern Python scientific computing tools
- Developed for educational and research purposes

## 📞 Contact

For questions, suggestions, or collaborations:

- **Issues**: [GitHub Issues](https://github.com/your-username/NeuroGenesis/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/NeuroGenesis/discussions)
- **Email**: your-email@example.com

---

**NeuroGenesis** - Advancing the frontiers of neural architecture research 🧠✨ 