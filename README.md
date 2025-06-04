# EM43 - Emergent Model with 4 States

## Overview
EM43 is an implementation of an [emergent model (EM)](https://new.researchhub.com/fund/4130/emergent-models-a-general-modeling-framework-as-an-alternative-to-neural-networks) featuring 4 states and a neighborhood of 3 cells. This project provides both training and inference capabilities for EM models with the following characteristics:

- **Architecture**: 1-D EM
- **States**: 4 distinct states (0-3)
- **Neighborhood**: Radius-1 neighborhood (3 cells)
- **Boundary Conditions**: Open boundary conditions with 2-cell separator "BB"
- **Optimization**: Numba-accelerated parallel processing for efficient computation
- **Class-based Implementation**: All functionality encapsulated in the `EM43` class

## Project Structure
The project is now organized with a clear separation of concerns:
1. `em43_python/`: Contains the Python implementation
   - `em43_class.py`: Main EM43 class implementation
   - `config.yaml`: Configuration file for all parameters
   - `em43_demo.py`: Demo training and evaluation script
   - `em43_ga.py`: Genetic Algorithm implementation
   - `em43_utilis.py`: Utility functions
2. `em43.html`: Web-based visualization interface

## Important note
This is a **minimal working implementation** of the EM framework. Some advanced features described in the original paper such as **meta-learning**, **inductive biases**, and **state retention**—are **not yet implemented**.

## Installation

### Prerequisites
- Python 3.11 or higher 

### Setup
1. Clone the repository:
```bash
git clone [repository-url](https://github.com/BrightStarLabs/EM43.git)
cd EM43
```
2. Create a virtual environment:
```bash
python -m venv .venv --prompt em43
source .venv/bin/activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```

The project requires the following Python packages:
- numpy: For numerical computations
- numba: For just-in-time compilation and parallel processing
- tqdm: For progress bars during training
- matplotlib: For visualization of results
- pyyaml: For configuration management

## Usage

### Training
To train the model:
```bash
python em43_python/em43_demo.py
```

The training process can be configured through the `em43_python/config.yaml` file, which includes parameters for:
- Population size and generations
- Mutation rates
- Tournament selection parameters
- Elite fraction
- Regularization parameters
- Simulation parameters (window size, max steps, etc.)
- Checkpoint configuration

Or use command line arguments:
```bash
python em43_python/em43_demo.py --help
```

### Inference and Evaluation
To run inference and evaluate the trained model:
```bash
python em43_python/em43_demo.py
```

The script will:
1. Train the model for the specified number of generations
2. Save the best genome to `best_genome.pkl`
3. Load and evaluate the best genome
4. Generate visualizations:
   - `prediction_plot.png`: Expected vs Predicted outputs
   - `program_colors.png`: Program colormap visualization

### Python API
The model can be used programmatically through the `EM43` class:
```python
from em43_python.em43_class import EM43

# Create and load from checkpoint
em43 = EM43()
em43.load_genome()  # Loads from best_genome.pkl by default

# Run evaluation with optional plotting
outputs = em43.evaluate(plot=True)
```

The `evaluate()` method returns the model's outputs and can optionally display:
- Interactive plots of expected vs predicted outputs
- Program colormap visualization
- Detailed evaluation metrics including:
  - Stored fitness
  - Average error
  - Success rate (|err|<0.1)
  - Accuracy (exact matches)


## Features
- Parallel processing using Numba's `prange` for significant speed improvements
- Random-Immigrant Strategy for maintaining genetic diversity
- Detailed telemetry tracking including average Hamming distance
- Configurable parameters through YAML configuration
- Command-line interface with argument parsing
- Class-based API for programmatic use
- Checkpoint saving for training resumption

## Configuration
The main configuration file is `em43_python/config.yaml`, which contains all hyperparameters organized into sections:
- Population parameters
- Mutation rates
- Regularization
- Simulation parameters
- Checkpoint settings
- Input/output configuration

## Warranty
This software is provided "as is" without warranty of any kind, express or
implied, including but not limited to the warranties of merchantability,
fitness for a particular purpose and noninfringement. In no event shall the
authors or copyright holders be liable for any claim, damages or other
liability, whether in an action of contract, tort or otherwise, arising from,
out of or in connection with the software or the use or other dealings in the
software.
