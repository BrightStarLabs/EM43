# EM43 - Emergent Model with 4 States

## Overview
EM43 is an implementation of an [emergent model (EM)](https://new.researchhub.com/fund/4130/emergent-models-a-general-modeling-framework-as-an-alternative-to-neural-networks) featuring 4 states and a neighborhood of 3 cells.

## Features
- Parallel processing using Numba's `prange` for significant speed improvements
- Random-Immigrant Strategy for maintaining genetic diversity
- Detailed telemetry tracking including average Hamming distance
- Configurable parameters through YAML configuration
- Command-line interface with argument parsing
- Class-based API for programmatic use
- Checkpoint saving for training resumption

## Installation
### Prerequisites
- Python 3.11 or higher 

### Setup
1. Clone the repository:
```bash
git clone https://github.com/BrightStarLabs/EM43.git
cd EM43
```
2. Create a virtual environment:
```bash
uv venv .venv --prompt em43 
# or  
python -m venv .venv --prompt em43
source .venv/bin/activate
```
3. Install dependencies:
```bash
uv pip install -r requirements.txt
# or 
pip install -r requirements.txt
```

## Demo Usage

The EM-4/3 demo provides three main stages of operation:

### 1. Full Training Mode
```bash
python em43_python/em43_demo.py
```
Runs all stages sequentially:
1. Trains the model for the specified number of generations
2. Saves the best genome
3. Runs inference on the best genome
4. Evaluates the model's performance

### 2. Inference Mode
```bash
python em43_python/em43_demo.py --stage infer
```
Starts from inference using the saved best genome and continues to evaluation.

### 3. Evaluation Mode
```bash
python em43_python/em43_demo.py --stage evaluate
```
Directly evaluates the saved best genome.

### Custom Configuration
You can customize the training process using various command-line arguments:
```bash
python em43_python/em43_demo.py --help
```

Key parameters include:
- `--pop_size`: Population size for the genetic algorithm
- `--generations`: Number of generations to run
- `--mut_rule`: Rule mutation rate
- `--mut_prog`: Program mutation rate
- `--prog_len`: Program length
- `--window`: Simulation window size
- `--max_steps`: Maximum steps in simulation
- `--halt_thresh`: Halt threshold for simulation

### Example Usage
```bash
# Run full training with custom parameters
python em43_python/em43_demo.py --pop_size 10000 --generations 200

# Run inference only using saved model
python em43_python/em43_demo.py --stage infer

# Evaluate saved model only
python em43_python/em43_demo.py --stage evaluate
```

### Output
The demo generates two visualization files:
- `prediction_plot.png`: Shows expected vs predicted outputs
- `program_colors.png`: Visualizes the program colormap

The evaluation output provides several key metrics:
- Stored fitness: The fitness score from training
- Avg |err|: Average absolute error
- Success rate: Percentage of outputs within 0.1 of expected
- Accuracy: Percentage of exact matches
- Program visualization: Shows the learned program rules
- Input/Output table: Detailed comparison of actual vs expected outputs

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
