# EM43 - Emergent Model with 4 States

## Overview
EM43 is an implementation of an [emergent model (EM)](https://new.researchhub.com/fund/4130/emergent-models-a-general-modeling-framework-as-an-alternative-to-neural-networks) featuring 4 states and a neighborhood of 3 cells. This project provides both training and inference capabilities for EM models with the following characteristics:

- **Architecture**: 1-D EM
- **States**: 4 distinct states (0-3)
- **Neighborhood**: Radius-1 neighborhood (3 cells)
- **Boundary Conditions**: Open boundary conditions with 2-cell separator "BB"
- **Optimization**: Numba-accelerated parallel processing for efficient computation

The project includes two main components:
1. `em43_numba.py`: Core simulation engine with Numba-accelerated parallel processing
2. `em43_numba_ga.py`: Genetic Algorithm implementation for training the model

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

## Usage

### Training
To train the model using the Genetic Algorithm:
```bash
python em43_numba_ga.py
```

The training process will:
- Initialize a population of candidate solutions
- Evaluate fitness using parallel processing
- Evolve the population using genetic operators
- Track progress with detailed telemetry
- Save checkpoints periodically

### Inference
To run inference on trained models:
```bash
python em43_numba.py
```

The inference engine provides fast evaluation of trained models on input data.

## Features
- Parallel processing using Numba's `prange` for significant speed improvements
- Random-Immigrant Strategy for maintaining genetic diversity
- Detailed telemetry tracking including average Hamming distance
- Configurable parameters for population size, generations, and mutation rates
- Checkpoint saving for training resumption

## Configuration
Key hyperparameters can be configured in `em43_numba_ga.py`:
- `POP_SIZE`: Population size (default: 20000)
- `N_GENERATIONS`: Number of generations (default: 300)
- `WINDOW`: Tape length (default: 200)
- `MAX_STEPS`: Maximum simulation steps (default: 800)


## Warranty
This software is provided "as is" without warranty of any kind, express or
implied, including but not limited to the warranties of merchantability,
fitness for a particular purpose and noninfringement. In no event shall the
authors or copyright holders be liable for any claim, damages or other
liability, whether in an action of contract, tort or otherwise, arising from,
out of or in connection with the software or the use or other dealings in the
software.

The software is experimental and may present unexpected behaviors. Users are
advised to test thoroughly before using in production environments.
