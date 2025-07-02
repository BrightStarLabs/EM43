# EM43 - Emergent Model with 4 States

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Demo Usage](#demo-usage)
- [Configuration](#configuration)
- [Task Tracking and Logging](#task-tracking-and-logging)
- [Documentation](#documentation)
- [Related Work](#related-work)

## Overview
EM43 is an implementation of an [emergent model (EM)](https://new.researchhub.com/fund/4130/emergent-models-a-general-modeling-framework-as-an-alternative-to-neural-networks) featuring 4 states and a neighborhood of 3 cells. The system uses a 1-dimensional cellular automaton with states `0` (blank), `1` (program), `2` (red marker), and `3` (blue boundary/halt), employing a "00" separator between program and input data.

## Features
- **4-state cellular automaton** with 3-cell neighborhood and "00" separator design
- **Parallel processing** using Numba's `prange` for significant speed improvements
- **Genetic Algorithm optimization** with Random-Immigrant Strategy for maintaining diversity
- **Random Search alternative** for baseline comparison and research studies
- **Unified CSV logging** to `log.csv` with UTC timestamps for experiment tracking
- **Task tracking system** with configurable task IDs and descriptions
- **Detailed telemetry** tracking including average Hamming distance
- **Configurable parameters** through YAML configuration
- **Command-line interface** with argument parsing
- **Class-based API** for programmatic use
- **Checkpoint saving** for training resumption

## Installation
### Prerequisites
- Python 3.11 or higher
- 4GB+ RAM recommended for default parameters
- Multi-core CPU recommended for parallel processing 

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
# Basic training with interactive task selection (prompts for task ID)
python em43_python/em43_demo.py

# Training with custom parameters (still prompts for task selection)
python em43_python/em43_demo.py --pop_size 10000 --generations 200

# Random Search instead of Genetic Algorithm
python em43_python/em43_demo.py --enable_rs true --generations 100

# Random Search with custom seed for reproducibility  
python em43_python/em43_demo.py --enable_rs true --rs_seed 123 --pop_size 5000

# Random Search with truly random seed (different results each run)
python em43_python/em43_demo.py --enable_rs true --rs_seed -1

# Parameter sweep example (with task selection prompt)
python em43_python/em43_demo.py --lambda_p 0.1

# Run inference only using saved model
python em43_python/em43_demo.py --stage infer

# Evaluate saved model only
python em43_python/em43_demo.py --stage evaluate
```

All training runs automatically log to `log.csv` with UTC timestamps for easy experiment tracking and comparison.

### Output
The demo generates several outputs:

**Visualization Files:**
- `prediction_plot.png`: Shows expected vs predicted outputs
- `program_colors.png`: Visualizes the program colormap

**Automatic Logging:**
- `log.csv`: Unified CSV log with UTC timestamps containing all run data

**Evaluation Metrics:**
- Stored fitness: The fitness score from training
- Avg |err|: Average absolute error
- Success rate: Percentage of outputs within 0.1 of expected
- Accuracy: Percentage of exact matches
- Program visualization: Shows the learned program rules
- Input/Output table: Detailed comparison of actual vs expected outputs

## Configuration
The main configuration file is `em43_python/config.yaml`, which contains all hyperparameters organized into sections:
- **Population parameters**: Population size, generations, elite fraction, tournament size
- **Mutation rates**: Rule and program mutation probabilities
- **Regularization**: Sparsity penalty, random immigrants, telemetry frequency
- **Simulation parameters**: Window size, max steps, halt threshold
- **Checkpoint settings**: Save frequency, checkpoint directory
- **Input/output configuration**: Input range and target output expressions
- **Task tracking**: Task ID and description for experiment management

## Task Tracking and Logging

**Interactive Task Selection:**
Before each training run, the system prompts for task selection:
- Tasks are defined in `task_descriptions.csv`
- Default tasks: `0` (undefined), `1` (multiply by 2), `2` (multiply by 3)
- Press Enter for default task 0, or enter task ID + confirmation
- Prevents accidental logging with wrong task association

All training runs are automatically logged to `log.csv` with the following information:
- **UTC timestamps**: When each checkpoint was saved
- **Run identification**: Unique run ID and selected task information
- **Fitness tracking**: Initial and final fitness values
- **Complete genomes**: Full rule tables and program sequences
- **Experimental metadata**: Task descriptions and checkpoint information

View logs with pandas:
```python
import pandas as pd
df = pd.read_csv('log.csv')
print(df.groupby('task_id')['final_fitness'].max())

# View task descriptions
tasks = pd.read_csv('task_descriptions.csv')
print(tasks)
```

## Documentation

For comprehensive information, see the following guides:

- **[EM43_SUMMARY.md](EM43_SUMMARY.md)**: High-level overview and quick start guide
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)**: Essential commands and parameter reference
- **[RANDOM_SEARCH_QUICKSTART.md](RANDOM_SEARCH_QUICKSTART.md)**: Quick testing guide for Random Search (start here!)
- **[RANDOM_SEARCH_GUIDE.md](RANDOM_SEARCH_GUIDE.md)**: Complete Random Search documentation
- **[TASK_TRACKING_GUIDE.md](TASK_TRACKING_GUIDE.md)**: Complete experiment management guide
- **[ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md)**: System architecture and design
- **[SEPARATOR_CHANGE_ANALYSIS.md](SEPARATOR_CHANGE_ANALYSIS.md)**: Technical details on separator changes
- **[UNIFIED_LOGGING_SUMMARY.md](UNIFIED_LOGGING_SUMMARY.md)**: Logging system updates
- **[INTERACTIVE_TASK_SELECTION_SUMMARY.md](INTERACTIVE_TASK_SELECTION_SUMMARY.md)**: Interactive task selection implementation

## Related Work

This implementation explores [emergent models](https://new.researchhub.com/fund/4130/emergent-models-a-general-modeling-framework-as-an-alternative-to-neural-networks) as an alternative to traditional neural networks, demonstrating how complex computational behaviors can emerge from simple cellular automaton rules.

## Warranty
This software is provided "as is" without warranty of any kind, express or
implied, including but not limited to the warranties of merchantability,
fitness for a particular purpose and noninfringement. In no event shall the
authors or copyright holders be liable for any claim, damages or other
liability, whether in an action of contract, tort or otherwise, arising from,
out of or in connection with the software or the use or other dealings in the
software.
