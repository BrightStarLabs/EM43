# EM43 - Emergent Model with 4 States

## 🎯 Overview

EM43 is a comprehensive implementation of an emergent model featuring a 4-state cellular automaton with 3-cell neighborhood. This refactored system provides a clean, modern interface for training, evaluating, and running inference with EM43 models, supporting both single-input and two-input computational tasks.

### ✨ Key Features

- **🔄 Unified Interface**: Single `EM43Wrapper` class for all task types
- **🧠 Auto-Detection**: Automatically detects task mode (1-input vs 2-input)
- **⚙️ Config Integration**: Seamless YAML-based configuration system
- **🚀 Easy Pipeline**: Simple `train()` → `evaluate()` → `infer()` workflow
- **📊 Rich Visualization**: Automatic plot generation and performance metrics
- **💾 Smart Genome Management**: Automatic save/load with format detection
- **🎛️ Command-Line Interface**: Comprehensive CLI with interactive mode
- **🧪 Comprehensive Testing**: Full test coverage ensuring reliability
- **⚡ Performance**: Numba-accelerated simulations with parallel processing
- **🌐 Distributed Logging**: Cloud-based training data logging for collaborative research

## 📁 Project Structure

```
em43_python_refactored/
├── README.md                 # This comprehensive guide
├── config.yaml              # Unified YAML configuration
├── em43_wrapper.py          # 🎯 Main unified interface
├── demo.py                  # 🖥️ Command-line interface
├── demo_logger.py           # 🌐 Enhanced demo with automatic logging
├── tasks_config.py          # Task definitions and validation sets
├── em43_ga.py              # Clean GA implementation
├── em43_numba.py           # Optimized simulation engine
├── evaluate.py             # Evaluation logic
├── inference.py            # Inference logic
├── tests/                  # ⇢ Test scripts & examples
│   ├── test.py             # Comprehensive test suite
│   └── example_usage.py    # 📖 Usage examples
├── notebooks/              # Interactive notebooks
│   ├── test_training_notebook.ipynb
│   └── decode_rule.ipynb
├── logger/                 # 🌐 Distributed logging system
│   ├── register.py         # User registration
│   ├── log.py              # Training data logger
│   ├── config_log.yaml     # Logger configuration
│   ├── test_api_registration.ipynb  # Logger testing
│   └── README.md           # Logger documentation
├── dp_checkpoints/         # Saved genomes directory
└── plots/                  # Generated plots directory
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install numpy numba pyyaml tqdm matplotlib
```

### 1. Basic Usage - Unified Interface

```python
from em43_wrapper import EM43Wrapper

# Create wrapper for Task 1 (multiply by 2)
model = EM43Wrapper(task_id=1)

# Train the model
model.train()

# Evaluate performance
results = model.evaluate()

# Run inference
outputs = model.infer([1, 2, 3, 4, 5])
print(f"Results: {outputs}")
```

### 2. Two-Input Tasks

```python
# Create wrapper for Task 20 (summation)
model = EM43Wrapper(task_id=20)
model.train()

# Two-input inference
outputs = model.infer(([1, 2, 3], [4, 5, 6]))
print(f"1+4={outputs[0]}, 2+5={outputs[1]}, 3+6={outputs[2]}")
```

### 3. Interactive Test Notebook

For a comprehensive introduction:

```bash
# Launch Jupyter and open the test notebook
jupyter notebook notebooks/test_training_notebook.ipynb
```

The notebook demonstrates training, evaluation, inference, and visualization.

## 🖥️ Command-Line Interface

### Interactive Mode

```bash
# Start interactive task selection
python demo.py --interactive

# List all available tasks
python demo.py --list-tasks
```

### Direct Execution

```bash
# Full pipeline (train → evaluate → infer)
python demo.py --task 1 --stage all

# Individual stages
python demo.py --task 1 --stage train
python demo.py --task 1 --stage evaluate
python demo.py --task 1 --stage infer --inputs "10,20,30"

# Two-input inference
python demo.py --task 20 --stage infer --inputs "1,2,3" --inputs-b "4,5,6"

# Custom parameters
python demo.py --task 1 --pop-size 500 --generations 100 --prog-len 15
```

### With Distributed Logging

```bash
# Use demo_logger.py for automatic logging to cloud API
python demo_logger.py --task 1 --stage train

# Features automatic registration, health checks, and offline fallback
python demo_logger.py --task 2 --stage all
```

## 🌐 Distributed Logging System

The system includes comprehensive distributed logging for collaborative research:

### Setup
```bash
# Navigate to logger directory
cd logger

# First-time setup: register your user
python register.py

# Test the system
python log.py
```

### Features
- **Automatic Data Logging**: Generation, fitness, population parameters
- **Task Information**: Task ID, description, timestamps
- **Security**: User registration with unique IDs, no API keys required
- **Git Exclusion**: `user_config.json` automatically excluded from version control

**📚 Complete documentation**: See `logger/README.md`

## 📋 Available Tasks

### 1-Input Tasks (Functions of single numbers)
| Task | Description | Training | Validation | Example |
|------|-------------|----------|------------|---------|
| 1 | multiply by 2 | 30 samples | 20 samples | 5 → 10 |
| 2 | multiply by 3 | 30 samples | 20 samples | 4 → 12 |
| 6 | add 1 | 30 samples | 20 samples | 7 → 8 |
| 7 | subtract 1 | 30 samples | 20 samples | 8 → 7 |
| 10 | modulo 4 | 30 samples | 20 samples | 9 → 1 |
| 11 | divide by 2 | 30 samples | 20 samples | 8 → 4 |
| 12 | square | 30 samples | 20 samples | 5 → 25 |
| 15 | factorial | 10 samples | 10 samples | 4 → 24 |
| 16 | fibonacci | 15 samples | 10 samples | 5 → 5 |

### 2-Input Tasks (Operations on pairs of numbers)
| Task | Description | Training | Validation | Example |
|------|-------------|----------|------------|---------|
| 20 | summation | 144 samples | 36 samples | (3,4) → 7 |
| 21 | multiplication | 144 samples | 36 samples | (3,4) → 12 |
| 22 | subtraction | 144 samples | 36 samples | (7,3) → 4 |
| 23 | maximum | 144 samples | 36 samples | (3,7) → 7 |
| 24 | minimum | 144 samples | 36 samples | (3,7) → 3 |
| 25 | power | 81 samples | 36 samples | (2,3) → 8 |
| 27 | GCD | 45 samples | 20 samples | (6,9) → 3 |
| 28 | LCM | 45 samples | 20 samples | (6,9) → 18 |

**Validation Strategy**: All tasks include out-of-distribution validation sets using larger input values than training to test generalization.

## ⚙️ Configuration System

### Config File Structure (`config.yaml`)

```yaml
population:
  pop_size: {value: 6000}      # Population size
  generations: {value: 300}    # Number of generations
  elite_frac: {value: 0.1}     # Elite preservation ratio
  tourney_k: {value: 3}        # Tournament selection size
  mut_rule: {value: 0.02}      # Rule mutation probability
  mut_prog: {value: 0.04}      # Program mutation probability

model:
  prog_len: {value: 20}        # Program length
  window: {value: 100}         # Training window size
  max_steps: {value: 200}      # Training max steps
  halt_thresh: {value: 0.50}   # Halting threshold

evaluation:
  window: {value: 10000}       # Evaluation/inference window
  max_steps: {value: 3000}     # Evaluation/inference max steps

input_output:
  task_id: {value: 1}          # Default task ID (-1 for custom)
```

### Custom Configuration

```python
# Override default parameters
model = EM43Wrapper(
    task_id=1,
    config_overrides={
        'population': {
            'pop_size': {'value': 1000},
            'generations': {'value': 100}
        },
        'model': {
            'prog_len': {'value': 15},
            'window': {'value': 200}
        }
    }
)
model.train()
```

## 🧬 System Architecture

### Core Components

1. **4-State Cellular Automaton**: States `0` (blank), `1` (program), `2` (red marker), `3` (blue boundary/halt)
2. **3-Cell Neighborhood**: Each cell's next state depends on itself and its two neighbors
3. **"00" Separator**: Separates program from input data
4. **Genetic Algorithm**: Evolves rule tables and initial programs
5. **Numba Acceleration**: JIT compilation for high-performance simulation

### Data Flow Pipeline

```
Configuration → Task Setup → Training (GA) → Evaluation → Inference
     ↓              ↓            ↓             ↓          ↓
YAML Config → Input/Target → Evolution → Validation → Custom Inputs
                Arrays                      Sets
```

### Two-Input System

For two-input tasks, the tape structure is:
```
[program] 00 0^(a+1) R 0^(b+1) R 0
```

- **Program**: Fixed-length sequence
- **Separator**: Always "00"
- **First input**: a+1 zeros followed by red marker R
- **Second input**: b+1 zeros followed by red marker R
- **Output space**: For result encoding

## ✅ Testing and Validation

### Run Comprehensive Tests
```bash
cd em43_python_refactored
python tests/test.py
```

### Test Coverage

The system includes **16 test sections** with **331 individual tests**:

- Configuration System (12 tests)
- Task Definitions (45 tests)
- Data Processing (8 tests)
- Genome Operations (15 tests)
- Fitness Evaluation (12 tests)
- GA Evolution (8 tests)
- Evaluation System (25 tests)
- Inference System (18 tests)
- Wrapper System (25 tests)
- Demo System (8 tests)
- Integration Workflows (15 tests)
- Performance Benchmarks (8 tests)

**Expected Results**: All 331 tests pass with mathematical verification and performance benchmarks.

## 🧬 Genetic Algorithm Details

### Core Components

1. **Initialization**: Random rule tables (64 entries) + programs (variable length)
2. **Selection**: Tournament selection with configurable tournament size
3. **Crossover**: Segment crossover with random segment length and position
4. **Mutation**: Per-element mutation with separate rates for rules vs programs
5. **Sanitization**: Enforces cellular automaton constraints

### Evolution Process
```python
for generation in range(generations):
    fitness = evaluate_population_parallel()
    population = sort_by_fitness(population, fitness)
    next_population = preserve_elite(population)
    
    while len(next_population) < pop_size:
        parent1, parent2 = tournament_selection(population, fitness)
        child = crossover(parent1, parent2)
        child = mutate(child)
        child = sanitize(child)
        next_population.append(child)
    
    population = next_population
```

## 📊 Performance Metrics

### Training Metrics
- **Fitness Evolution**: Track improvement over generations
- **Best Fitness**: Final best fitness achieved
- **Training Time**: Total evolution time
- **Convergence**: Fitness stability analysis

### Evaluation Metrics
- **Accuracy**: Percentage of exact matches
- **Success Rate**: Percentage within ±0.1 tolerance
- **Mean Error**: Average absolute error
- **Max Error**: Maximum absolute error
- **Failed Simulations**: Count of simulation failures

### Performance Benchmarks
- **Fitness Evaluation**: < 5s for 50 genomes
- **Memory Usage**: Efficient numpy array management
- **Inference Speed**: < 5s for 20 inputs

## 🔧 Advanced Usage

### Custom Task Definition
```python
# Edit tasks_config.py to add custom task
custom_task = Task(
    description="multiply by 7",
    inputs=np.array([[1], [2], [3], [4]]),
    targets=np.array([7, 14, 21, 28]),
    mode="1input"
)
TASKS[-1] = custom_task

# Use custom task
model = EM43Wrapper(task_id=-1)
```

### Batch Processing
```python
# Process multiple tasks
tasks_to_test = [1, 2, 6, 7, 20, 21]
results = {}

for task_id in tasks_to_test:
    model = EM43Wrapper(task_id=task_id)
    model.train(verbose=False)
    eval_results = model.evaluate(verbose=False, plot=False)
    results[task_id] = eval_results['accuracy']

print("Task accuracies:", results)
```

## 🐛 Troubleshooting

### Common Issues

1. **"Task X not found"**
   ```bash
   python demo.py --list-tasks  # Check available tasks
   ```

2. **"Window size too small"**
   - System will suggest minimum required size
   - Increase window in config.yaml or via overrides

3. **"All simulations failed"**
   - Increase max_steps in evaluation section
   - Check task compatibility and parameters

4. **Memory issues**
   - Reduce population size and window size
   - Use quiet mode: `verbose=False`, `plot=False`

### Performance Tips

1. **Fast Prototyping**: `pop_size=50-100`, `generations=10-20`
2. **Production Training**: `pop_size=1000+`, `generations=100+`
3. **Large Inputs**: System auto-adjusts window size
4. **Batch Processing**: Use quiet mode for speed

## 📚 Documentation & Support

### Key Files to Review
1. **`tests/example_usage.py`**: Practical usage examples
2. **`demo.py`**: Command-line interface patterns
3. **`tasks_config.py`**: Task definition structure
4. **`tests/test.py`**: Comprehensive testing examples
5. **`logger/README.md`**: Distributed logging documentation

### Quick Reference

```bash
# Quick start
python demo.py --interactive

# Full pipeline
python demo.py --task 1 --stage all

# Fast training
python demo.py --task 1 --pop-size 100 --generations 20

# Validation
python tests/test.py
```

## 🎯 Related Work

This implementation explores [emergent models](https://new.researchhub.com/fund/4130/emergent-models-a-general-modeling-framework-as-an-alternative-to-neural-networks) as an alternative to traditional neural networks, demonstrating how complex computational behaviors can emerge from simple cellular automaton rules.

**Happy evolving! 🧬✨**
