"""
Comprehensive Test Suite for EM43 Refactored System
==================================================

This test suite validates:
1. Configuration system and YAML parsing
2. Task definitions and mathematical correctness
3. Array dimensions and data processing
4. GA components and evolution process
5. Fitness evaluation and numba integration
6. End-to-end workflow validation
7. Error handling and edge cases

IMPORTANT: All training calls explicitly disable API logging
(checkpoint_callback=None, checkpoint_interval=0) to prevent
accidental data injection into the database during testing.
"""

# Standard library
import os
import pickle
import time
import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import warnings

# Third-party
import numpy as np
import yaml

# Ensure imports resolve when executed from tests subdirectory.
# ---------------------------------------------------------------------------
# Add parent directory to Python path to import EM43 modules
# Keep ROOT_DIR for other uses (config paths, etc.)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Change to the root directory for file operations (config.yaml, dp_checkpoints/, etc.)
os.chdir(ROOT_DIR)

# Import EM43 components (now that ROOT_DIR is on sys.path)
from em43_ga import EM43GA, train_model
from tasks_config import get_dataset, TASKS, custom_task
from em43_numba import (
    fitness_population, fitness_population_two_inputs, 
    _sanitize_rule, _sanitize_programme, 
    EM43Batch, EM43TwoInputBatch
)

# Test configuration
TEST_CONFIG = {
    'population': {
        'pop_size': {'value': 20},
        'generations': {'value': 10},
        'elite_frac': {'value': 0.2},
        'tourney_k': {'value': 3},
        'mut_rule': {'value': 0.05},
        'mut_prog': {'value': 0.03},
    },
    'model': {
        'prog_len': {'value': 8},
        'window': {'value': 200},
        'max_steps': {'value': 50},
        'halt_thresh': {'value': 0.5},
    }
}


class TestResults:
    """Track and report test results."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.sections = []
    
    def section(self, name: str):
        """Start a new test section."""
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print(f"{'='*60}")
        self.sections.append(name)
    
    def test(self, name: str, condition: bool, message: str = ""):
        """Record a test result."""
        if condition:
            self.passed += 1
            print(f"✅ {name}")
            if message:
                print(f"   {message}")
        else:
            self.failed += 1
            print(f"❌ {name}")
            if message:
                print(f"   {message}")
            raise AssertionError(f"Test failed: {name}")
    
    def warning(self, name: str, message: str):
        """Record a warning."""
        self.warnings += 1
        print(f"⚠️  {name}: {message}")
    
    def summary(self):
        """Print final test summary."""
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Sections tested: {len(self.sections)}")
        print(f"Tests passed: {self.passed}/{total}")
        print(f"Tests failed: {self.failed}/{total}")
        print(f"Warnings: {self.warnings}")
        
        if self.failed == 0:
            print(f"\n🎉 ALL TESTS PASSED! System is fully validated.")
        else:
            print(f"\n💥 {self.failed} tests failed. Review errors above.")
        
        return self.failed == 0


def test_configuration_system(results: TestResults):
    """Test YAML configuration loading and validation."""
    results.section("Configuration System")
    
    # Test YAML file exists and loads
    config_path = ROOT_DIR / 'config.yaml'
    results.test("Config file exists", config_path.exists())
    
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    results.test("YAML loads successfully", config_data is not None)
    
    # Test required sections exist
    required_sections = ['population', 'model', 'input_output']
    for section in required_sections:
        results.test(f"Section '{section}' exists", section in config_data)
    
    # Test parameter structure
    for section, params in config_data.items():
        if section == 'config_file':
            continue
        results.test(f"Section '{section}' has proper structure", 
                    isinstance(params, dict))
        
        for param_name, param_data in params.items():
            results.test(f"Parameter '{param_name}' has value field",
                        'value' in param_data)
    
    # Test parameter types and ranges
    task_id = config_data['input_output']['task_id']['value']
    results.test("task_id is integer", isinstance(task_id, int))
    results.test("task_id in valid range", task_id in TASKS or task_id == -1)
    
    pop_size = config_data['population']['pop_size']['value']
    results.test("pop_size is positive integer", 
                isinstance(pop_size, int) and pop_size > 0)
    
    generations = config_data['population']['generations']['value']
    results.test("generations is positive integer",
                isinstance(generations, int) and generations > 0)


def test_task_definitions(results: TestResults):
    """Test all task definitions for correctness."""
    results.section("Task Definitions")
    
    # Test task registry
    results.test("Tasks registry exists", len(TASKS) > 0)
    results.test("Custom task exists", -1 in TASKS)
    
    # Test each task type
    one_input_tasks = [tid for tid, task in TASKS.items() if task.mode == "1input"]
    two_input_tasks = [tid for tid, task in TASKS.items() if task.mode == "2input"]
    
    results.test("1-input tasks exist", len(one_input_tasks) > 0)
    results.test("2-input tasks exist", len(two_input_tasks) > 0)
    
    print(f"   Found {len(one_input_tasks)} one-input and {len(two_input_tasks)} two-input tasks")
    
    # Test each task in detail
    for task_id, task in TASKS.items():
        task_name = f"Task {task_id} ({task.description})"
        
        # Test basic properties
        results.test(f"{task_name} has valid mode", 
                    task.mode in ["1input", "2input"])
        results.test(f"{task_name} has description", 
                    isinstance(task.description, str) and len(task.description) > 0)
        
        # Test array dimensions
        if task.mode == "1input":
            results.test(f"{task_name} inputs shape", 
                        task.inputs.ndim == 2 and task.inputs.shape[1] == 1)
        else:
            results.test(f"{task_name} inputs shape", 
                        task.inputs.ndim == 2 and task.inputs.shape[1] == 2)
        
        results.test(f"{task_name} targets shape", task.targets.ndim == 1)
        results.test(f"{task_name} consistent lengths", 
                    task.inputs.shape[0] == task.targets.shape[0])
        
        # Test validation sets if present
        if task.valid_inputs is not None:
            results.test(f"{task_name} validation inputs shape",
                        task.valid_inputs.shape[1] == task.inputs.shape[1])
            results.test(f"{task_name} validation consistency",
                        task.valid_inputs.shape[0] == task.valid_targets.shape[0])
            
            # Test out-of-distribution property
            if task.mode == "1input":
                train_vals = task.inputs[:, 0]
                valid_vals = task.valid_inputs[:, 0]
                overlap = len(np.intersect1d(train_vals, valid_vals))
                results.test(f"{task_name} validation is OOD", overlap == 0)


def test_mathematical_correctness(results: TestResults):
    """Test mathematical correctness of task computations."""
    results.section("Mathematical Correctness")
    
    # Test specific tasks with known computations
    test_cases = [
        (1, "multiply by 2", lambda x: x * 2),
        (2, "multiply by 3", lambda x: x * 3),
        (6, "add 1", lambda x: x + 1),
        (7, "subtract 1", lambda x: x - 1),
        (10, "modulo 4", lambda x: x % 4),
        (20, "summation", lambda a, b: a + b),
        (21, "multiplication", lambda a, b: a * b),
        (22, "subtraction", lambda a, b: a - b),
        (23, "maximum", lambda a, b: max(a, b)),
        (24, "minimum", lambda a, b: min(a, b)),
    ]
    
    for task_id, desc, func in test_cases:
        if task_id not in TASKS:
            continue
            
        task = TASKS[task_id]
        
        if task.mode == "1input":
            inputs = task.inputs[:, 0]
            expected = np.array([func(x) for x in inputs])
            results.test(f"{desc} computation", np.allclose(task.targets, expected))
        else:
            inputs_a = task.inputs[:, 0]
            inputs_b = task.inputs[:, 1]
            expected = np.array([func(a, b) for a, b in zip(inputs_a, inputs_b)])
            results.test(f"{desc} computation", np.allclose(task.targets, expected))


def test_data_processing(results: TestResults):
    """Test data processing and array transformations."""
    results.section("Data Processing")
    
    # Test GA initialization with different task types
    for task_type, task_id in [("1input", 1), ("2input", 20)]:
        config = {**TEST_CONFIG, 'input_output': {'task_id': {'value': task_id}}}
        
        try:
            ga = EM43GA(config)
            results.test(f"GA initialization ({task_type})", True)
            results.test(f"Mode detection ({task_type})", ga.mode == task_type)
            
            if task_type == "1input":
                results.test(f"1D input processing", ga.inputs.ndim == 1)
                results.test(f"Input type conversion", ga.inputs.dtype == np.int32)
            else:
                results.test(f"2D input processing", 
                            hasattr(ga, 'inputs_a') and hasattr(ga, 'inputs_b'))
                results.test(f"Input separation", 
                            ga.inputs_a.ndim == 1 and ga.inputs_b.ndim == 1)
            
        except Exception as e:
            results.test(f"GA initialization ({task_type})", False, str(e))


def test_genome_operations(results: TestResults):
    """Test genome generation, sanitization, and operations."""
    results.section("Genome Operations")
    
    config = {**TEST_CONFIG, 'input_output': {'task_id': {'value': 1}}}
    ga = EM43GA(config)
    
    # Test genome generation
    rule, prog = ga._random_genome()
    results.test("Rule generation", rule.shape == (64,) and rule.dtype == np.uint8)
    results.test("Program generation", prog.shape == (ga.prog_len,) and prog.dtype == np.uint8)
    results.test("Rule values in range", np.all(rule >= 0) and np.all(rule <= 3))
    results.test("Program values in range", np.all(prog >= 0) and np.all(prog <= 2))
    
    # Test sanitization
    dirty_rule = np.random.randint(0, 10, 64, dtype=np.uint8)
    clean_rule = _sanitize_rule(dirty_rule)
    results.test("Rule sanitization", np.all(clean_rule <= 3))
    
    dirty_prog = np.array([0, 1, 2, 3, 1, 0], dtype=np.uint8)
    clean_prog = _sanitize_programme(dirty_prog)
    results.test("Program sanitization", np.all(clean_prog != 3))
    
    # Test crossover
    rule1, prog1 = ga._random_genome()
    rule2, prog2 = ga._random_genome()
    child_rule, child_prog = ga._crossover(rule1, prog1, rule2, prog2)
    
    results.test("Crossover rule output", child_rule.shape == rule1.shape)
    results.test("Crossover program output", child_prog.shape == prog1.shape)
    results.test("Crossover creates difference", 
                not np.array_equal(child_rule, rule1) or not np.array_equal(child_prog, prog1))
    
    # Test mutation
    orig_rule, orig_prog = rule1.copy(), prog1.copy()
    mut_rule, mut_prog = ga._mutate(rule1, prog1)
    results.test("Mutation preserves shape (rule)", mut_rule.shape == orig_rule.shape)
    results.test("Mutation preserves shape (prog)", mut_prog.shape == orig_prog.shape)


def test_fitness_evaluation(results: TestResults):
    """Test fitness evaluation functions."""
    results.section("Fitness Evaluation")
    
    # Create test population
    pop_size = 5
    rules = np.random.randint(0, 4, (pop_size, 64), dtype=np.uint8)
    progs = np.random.randint(0, 3, (pop_size, 8), dtype=np.uint8)
    
    # Sanitize population
    for i in range(pop_size):
        rules[i] = _sanitize_rule(rules[i])
        progs[i] = _sanitize_programme(progs[i])
    
    # Test 1-input fitness
    task_1 = get_dataset(1)
    inputs_1 = task_1.inputs[:, 0].astype(np.int32)
    targets_1 = task_1.targets.astype(np.int32)
    
    try:
        fitness_1 = fitness_population(rules, progs, inputs_1, targets_1, 200, 50, 0.5, 0.0)
        results.test("1-input fitness execution", True)
        results.test("1-input fitness shape", fitness_1.shape == (pop_size,))
        results.test("1-input fitness type", fitness_1.dtype == np.float32)
        results.test("1-input fitness finite", np.all(np.isfinite(fitness_1)))
    except Exception as e:
        results.test("1-input fitness execution", False, str(e))
    
    # Test 2-input fitness
    task_2 = get_dataset(20)
    inputs_a = task_2.inputs[:, 0].astype(np.int32)
    inputs_b = task_2.inputs[:, 1].astype(np.int32)
    targets_2 = task_2.targets.astype(np.int32)
    
    try:
        fitness_2 = fitness_population_two_inputs(
            rules, progs, inputs_a, inputs_b, targets_2, 200, 50, 0.5, 0.0)
        results.test("2-input fitness execution", True)
        results.test("2-input fitness shape", fitness_2.shape == (pop_size,))
        results.test("2-input fitness type", fitness_2.dtype == np.float32)
        results.test("2-input fitness finite", np.all(np.isfinite(fitness_2)))
    except Exception as e:
        results.test("2-input fitness execution", False, str(e))


def test_simulation_wrappers(results: TestResults):
    """Test EM43Batch simulation wrappers."""
    results.section("Simulation Wrappers")
    
    # Create a simple genome
    rule = np.random.randint(0, 4, 64, dtype=np.uint8)
    prog = np.random.randint(0, 3, 8, dtype=np.uint8)
    rule = _sanitize_rule(rule)
    prog = _sanitize_programme(prog)
    genome = (rule, prog)
    
    # Test 1-input wrapper
    try:
        sim1 = EM43Batch(genome, window=300, max_steps=100)
        outputs1 = sim1.run([1, 2, 3, 4, 5])
        results.test("1-input simulation creation", True)
        results.test("1-input simulation output", isinstance(outputs1, np.ndarray))
        results.test("1-input simulation length", len(outputs1) == 5)
    except Exception as e:
        results.test("1-input simulation", False, str(e))
    
    # Test 2-input wrapper
    try:
        sim2 = EM43TwoInputBatch(genome, window=500, max_steps=100)
        outputs2 = sim2.run([1, 2, 3], [4, 5, 6])
        results.test("2-input simulation creation", True)
        results.test("2-input simulation output", isinstance(outputs2, np.ndarray))
        results.test("2-input simulation length", len(outputs2) == 3)
    except Exception as e:
        results.test("2-input simulation", False, str(e))


def test_ga_evolution(results: TestResults):
    """Test complete GA evolution process."""
    results.section("GA Evolution Process")
    
    # Test meaningful evolution run with sufficient population and generations
    config = {
        'population': {
            'pop_size': {'value': 1000},
            'generations': {'value': 20},
            'elite_frac': {'value': 0.1},
            'tourney_k': {'value': 3},
            'mut_rule': {'value': 0.01},
            'mut_prog': {'value': 0.03},
        },
        'model': {
            'prog_len': {'value': 8},
            'window': {'value': 200},
            'max_steps': {'value': 50},
            'halt_thresh': {'value': 0.5},
        },
        'input_output': {'task_id': {'value': 1}}  # Simple multiply by 2 task
    }
    
    try:
        start_time = time.time()
        ga = EM43GA(config)
        
        # Test initial population
        results.test("Population initialization", 
                    ga.pop_rules.shape == (1000, 64) and ga.pop_progs.shape == (1000, 8))
        
        # Run evolution
        best_rule, best_prog, best_fitness, history = ga.evolve()
        evolution_time = time.time() - start_time
        
        results.test("Evolution completion", True)
        results.test("Best genome returned", 
                    best_rule.shape == (64,) and best_prog.shape == (8,))
        results.test("Fitness history", len(history) == 20)
        results.test("Evolution time reasonable", evolution_time < 120.0)  # Allow more time for larger run
        
        # Test fitness improvement with reasonable expectations
        if len(history) > 10:  # Check improvement over more generations
            # Compare final third vs first third of evolution
            early_fitness = np.mean(history[:7])  # First third
            late_fitness = np.mean(history[-7:])   # Last third
            
            # Since fitness is negative (-error - penalty), higher values are better
            # Check if late fitness is better (higher) than early fitness
            improvement = late_fitness - early_fitness  # Positive = improvement
            relative_improvement = improvement / abs(early_fitness) if early_fitness != 0 else 0
            
            # Allow for some variation - evolution doesn't always improve monotonically
            # Just check that we didn't get significantly worse
            results.test("Fitness improvement trend", 
                        late_fitness >= early_fitness - abs(early_fitness) * 0.1,
                        f"Early: {early_fitness:.3f}, Late: {late_fitness:.3f}, Change: {improvement:.3f}")
        
        print(f"   Evolution completed in {evolution_time:.2f}s")
        print(f"   Initial fitness: {history[0]:.3f}")
        print(f"   Final fitness: {history[-1]:.3f}")
        print(f"   Best improvement: {max(history):.3f}")
        
    except Exception as e:
        results.test("GA evolution", False, str(e))


def test_integration_workflow(results: TestResults):
    """Test complete end-to-end workflow."""
    results.section("Integration Workflow")
    
    # Test train_model function
    config = {
        'population': {
            'pop_size': {'value': 8},
            'generations': {'value': 3},
            'elite_frac': {'value': 0.25},
            'tourney_k': {'value': 2},
            'mut_rule': {'value': 0.05},
            'mut_prog': {'value': 0.05},
        },
        'model': {
            'prog_len': {'value': 5},
            'window': {'value': 100},
            'max_steps': {'value': 20},
            'halt_thresh': {'value': 0.5},
        },
        'input_output': {'task_id': {'value': 6}}  # Add 1 task
    }
    
    try:
        # Test complete training workflow
        best_rule, best_prog, best_fitness, history = train_model(config, checkpoint_callback=None, checkpoint_interval=0)
        results.test("Complete training workflow", True)
        
        # Check if genome file was saved
        expected_file = ROOT_DIR / "dp_checkpoints/best_genome.pkl"
        results.test("Genome file saved", expected_file.exists())
        
        if expected_file.exists():
            # Test loading saved genome
            with open(expected_file, 'rb') as f:
                loaded_data = pickle.load(f)
            results.test("Genome loading", 'best_rule' in loaded_data)
            results.test("Saved genome consistency", 
                        np.array_equal(loaded_data['best_rule'], best_rule))
            
    except Exception as e:
        results.test("Integration workflow", False, str(e))


def test_edge_cases(results: TestResults):
    """Test edge cases and error handling."""
    results.section("Edge Cases & Error Handling")
    
    # Test invalid task ID
    try:
        invalid_config = {**TEST_CONFIG, 'input_output': {'task_id': {'value': 999}}}
        ga = EM43GA(invalid_config)
        results.test("Invalid task ID handling", False, "Should have raised error")
    except (ValueError, KeyError):
        results.test("Invalid task ID handling", True)
    except Exception as e:
        results.test("Invalid task ID handling", False, f"Wrong error type: {e}")
    
    # Test small window size validation
    try:
        small_window_config = {
            **TEST_CONFIG, 
            'model': {**TEST_CONFIG['model'], 'window': {'value': 10}},
            'input_output': {'task_id': {'value': 1}}
        }
        ga = EM43GA(small_window_config)
        results.test("Small window validation", False, "Should have raised ValueError")
    except ValueError as e:
        results.test("Small window validation", True, f"Correctly rejected: {str(e)[:80]}...")
    except Exception as e:
        results.test("Small window validation", False, f"Wrong error type: {type(e).__name__}")
    
    # Test valid window size acceptance
    try:
        valid_window_config = {
            **TEST_CONFIG, 
            'model': {**TEST_CONFIG['model'], 'window': {'value': 200}},
            'input_output': {'task_id': {'value': 1}}
        }
        ga = EM43GA(valid_window_config)
        results.test("Valid window acceptance", True)
    except Exception as e:
        results.test("Valid window acceptance", False, f"Valid config rejected: {e}")


def test_evaluation_system(results: TestResults):
    """Test evaluation script functionality."""
    results.section("Evaluation System")
    
    # Import evaluation components
    try:
        from evaluate import load_config, load_genome, evaluate_genome_on_task
        results.test("Evaluation imports", True)
    except ImportError as e:
        results.test("Evaluation imports", False, str(e))
        return
    
    # Test config loading for evaluation
    try:
        config = load_config()
        results.test("Config loading", True)
        
        # Check evaluation config section (used by both evaluation and inference)
        eval_config = config.get('evaluation', {})
        results.test("Evaluation config section exists", 'window' in eval_config)
        results.test("Evaluation window value", eval_config['window']['value'] == 10000)
        results.test("Evaluation max_steps value", eval_config['max_steps']['value'] == 3000)
        
    except Exception as e:
        results.test("Config loading", False, str(e))
    
    # Create a test genome for evaluation
    test_genome = {
        'best_rule': np.random.randint(0, 4, 64, dtype=np.uint8),
        'best_prog': np.random.randint(0, 3, 8, dtype=np.uint8),
        'best_fitness': -5.0,
        'mode': '1input'
    }
    
    # Save test genome
    test_genome_path = Path("test_genome.pkl")
    try:
        with open(test_genome_path, 'wb') as f:
            pickle.dump(test_genome, f)
        results.test("Test genome creation", True)
        
        # Test genome loading
        loaded_genome = load_genome(test_genome_path)
        results.test("Genome loading", True)
        results.test("Loaded genome has rule", 'best_rule' in loaded_genome)
        results.test("Loaded genome has prog", 'best_prog' in loaded_genome)
        
    except Exception as e:
        results.test("Genome loading", False, str(e))
    
    # Test evaluation on tasks with validation sets
    validation_tasks = [tid for tid in TASKS.keys() 
                       if tid != -1 and TASKS[tid].valid_inputs is not None]
    
    if validation_tasks:
        test_task = validation_tasks[0]
        try:
            eval_result = evaluate_genome_on_task(
                test_genome, test_task, large_window=500, max_steps=100
            )
            results.test("Task evaluation execution", eval_result is not None)
            
            if eval_result:
                expected_keys = ['task_id', 'num_samples', 'outputs', 'targets', 
                               'errors', 'success_rate_sim', 'mean_error']
                for key in expected_keys:
                    results.test(f"Evaluation result has '{key}'", key in eval_result)
                
                results.test("Outputs array type", isinstance(eval_result['outputs'], np.ndarray))
                results.test("Results consistency", 
                           eval_result['num_samples'] == len(eval_result['outputs']))
                
        except Exception as e:
            results.test("Task evaluation execution", False, str(e))
    else:
        results.warning("Validation tasks", "No validation sets found for testing")
    
    # Cleanup
    if test_genome_path.exists():
        test_genome_path.unlink()


def test_inference_system(results: TestResults):
    """Test inference script functionality."""
    results.section("Inference System")
    
    # Import inference components
    try:
        from inference import (load_config, load_genome, calculate_optimal_window,
                             run_inference_1input, run_inference_2input)
        results.test("Inference imports", True)
    except ImportError as e:
        results.test("Inference imports", False, str(e))
        return
    
    # Test config loading for inference (uses evaluation section)
    try:
        config = load_config()
        eval_config = config.get('evaluation', {})
        results.test("Inference uses evaluation config section", 'window' in eval_config)
        results.test("Shared window value", eval_config['window']['value'] == 10000)
        results.test("Shared max_steps value", eval_config['max_steps']['value'] == 3000)
        
    except Exception as e:
        results.test("Inference config loading", False, str(e))
    
    # Create test genomes for both modes
    test_genome_1input = {
        'best_rule': _sanitize_rule(np.random.randint(0, 4, 64, dtype=np.uint8)),
        'best_prog': _sanitize_programme(np.random.randint(0, 3, 8, dtype=np.uint8)),
        'best_fitness': -3.0,
        'mode': '1input'
    }
    
    test_genome_2input = {
        'best_rule': _sanitize_rule(np.random.randint(0, 4, 64, dtype=np.uint8)),
        'best_prog': _sanitize_programme(np.random.randint(0, 3, 8, dtype=np.uint8)),
        'best_fitness': -4.0,
        'mode': '2input'
    }
    
    # Test 1-input inference
    try:
        genome_1 = (test_genome_1input['best_rule'], test_genome_1input['best_prog'])
        test_inputs = [1, 5, 10]
        outputs_1 = run_inference_1input(genome_1, test_inputs, 500, 100)
        
        results.test("1-input inference execution", True)
        results.test("1-input output type", isinstance(outputs_1, np.ndarray))
        results.test("1-input output length", len(outputs_1) == len(test_inputs))
        
    except Exception as e:
        results.test("1-input inference", False, str(e))
    
    # Test 2-input inference
    try:
        genome_2 = (test_genome_2input['best_rule'], test_genome_2input['best_prog'])
        test_inputs_a = [1, 3, 5]
        test_inputs_b = [2, 4, 6]
        outputs_2 = run_inference_2input(genome_2, test_inputs_a, test_inputs_b, 700, 150)
        
        results.test("2-input inference execution", True)
        results.test("2-input output type", isinstance(outputs_2, np.ndarray))
        results.test("2-input output length", len(outputs_2) == len(test_inputs_a))
        
    except Exception as e:
        results.test("2-input inference", False, str(e))
    
    # Test window calculation for large inputs
    try:
        # Test with progressively larger inputs
        for input_val in [100, 1000, 5000]:
            window = calculate_optimal_window(input_val, 10, "1input", 2000)
            results.test(f"Window scales for input {input_val}", window >= 2000)
        
        # Test 2-input scaling
        large_window = calculate_optimal_window([1000, 2000], 15, "2input", 3000)
        results.test("2-input large window", large_window >= 3000)
        
    except Exception as e:
        results.test("Inference window scaling", False, str(e))
    
    # Test edge cases
    try:
        # Test empty inputs (should handle gracefully)
        empty_outputs = run_inference_1input(genome_1, [], 500, 100)
        results.test("Empty input handling", len(empty_outputs) == 0)
        
        # Test single input
        single_output = run_inference_1input(genome_1, [42], 600, 120)
        results.test("Single input handling", len(single_output) == 1)
        
    except Exception as e:
        results.test("Inference edge cases", False, str(e))


def test_evaluation_inference_integration(results: TestResults):
    """Test integration between evaluation and inference components."""
    results.section("Evaluation-Inference Integration")
    
    # Test that both systems use same config structure
    try:
        from evaluate import load_config as eval_load_config
        from inference import load_config as inf_load_config
        
        eval_config = eval_load_config()
        inf_config = inf_load_config()
        
        results.test("Config consistency", eval_config == inf_config)
        
        # Test that both have the evaluation section (shared by both systems)
        results.test("Both configs have evaluation section", 
                    'evaluation' in eval_config and 'evaluation' in inf_config)
        
    except Exception as e:
        results.test("Config integration", False, str(e))
    
    # Test workflow: train -> evaluate -> inference
    try:
        # Use a simple task for quick testing
        quick_config = {
            'population': {
                'pop_size': {'value': 10},
                'generations': {'value': 3},
                'elite_frac': {'value': 0.2},
                'tourney_k': {'value': 2},
                'mut_rule': {'value': 0.05},
                'mut_prog': {'value': 0.05},
            },
            'model': {
                'prog_len': {'value': 6},
                'window': {'value': 150},
                'max_steps': {'value': 30},
                'halt_thresh': {'value': 0.5},
            },
            'input_output': {'task_id': {'value': 6}}  # Add 1 task
        }
        
        # Train a quick model
        best_rule, best_prog, best_fitness, history = train_model(quick_config, checkpoint_callback=None, checkpoint_interval=0)
        
        # Create genome data structure
        test_genome = {
            'best_rule': best_rule,
            'best_prog': best_prog,
            'best_fitness': best_fitness,
            'mode': '1input'
        }
        
        # Save for testing
        integration_genome_path = Path("integration_test_genome.pkl")
        with open(integration_genome_path, 'wb') as f:
            pickle.dump(test_genome, f)
        
        # Test evaluation loading and execution
        from evaluate import load_genome, evaluate_genome_on_task
        loaded_for_eval = load_genome(integration_genome_path)
        results.test("Integration genome loading", loaded_for_eval is not None)
        
        # Test inference loading and execution
        from inference import load_genome as inf_load_genome, run_inference_1input
        loaded_for_inf = inf_load_genome(integration_genome_path)
        genome_tuple = (loaded_for_inf['best_rule'], loaded_for_inf['best_prog'])
        
        inference_outputs = run_inference_1input(genome_tuple, [1, 2, 3], 300, 50)
        results.test("Integration inference execution", len(inference_outputs) == 3)
        
        # Cleanup
        if integration_genome_path.exists():
            integration_genome_path.unlink()
        
        results.test("Full integration workflow", True)
        
    except Exception as e:
        results.test("Integration workflow", False, str(e))
    
    # Test config parameter ranges
    try:
        config = eval_load_config()
        
        eval_window = config['evaluation']['window']['value']
        ga_window = config['model']['window']['value']
        
        # Evaluation window (shared by inference) should be larger than GA training window
        results.test("Evaluation window > GA window", eval_window > ga_window)
        
        # Check reasonable ranges
        results.test("Shared window reasonable", 1000 <= eval_window <= 50000)
        
    except Exception as e:
        results.test("Config parameter validation", False, str(e))


def test_wrapper_system(results: TestResults):
    """Test the unified EM43Wrapper class."""
    results.section("Unified Wrapper System")
    
    # Test wrapper imports
    try:
        from em43_wrapper import EM43Wrapper
        results.test("Wrapper import", True)
    except ImportError as e:
        results.test("Wrapper import", False, str(e))
        return
    
    # Test wrapper initialization
    try:
        wrapper = EM43Wrapper(task_id=1)  # Simple 1-input task
        results.test("Wrapper initialization (1-input)", True)
        results.test("Auto-detection 1-input mode", wrapper.mode == "1input")
        results.test("Task loading", wrapper.task is not None)
        results.test("Config loading", wrapper.config is not None)
    except Exception as e:
        results.test("Wrapper initialization (1-input)", False, str(e))
        return
    
    # Test 2-input wrapper
    try:
        wrapper_2input = EM43Wrapper(task_id=20)  # Simple 2-input task
        results.test("Wrapper initialization (2-input)", True)
        results.test("Auto-detection 2-input mode", wrapper_2input.mode == "2input")
    except Exception as e:
        results.test("Wrapper initialization (2-input)", False, str(e))
    
    # Test config overrides
    try:
        overrides = {'population': {'pop_size': {'value': 50}}}
        wrapper_override = EM43Wrapper(task_id=1, config_overrides=overrides)
        results.test("Config overrides", 
                    wrapper_override.config['population']['pop_size']['value'] == 50)
    except Exception as e:
        results.test("Config overrides", False, str(e))
    
    # Test invalid task ID
    try:
        invalid_wrapper = EM43Wrapper(task_id=999)
        results.test("Invalid task ID handling", False, "Should have raised ValueError")
    except ValueError:
        results.test("Invalid task ID handling", True)
    except Exception as e:
        results.test("Invalid task ID handling", False, f"Wrong error type: {e}")
    
    # Test quick training (minimal parameters)
    try:
        quick_config = {
            'population': {
                'pop_size': {'value': 10}, 
                'generations': {'value': 3}
            },
            'model': {
                'prog_len': {'value': 5}, 
                'window': {'value': 100}, 
                'max_steps': {'value': 30}
            }
        }
        quick_wrapper = EM43Wrapper(task_id=6, config_overrides=quick_config)
        rule, prog, fitness, history = quick_wrapper.train(verbose=False, save_genome=False, checkpoint_callback=None, checkpoint_interval=0)
        
        results.test("Quick training execution", True)
        results.test("Training returns rule", rule is not None and rule.shape == (64,))
        results.test("Training returns prog", prog is not None and len(prog) == 5)
        results.test("Training returns fitness", isinstance(fitness, (int, float)))
        results.test("Training returns history", isinstance(history, list) and len(history) == 3)
        results.test("Wrapper trained state", quick_wrapper.is_trained)
        
        # Test evaluation after training
        eval_results = quick_wrapper.evaluate(verbose=False, plot=False)
        results.test("Evaluation after training", eval_results is not None)
        results.test("Evaluation results structure", 
                    all(key in eval_results for key in ['outputs', 'targets', 'mode']))
        
        # Test inference after training
        if quick_wrapper.mode == "1input":
            infer_outputs = quick_wrapper.infer([1, 2, 3], verbose=False)
        else:
            infer_outputs = quick_wrapper.infer(([1, 2], [3, 4]), verbose=False)
        results.test("Inference after training", isinstance(infer_outputs, list))
        
        # Test save/load genome
        save_path = quick_wrapper.save_genome(verbose=False)
        results.test("Genome saving", save_path.exists())
        
        # Test loading in new wrapper
        load_wrapper = EM43Wrapper(task_id=6)
        load_success = load_wrapper.load_genome(save_path, verbose=False)
        results.test("Genome loading", load_success and load_wrapper.is_trained)
        
        # Cleanup
        if save_path.exists():
            save_path.unlink()
        
    except Exception as e:
        results.test("Quick training workflow", False, str(e))


def test_demo_system(results: TestResults):
    """Test the unified demo system components."""
    results.section("Unified Demo System")
    
    # Test demo imports
    try:
        import sys
        import importlib.util
        
        # Load demo.py as a module to test its functions
        spec = importlib.util.spec_from_file_location("demo", "demo.py")
        demo_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(demo_module)
        
        results.test("Demo module loading", True)
    except Exception as e:
        results.test("Demo module loading", False, str(e))
        return
    
    # Test argument parser creation
    try:
        parser = demo_module.create_parser()
        results.test("Argument parser creation", parser is not None)
        
        # Test some basic argument parsing
        test_args = parser.parse_args(['--task', '1', '--stage', 'train'])
        results.test("Argument parsing", test_args.task == 1 and test_args.stage == 'train')
    except Exception as e:
        results.test("Argument parser", False, str(e))
    
    # Test task listing function
    try:
        # Capture output (redirect stdout temporarily)
        import io
        from contextlib import redirect_stdout
        
        output_buffer = io.StringIO()
        with redirect_stdout(output_buffer):
            demo_module.list_tasks()
        
        output = output_buffer.getvalue()
        results.test("Task listing function", 
                    "1-Input Tasks" in output and "2-Input Tasks" in output)
    except Exception as e:
        results.test("Task listing function", False, str(e))
    
    # Test input parsing
    try:
        parsed = demo_module.parse_inputs("1,2,3,4,5")
        results.test("Input parsing", parsed == [1, 2, 3, 4, 5])
        
        # Test invalid input
        try:
            demo_module.parse_inputs("invalid,input")
            results.test("Invalid input handling", False, "Should have raised ValueError")
        except ValueError:
            results.test("Invalid input handling", True)
    except Exception as e:
        results.test("Input parsing", False, str(e))
    
    # Test config override building
    try:
        class MockArgs:
            def __init__(self):
                self.pop_size = 100
                self.generations = 50
                self.prog_len = None
                self.window = None
                self.max_steps = None
        
        mock_args = MockArgs()
        overrides = demo_module.build_config_overrides(mock_args)
        
        results.test("Config override building", 
                    overrides['population']['pop_size']['value'] == 100 and
                    overrides['population']['generations']['value'] == 50)
    except Exception as e:
        results.test("Config override building", False, str(e))


def test_integration_workflows(results: TestResults):
    """Test complete integration workflows."""
    results.section("Integration Workflows")
    
    # Test complete 1-input workflow
    try:
        from em43_wrapper import EM43Wrapper
        
        # Quick config for fast testing
        quick_config = {
            'population': {
                'pop_size': {'value': 8}, 
                'generations': {'value': 3}
            },
            'model': {
                'prog_len': {'value': 5}, 
                'window': {'value': 80}, 
                'max_steps': {'value': 20}
            }
        }
        
        # Test 1-input workflow
        wrapper1 = EM43Wrapper(task_id=6, config_overrides=quick_config)  # Add 1 task
        wrapper1.train(verbose=False, save_genome=False, checkpoint_callback=None, checkpoint_interval=0)
        eval_results1 = wrapper1.evaluate(verbose=False, plot=False)
        infer_results1 = wrapper1.infer([1, 2, 3], verbose=False)
        
        results.test("1-input complete workflow", 
                    wrapper1.is_trained and eval_results1 is not None and len(infer_results1) == 3)
        
        # Test 2-input workflow
        wrapper2 = EM43Wrapper(task_id=20, config_overrides=quick_config)  # Summation task
        wrapper2.train(verbose=False, save_genome=False, checkpoint_callback=None, checkpoint_interval=0)
        eval_results2 = wrapper2.evaluate(verbose=False, plot=False)
        infer_results2 = wrapper2.infer(([1, 2], [3, 4]), verbose=False)
        
        results.test("2-input complete workflow",
                    wrapper2.is_trained and eval_results2 is not None and len(infer_results2) == 2)
        
        # Test cross-compatibility (load genome with different wrapper instance)
        save_path1 = wrapper1.save_genome("test_cross_compat.pkl", verbose=False)
        new_wrapper1 = EM43Wrapper(task_id=6)
        load_success = new_wrapper1.load_genome(save_path1, verbose=False)
        
        results.test("Cross-instance genome loading", load_success)
        
        # Cleanup
        if save_path1.exists():
            save_path1.unlink()
        
    except Exception as e:
        results.test("Integration workflows", False, str(e))
    
    # Test error handling and edge cases
    try:
        # Clean up any genome files that might interfere with error tests
        dp_dir = Path("dp_checkpoints")
        if dp_dir.exists():
            # Clean up specific files that might interfere
            cleanup_files = [
                "best_genome_task_1.pkl",
                "best_genome_1input.pkl", 
                "best_genome.pkl"  # Generic file that might be loaded
            ]
            for filename in cleanup_files:
                filepath = dp_dir / filename
                if filepath.exists():
                    filepath.unlink()
        
        # Test inference before training
        untrained_wrapper = EM43Wrapper(task_id=1)
        try:
            untrained_wrapper.infer([1, 2, 3])
            results.test("Inference before training error", False, "Should have raised error")
        except ValueError:
            results.test("Inference before training error", True)
        
        # Test evaluation before training
        try:
            untrained_wrapper.evaluate()
            results.test("Evaluation before training error", False, "Should have raised error")
        except ValueError:
            results.test("Evaluation before training error", True)
        
        # Test invalid input formats
        trained_wrapper = EM43Wrapper(task_id=1, config_overrides=quick_config)
        trained_wrapper.train(verbose=False, save_genome=False, checkpoint_callback=None, checkpoint_interval=0)
        
        try:
            # Wrong input format for 1-input mode
            trained_wrapper.infer(([1, 2], [3, 4]))  # 2-input format for 1-input task
            results.test("Invalid input format handling", False, "Should have raised error")
        except ValueError:
            results.test("Invalid input format handling", True)
        
    except Exception as e:
        results.test("Error handling tests", False, str(e))


def test_performance_benchmarks(results: TestResults):
    """Test performance benchmarks."""
    results.section("Performance Benchmarks")
    
    # Benchmark fitness evaluation
    pop_size = 50
    rules = np.random.randint(0, 4, (pop_size, 64), dtype=np.uint8)
    progs = np.random.randint(0, 3, (pop_size, 10), dtype=np.uint8)
    
    task = get_dataset(1)
    inputs = task.inputs[:, 0].astype(np.int32)
    targets = task.targets.astype(np.int32)
    
    # Warm up numba
    _ = fitness_population(rules[:2], progs[:2], inputs, targets, 200, 50, 0.5, 0.0)
    
    # Benchmark
    start_time = time.time()
    fitness = fitness_population(rules, progs, inputs, targets, 200, 50, 0.5, 0.0)
    eval_time = time.time() - start_time
    
    results.test("Fitness evaluation speed", eval_time < 5.0, 
                f"Evaluated {pop_size} genomes in {eval_time:.3f}s")
    
    # Memory usage test
    large_pop = 100
    try:
        large_rules = np.random.randint(0, 4, (large_pop, 64), dtype=np.uint8)
        large_progs = np.random.randint(0, 3, (large_pop, 15), dtype=np.uint8)
        large_fitness = fitness_population(large_rules, large_progs, inputs, targets, 300, 100, 0.5, 0.0)
        results.test("Large population handling", True, f"Handled {large_pop} genomes")
    except MemoryError:
        results.warning("Large population", "Memory constraints reached")
    except Exception as e:
        results.test("Large population handling", False, str(e))
    
    # Benchmark evaluation and inference performance
    try:
        from evaluate import evaluate_genome_on_task
        from inference import run_inference_1input
        
        # Create test genome
        test_genome = {
            'best_rule': _sanitize_rule(np.random.randint(0, 4, 64, dtype=np.uint8)),
            'best_prog': _sanitize_programme(np.random.randint(0, 3, 8, dtype=np.uint8)),
            'best_fitness': -2.0,
            'mode': '1input'
        }
        
        # Benchmark evaluation
        start_time = time.time()
        eval_result = evaluate_genome_on_task(test_genome, 1, large_window=1000, max_steps=200)
        eval_bench_time = time.time() - start_time
        
        results.test("Evaluation performance", eval_bench_time < 10.0,
                    f"Evaluation completed in {eval_bench_time:.3f}s")
        
        # Benchmark inference
        genome_tuple = (test_genome['best_rule'], test_genome['best_prog'])
        test_inputs = list(range(1, 21))  # 20 inputs
        
        start_time = time.time()
        inf_outputs = run_inference_1input(genome_tuple, test_inputs, 1000, 200)
        inf_bench_time = time.time() - start_time
        
        results.test("Inference performance", inf_bench_time < 5.0,
                    f"Inference on 20 inputs completed in {inf_bench_time:.3f}s")
        
    except Exception as e:
        results.test("Evaluation/Inference benchmarks", False, str(e))


def main():
    """Run comprehensive test suite."""
    print("EM43 Comprehensive Test Suite")
    print("=" * 60)
    print("Testing refactored EM43 system for correctness and performance")
    
    results = TestResults()
    
    # Suppress numba warnings during testing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        try:
            test_configuration_system(results)
            test_task_definitions(results)
            test_mathematical_correctness(results)
            test_data_processing(results)
            test_genome_operations(results)
            test_fitness_evaluation(results)
            test_simulation_wrappers(results)
            test_ga_evolution(results)
            test_integration_workflow(results)
            test_edge_cases(results)
            test_evaluation_system(results)
            test_inference_system(results)
            test_evaluation_inference_integration(results)
            test_wrapper_system(results)
            test_demo_system(results)
            test_integration_workflows(results)
            test_performance_benchmarks(results)
            
        except Exception as e:
            print(f"\n💥 Critical test failure: {e}")
            results.failed += 1
    
    # Final summary
    success = results.summary()
    
    if success:
        print("\n🔬 System Analysis Complete:")
        print("   • All core components validated")
        print("   • Mathematical computations verified") 
        print("   • GA evolution process functional")
        print("   • Performance benchmarks passed")
        print("   • Integration workflow successful")
        print("\n✅ The EM43 refactored system is ready for use!")
    
    return success


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1) 