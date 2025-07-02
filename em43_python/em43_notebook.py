"""
EM43 Notebook API - Simplified interface for Jupyter notebooks and statistical analysis
====================================================================================

This module provides a clean, notebook-friendly API for EM43 that:
- No command-line arguments required
- Separate control of training, inference, evaluation
- Suitable for statistical analysis and parameter sweeps
- Clean programmatic interface

Usage:
    trainer = EM43Trainer(method='ga', generations=100, pop_size=1000)
    best_genome, fitness = trainer.train(inputs=[1,2,3], targets=[2,4,6])
    outputs = trainer.infer([4,5,6])
    metrics = trainer.evaluate([4,5,6], [8,10,12])
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import warnings

# Import the core functionality
from em43_numba import _sanitize_rule, _sanitize_programme, fitness_population, EM43Batch
from em43_ga import GenomeAlgorithm
from em43_rs import RandomSearch

class SimpleArgs:
    """Simple args object to replace argparse dependency."""
    def __init__(self, **kwargs):
        # Set all the defaults from config.yaml
        defaults = {
            # Population
            'pop_size': 5000,
            'generations': 100, 
            'elite_frac': 0.1,
            'tourney_k': 3,
            
            # Mutation
            'mut_rule': 0.03,
            'mut_prog': 0.08,
            'prog_len': 10,
            
            # Regularization  
            'lambda_p': 0.01,
            'random_immigrants': 0.2,
            'complex_telemetry': 30,
            
            # Simulation
            'window': 100,
            'max_steps': 300,
            'halt_thresh': 0.50,
            
            # Checkpointing
            'check_every': 100,
            'save_dir': 'dp_checkpoints',
            
            # Random Search
            'enable_rs': 'false',
            'rs_seed': -1,
            
            # Diversity (new stochastic tournament parameters)
            'enable_stochastic_tournament': 'false',
            'initial_temperature': 2.0,
            'cooling_rate': 0.98,
            'min_temperature': 0.1,
            
            # Input/Output (will be set programmatically)
            'input_set': 'np.arange(1, 11)',
            'target_out': 'np.arange(1, 11) * 2'
        }
        
        # Update with provided kwargs
        for key, value in defaults.items():
            setattr(self, key, kwargs.get(key, value))
        
        # Override with any additional kwargs
        for key, value in kwargs.items():
            if key not in defaults:
                setattr(self, key, value)

class EM43Trainer:
    """
    Simplified trainer for notebook use.
    
    Provides separate control of training, inference, and evaluation.
    """
    
    def __init__(self, 
                 method: str = 'ga',
                 generations: int = 100,
                 pop_size: int = 5000,
                 prog_len: int = 10,
                 seed: Optional[int] = None,
                 lambda_p: float = 0.01,
                 verbose: bool = True,
                 enable_logging: bool = False,
                 # Diversity parameters for stochastic tournament
                 enable_stochastic_tournament: bool = False,
                 initial_temperature: float = 2.0,
                 cooling_rate: float = 0.98,
                 min_temperature: float = 0.1,
                 **kwargs):
        """
        Initialize the trainer.
        
        Args:
            method: 'ga' or 'random_search'
            generations: Number of generations to run
            pop_size: Population size
            prog_len: Program length
            seed: Random seed (None for random, int for reproducible)
            lambda_p: Sparsity penalty coefficient
            verbose: Print progress information
            enable_logging: Enable CSV logging (default: False for notebooks)
            enable_stochastic_tournament: Enable stochastic tournament selection for diversity
            initial_temperature: Initial temperature for stochastic selection (higher = less greedy)
            cooling_rate: Temperature cooling rate per generation
            min_temperature: Minimum temperature to maintain diversity
            **kwargs: Additional parameters passed to the algorithm
        """
        self.method = method
        self.verbose = verbose
        self.enable_logging = enable_logging
        
        # Store parameters
        self.params = {
            'generations': generations,
            'pop_size': pop_size,
            'prog_len': prog_len,
            'lambda_p': lambda_p,
            # Diversity parameters
            'enable_stochastic_tournament': 'true' if enable_stochastic_tournament else 'false',
            'initial_temperature': initial_temperature,
            'cooling_rate': cooling_rate,
            'min_temperature': min_temperature,
            **kwargs
        }
        
        # Handle seeding
        if method == 'random_search':
            self.params['enable_rs'] = 'true'
            self.params['rs_seed'] = seed if seed is not None else -1
        
        # Storage for results
        self.best_rule = None
        self.best_prog = None
        self.best_fitness = None
        self.fitness_curve = None
        
    def train(self, 
              inputs: List[int], 
              targets: List[int],
              task_id: str = "notebook_task",
              task_description: str = "Notebook training") -> Tuple[Tuple[np.ndarray, np.ndarray], float]:
        """
        Train the model.
        
        Args:
            inputs: List of input values
            targets: List of target output values  
            task_id: Task identifier for logging
            task_description: Task description for logging
            
        Returns:
            ((rule, program), fitness): Best genome and its fitness
        """
        if len(inputs) != len(targets):
            raise ValueError("inputs and targets must have same length")
        
        # Convert to numpy expressions for the algorithms
        inputs_expr = f"np.array({inputs})"
        targets_expr = f"np.array({targets})"
        
        # Create args object
        args = SimpleArgs(
            input_set=inputs_expr,
            target_out=targets_expr,
            **self.params
        )
        
        if self.verbose:
            print(f"Training {self.method.upper()} for {args.generations} generations...")
            print(f"Population: {args.pop_size}, Program length: {args.prog_len}")
            print(f"Inputs: {inputs}")
            print(f"Targets: {targets}")
        
        # Train based on method
        if self.method == 'ga':
            trainer = GenomeAlgorithmNoUI(args, task_id, task_description, self.enable_logging)
        elif self.method == 'random_search':
            trainer = RandomSearchNoUI(args, task_id, task_description, self.enable_logging)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Run training
        self.best_rule, self.best_prog, self.best_fitness = trainer.run()
        self.fitness_curve = trainer.best_curve
        
        if self.verbose:
            print(f"Training complete! Best fitness: {self.best_fitness:.3f}")
        
        return (self.best_rule, self.best_prog), self.best_fitness
    
    def infer(self, inputs: List[int]) -> List[int]:
        """
        Run inference on the trained model.
        
        Args:
            inputs: List of input values
            
        Returns:
            List of output values
        """
        if self.best_rule is None or self.best_prog is None:
            raise ValueError("Must train model first or load genome")
        
        # Use EM43Batch for inference
        simulator = EM43Batch(
            (self.best_rule, self.best_prog),
            window=self.params.get('window', 500),
            max_steps=self.params.get('max_steps', 800),
            halt_thresh=self.params.get('halt_thresh', 0.50)
        )
        
        outputs = simulator.run(inputs)
        
        if self.verbose:
            print(f"Inference: {inputs} → {outputs.tolist()}")
        
        return outputs.tolist()
    
    def evaluate(self, inputs: List[int], targets: List[int]) -> Dict[str, float]:
        """
        Evaluate the model performance.
        
        Args:
            inputs: List of input values
            targets: List of expected output values
            
        Returns:
            Dictionary with evaluation metrics
        """
        outputs = self.infer(inputs)
        outputs = np.array(outputs)
        targets = np.array(targets)
        
        # Calculate metrics
        errors = np.abs(outputs - targets)
        avg_error = float(errors.mean())
        max_error = float(errors.max())
        success_rate = float((errors < 0.1).mean())
        accuracy = float((outputs == targets).mean())
        
        metrics = {
            'avg_error': avg_error,
            'max_error': max_error, 
            'success_rate': success_rate,
            'accuracy': accuracy,
            'fitness': self.best_fitness
        }
        
        if self.verbose:
            print(f"Evaluation metrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.3f}")
        
        return metrics
    
    def load_genome(self, rule: np.ndarray, program: np.ndarray, fitness: Optional[float] = None):
        """
        Load a pre-trained genome.
        
        Args:
            rule: Rule array (64,)
            program: Program array  
            fitness: Optional fitness value
        """
        self.best_rule = rule.copy()
        self.best_prog = program.copy()
        self.best_fitness = fitness
        
        if self.verbose:
            print(f"Loaded genome: rule shape {rule.shape}, program shape {program.shape}")
    
    def get_fitness_curve(self) -> Optional[List[float]]:
        """Get the fitness curve from training."""
        return self.fitness_curve


class GenomeAlgorithmNoUI(GenomeAlgorithm):
    """GenomeAlgorithm without interactive UI for notebook use."""
    
    def __init__(self, args, task_id: str, task_description: str, enable_logging: bool):
        self.enable_logging = enable_logging
        self.task_id = task_id
        self.task_description = task_description
        
        # Set attributes without calling parent __init__ to avoid UI
        self.POP_SIZE = args.pop_size
        self.GENERATIONS = args.generations
        self.ELITE_FRAC = args.elite_frac
        self.TOURNEY_K = args.tourney_k
        self.P_MUT_RULE = args.mut_rule
        self.P_MUT_PROG = args.mut_prog
        self.PROG_LEN = args.prog_len
        self.LAMBDA_P = args.lambda_p
        self.RANDOM_IMMIGRANTS = args.random_immigrants
        self.COMPLEX_TELEMETRY = args.complex_telemetry
        
        # Stochastic Tournament and Temperature Cooling
        self.ENABLE_STOCHASTIC_TOURNAMENT = getattr(args, 'enable_stochastic_tournament', 'false').lower() == 'true'
        self.INITIAL_TEMPERATURE = getattr(args, 'initial_temperature', 2.0)
        self.COOLING_RATE = getattr(args, 'cooling_rate', 0.98)
        self.MIN_TEMPERATURE = getattr(args, 'min_temperature', 0.1)
        
        # Simulation parameters
        self.INPUT_SET = eval(args.input_set, {'np': np})
        self.TARGET_OUT = eval(args.target_out, {'np': np})
        self.WINDOW = args.window
        self.MAX_STEPS = args.max_steps
        self.HALT_THRESH = args.halt_thresh
        
        # Set task info without UI
        self.TASK_ID = task_id
        self.TASK_DESCRIPTION = task_description
        
        # Required attributes for compatibility with parent run() method
        import uuid
        from pathlib import Path
        self.RUN_ID = str(uuid.uuid4())[:8]
        self.RUN_LOG_FILE = "log.csv"
        self.CHECK_EVERY = args.check_every
        self.SAVE_DIR = Path(args.save_dir) if args.save_dir else Path(__file__).parent.parent / Path("dp_checkpoints")
        
        # Initialize other attributes
        self.N_ELITE = int(np.ceil(self.ELITE_FRAC * self.POP_SIZE))
        self.N_IMM = max(1, int(self.RANDOM_IMMIGRANTS * self.POP_SIZE))
        
        self.pop_rules = np.empty((self.POP_SIZE, 64), np.uint8)
        self.pop_progs = np.empty((self.POP_SIZE, self.PROG_LEN), np.uint8)
        
        self.best_curve = []
        self.mean_curve = []
        self.initial_fitness = None
        
        # Initialize CSV logging only if enabled
        if self.enable_logging:
            self._initialize_csv_log()
    
    def _initialize_csv_log(self):
        """Override to only initialize if logging is enabled."""
        if self.enable_logging:
            super()._initialize_csv_log()
    
    def _log_run_data(self, checkpoint: int, final_fitness: float, best_rule: np.ndarray, best_prog: np.ndarray):
        """Override to only log if logging is enabled."""
        if self.enable_logging:
            super()._log_run_data(checkpoint, final_fitness, best_rule, best_prog)
    
    def run(self):
        """Override run method for cleaner notebook output."""
        # Minimal startup message
        if self.enable_logging:
            print(f"Starting GA training (ID: {self.RUN_ID})")
        
        # Initialize population
        for i in range(self.POP_SIZE):
            r, p = self.random_genome()
            self.pop_rules[i], self.pop_progs[i] = r, p

        # Import here to avoid issues
        from tqdm import tqdm
        import pickle

        for gen in tqdm(range(1, self.GENERATIONS+1), ncols=80, desc="GA", disable=not hasattr(self, 'verbose') or not getattr(self, 'verbose', True)):
            fit = fitness_population(
                self.pop_rules, 
                self.pop_progs, 
                self.INPUT_SET, 
                self.TARGET_OUT,
                self.WINDOW,
                self.MAX_STEPS,
                self.HALT_THRESH,
                self.LAMBDA_P
            )
            order = np.argsort(fit)[::-1]
            self.pop_rules, self.pop_progs, fit = self.pop_rules[order], self.pop_progs[order], fit[order]

            # Store initial fitness from first generation
            if self.initial_fitness is None:
                self.initial_fitness = float(fit[0])

            self.best_curve.append(float(fit[0]))
            self.mean_curve.append(float(fit.mean()))

            # Only log checkpoints if logging is enabled
            if self.enable_logging and (gen % self.CHECK_EVERY == 0 or gen == self.GENERATIONS):
                chk = {
                    "gen": gen,
                    "best_rule": self.pop_rules[0],
                    "best_prog": self.pop_progs[0],
                    "best_fitness": float(fit[0]),
                    "mean_fitness": float(fit.mean()),
                    "best_curve": self.best_curve,
                    "mean_curve": self.mean_curve,
                    "target_out": self.TARGET_OUT,
                }
                
                chk_path = self.SAVE_DIR / f"checkpoint_gen_{gen}.pkl"
                chk_path.parent.mkdir(exist_ok=True)
                with open(chk_path, 'wb') as f:
                    pickle.dump(chk, f)
                
                self._log_run_data(gen, float(fit[0]), self.pop_rules[0], self.pop_progs[0])

            # Evolution operations (same as parent)
            next_rules = np.empty_like(self.pop_rules)
            next_progs = np.empty_like(self.pop_progs)

            # Elite preservation
            next_rules[:self.N_ELITE] = self.pop_rules[:self.N_ELITE]
            next_progs[:self.N_ELITE] = self.pop_progs[:self.N_ELITE]

            # Fill remaining with tournament selection and crossover
            for i in range(self.N_ELITE, self.POP_SIZE - self.N_IMM):
                r1, p1 = self.select_parent(self.pop_rules, self.pop_progs, fit, gen)
                r2, p2 = self.select_parent(self.pop_rules, self.pop_progs, fit, gen)
                next_rules[i], next_progs[i] = self.crossover(r1, p1, r2, p2)

            # Mutation
            for i in range(self.N_ELITE, self.POP_SIZE):
                next_rules[i], next_progs[i] = self.mutate(next_rules[i], next_progs[i])

            # Random immigrants
            for _ in range(self.N_IMM):
                idx = np.random.randint(self.N_ELITE, self.POP_SIZE)
                next_rules[idx], next_progs[idx] = self.random_genome()

            self.pop_rules, self.pop_progs = next_rules, next_progs

        # Save the best genome only if logging is enabled
        if self.enable_logging:
            best = {
                "gen": self.GENERATIONS,
                "best_rule": self.pop_rules[0],
                "best_prog": self.pop_progs[0],
                "best_fitness": float(fit[0]),
                "mean_fitness": float(fit.mean()),
                "best_curve": self.best_curve,
                "mean_curve": self.mean_curve,
                "target_out": self.TARGET_OUT,
            }
            with open(self.SAVE_DIR / "best_genome.pkl", "wb") as f:
                pickle.dump(best, f)
        
        return self.pop_rules[0], self.pop_progs[0], float(fit[0])


class RandomSearchNoUI(RandomSearch):
    """RandomSearch without interactive UI for notebook use."""
    
    def __init__(self, args, task_id: str, task_description: str, enable_logging: bool):
        self.enable_logging = enable_logging
        self.task_id = task_id  
        self.task_description = task_description
        
        # Set attributes without calling parent __init__ to avoid UI
        self.POP_SIZE = args.pop_size
        self.GENERATIONS = args.generations
        self.PROG_LEN = args.prog_len
        
        # Random search seeding
        if args.rs_seed == -1:
            self.RS_SEED = None
            self.rng = np.random.default_rng()
            self.actual_seed = "random"
        else:
            self.RS_SEED = args.rs_seed
            self.rng = np.random.default_rng(self.RS_SEED)
            self.actual_seed = self.RS_SEED
        
        self.LAMBDA_P = args.lambda_p
        self.COMPLEX_TELEMETRY = args.complex_telemetry
        
        # Diversity parameters (for consistency, but not used in Random Search)
        self.ENABLE_STOCHASTIC_TOURNAMENT = getattr(args, 'enable_stochastic_tournament', 'false').lower() == 'true'
        self.INITIAL_TEMPERATURE = getattr(args, 'initial_temperature', 2.0)
        self.COOLING_RATE = getattr(args, 'cooling_rate', 0.98)
        self.MIN_TEMPERATURE = getattr(args, 'min_temperature', 0.1)
        
        # Simulation parameters
        self.INPUT_SET = eval(args.input_set, {'np': np})
        self.TARGET_OUT = eval(args.target_out, {'np': np})
        self.WINDOW = args.window
        self.MAX_STEPS = args.max_steps
        self.HALT_THRESH = args.halt_thresh
        
        # Set task info without UI
        self.TASK_ID = task_id
        self.TASK_DESCRIPTION = task_description
        
        # Required attributes for compatibility with parent run() method
        import uuid
        from pathlib import Path
        self.RUN_ID = str(uuid.uuid4())[:8]
        self.RUN_LOG_FILE = "log.csv"
        self.CHECK_EVERY = args.check_every
        self.SAVE_DIR = Path(args.save_dir) if args.save_dir else Path(__file__).parent.parent / Path("dp_checkpoints")
        
        # Initialize other attributes
        self.pop_rules = np.empty((self.POP_SIZE, 64), np.uint8)
        self.pop_progs = np.empty((self.POP_SIZE, self.PROG_LEN), np.uint8)
        
        self.global_best_rule = None
        self.global_best_prog = None
        self.global_best_fitness = -np.inf
        
        self.best_curve = []
        self.mean_curve = []
        self.initial_fitness = None
        
        # Initialize CSV logging only if enabled
        if self.enable_logging:
            self._initialize_csv_log()
    
    def _initialize_csv_log(self):
        """Override to only initialize if logging is enabled."""
        if self.enable_logging:
            super()._initialize_csv_log()
    
    def _log_run_data(self, checkpoint: int, final_fitness: float, best_rule: np.ndarray, best_prog: np.ndarray):
        """Override to only log if logging is enabled."""
        if self.enable_logging:
            super()._log_run_data(checkpoint, final_fitness, best_rule, best_prog)
    
    def run(self):
        """Override run method for cleaner notebook output."""
        # Minimal startup message
        if self.enable_logging:
            print(f"Starting Random Search training (ID: {self.RUN_ID})")

        # Import here to avoid issues
        from tqdm import tqdm
        import pickle

        for gen in tqdm(range(1, self.GENERATIONS+1), ncols=80, desc="Random Search", disable=not hasattr(self, 'verbose') or not getattr(self, 'verbose', True)):
            # Generate completely new random population each generation
            self.generate_random_population()
            
            # Evaluate population in parallel
            fit = fitness_population(
                self.pop_rules, 
                self.pop_progs, 
                self.INPUT_SET, 
                self.TARGET_OUT,
                self.WINDOW,
                self.MAX_STEPS,
                self.HALT_THRESH,
                self.LAMBDA_P
            )
            
            # Find best in current generation
            best_idx = np.argmax(fit)
            current_best_fitness = float(fit[best_idx])
            
            # Update global best if current generation has better fitness
            if current_best_fitness > self.global_best_fitness:
                self.global_best_fitness = current_best_fitness
                self.global_best_rule = self.pop_rules[best_idx].copy()
                self.global_best_prog = self.pop_progs[best_idx].copy()

            # Store initial fitness from first generation
            if self.initial_fitness is None:
                self.initial_fitness = current_best_fitness

            # Track fitness curves using global best
            self.best_curve.append(self.global_best_fitness)
            self.mean_curve.append(float(fit.mean()))

            # Only log checkpoints if logging is enabled
            if self.enable_logging and (gen % self.CHECK_EVERY == 0 or gen == self.GENERATIONS):
                chk = {
                    "gen": gen,
                    "best_rule": self.global_best_rule,
                    "best_prog": self.global_best_prog,
                    "best_fitness": self.global_best_fitness,
                    "mean_fitness": float(fit.mean()),
                    "best_curve": self.best_curve,
                    "mean_curve": self.mean_curve,
                    "target_out": self.TARGET_OUT,
                }
                
                chk_path = self.SAVE_DIR / f"checkpoint_gen_{gen}.pkl"
                chk_path.parent.mkdir(exist_ok=True)
                with open(chk_path, 'wb') as f:
                    pickle.dump(chk, f)
                
                self._log_run_data(gen, self.global_best_fitness, self.global_best_rule, self.global_best_prog)

        # Save the best genome only if logging is enabled
        if self.enable_logging:
            best = {
                "gen": self.GENERATIONS,
                "best_rule": self.global_best_rule,
                "best_prog": self.global_best_prog,
                "best_fitness": self.global_best_fitness,
                "mean_fitness": float(fit.mean()),
                "best_curve": self.best_curve,
                "mean_curve": self.mean_curve,
                "target_out": self.TARGET_OUT,
            }
            with open(self.SAVE_DIR / "best_genome.pkl", "wb") as f:
                pickle.dump(best, f)
        
        return self.global_best_rule, self.global_best_prog, self.global_best_fitness


# Quick utility functions for common tasks
def quick_train_ga(inputs: List[int], targets: List[int], generations: int = 50, enable_stochastic_tournament: bool = False, **kwargs) -> Tuple[Tuple[np.ndarray, np.ndarray], float]:
    """Quick GA training with minimal setup."""
    trainer = EM43Trainer(method='ga', generations=generations, verbose=False, 
                         enable_stochastic_tournament=enable_stochastic_tournament, **kwargs)
    return trainer.train(inputs, targets)

def quick_train_rs(inputs: List[int], targets: List[int], generations: int = 50, **kwargs) -> Tuple[Tuple[np.ndarray, np.ndarray], float]:
    """Quick Random Search training with minimal setup."""
    trainer = EM43Trainer(method='random_search', generations=generations, verbose=False, **kwargs)
    return trainer.train(inputs, targets)

def compare_methods(inputs: List[int], targets: List[int], generations: int = 50, n_runs: int = 5) -> Dict[str, List[float]]:
    """Compare GA vs Random Search across multiple runs."""
    results = {'ga': [], 'rs': []}
    
    print(f"Comparing GA vs Random Search ({n_runs} runs each)...")
    
    for i in range(n_runs):
        # GA
        _, ga_fitness = quick_train_ga(inputs, targets, generations, seed=i)
        results['ga'].append(ga_fitness)
        
        # Random Search  
        _, rs_fitness = quick_train_rs(inputs, targets, generations, seed=i)
        results['rs'].append(rs_fitness)
        
        print(f"Run {i+1}: GA={ga_fitness:.3f}, RS={rs_fitness:.3f}")
    
    print(f"Average: GA={np.mean(results['ga']):.3f}, RS={np.mean(results['rs']):.3f}")
    return results

def compare_diversity_methods(inputs: List[int], targets: List[int], generations: int = 50, n_runs: int = 5, **kwargs) -> Dict[str, List[float]]:
    """Compare GA with standard tournament vs stochastic tournament across multiple runs."""
    results = {'standard_ga': [], 'stochastic_ga': []}
    
    print(f"Comparing Standard GA vs Stochastic Tournament GA ({n_runs} runs each)...")
    
    for i in range(n_runs):
        # Standard GA
        _, standard_fitness = quick_train_ga(inputs, targets, generations, seed=i, 
                                           enable_stochastic_tournament=False, **kwargs)
        results['standard_ga'].append(standard_fitness)
        
        # Stochastic Tournament GA
        _, stochastic_fitness = quick_train_ga(inputs, targets, generations, seed=i, 
                                             enable_stochastic_tournament=True, **kwargs)
        results['stochastic_ga'].append(stochastic_fitness)
        
        print(f"Run {i+1}: Standard={standard_fitness:.3f}, Stochastic={stochastic_fitness:.3f}")
    
    print(f"Average: Standard GA={np.mean(results['standard_ga']):.3f}, Stochastic GA={np.mean(results['stochastic_ga']):.3f}")
    return results 