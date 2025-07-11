"""
EM43 Genetic Algorithm - Minimal Implementation
==============================================
Minimal GA implementation supporting both 1-input and 2-input modes.
No advanced features - just core GA functionality.
"""

import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
from em43_numba import _sanitize_rule, _sanitize_programme, fitness_population, fitness_population_two_inputs
from tasks_config import get_dataset, custom_task

# Type alias for checkpoint callback
from typing import Callable, Optional


rng = np.random.default_rng()

class EM43GA:
    """Minimal GA implementation for EM43."""
    
    def __init__(self, config, checkpoint_callback: Optional[Callable[[int, float, np.ndarray, np.ndarray], None]] = None, checkpoint_interval: int = 0):
        """
        Initialize GA with config.
        
        Args:
            config: Configuration dictionary
        """
        
        # Basic GA parameters
        self.pop_size = config['population']['pop_size']['value']
        self.generations = config['population']['generations']['value']
        self.elite_frac = config['population']['elite_frac']['value']
        self.tourney_k = config['population']['tourney_k']['value']
        self.mut_rule = config['population']['mut_rule']['value']
        self.mut_prog = config['population']['mut_prog']['value']
        
        # Model parameters  
        self.prog_len = config['model']['prog_len']['value']
        self.window = config['model']['window']['value']
        self.max_steps = config['model']['max_steps']['value']
        self.halt_thresh = config['model']['halt_thresh']['value']
        
        # Derived parameters
        self.n_elite = int(self.elite_frac * self.pop_size)

        # Checkpoint logging
        self.checkpoint_callback = checkpoint_callback
        self.checkpoint_interval = checkpoint_interval if checkpoint_interval > 0 else None
        
        # Initialize population
        self.pop_rules = np.empty((self.pop_size, 64), np.uint8)
        self.pop_progs = np.empty((self.pop_size, self.prog_len), np.uint8)
        
        # Setup dataset and determine mode
        self._setup_dataset(config)
        
        # Validate configuration
        self._validate_config()
        
        # Initialize population
        self._initialize_population()
        
        print(f"GA initialized: {self.mode} mode, pop={self.pop_size}, gens={self.generations}")
    
    def _setup_dataset(self, config):
        """Load dataset from config and set up input/target arrays."""
        task_id = config['input_output']['task_id']['value']
        
        # Load task (with error handling)
        try:
            task = custom_task if task_id == -1 else get_dataset(task_id)
        except KeyError:
            raise ValueError(f"Task ID {task_id} not found in tasks registry")
        
        self.mode = task.mode
        
        ds_inputs = task.inputs
        ds_targets = task.targets

        if self.mode == "1input":
            # tasks_config.py always provides (N, 1) for 1-input, flatten to (N,)
            self.inputs = ds_inputs[:, 0].astype(np.int32)
            self.targets = ds_targets.astype(np.int32)
        else:  # 2input mode
            # tasks_config.py always provides (N, 2) for 2-input, extract both columns
            self.inputs_a = ds_inputs[:, 0].astype(np.int32)
            self.inputs_b = ds_inputs[:, 1].astype(np.int32)
            self.targets = ds_targets.astype(np.int32)
    
    def _validate_config(self):
        """Validate configuration parameters for simulation requirements."""
        # Calculate minimum window size based on tape structure
        if self.mode == "1input":
            # Tape: [program] 00 0^(max_input+1) R 0... (output space)
            max_input = int(np.max(self.inputs))
            min_window = self.prog_len + 2 + (max_input + 1) + 1 + 20  # +20 for output buffer
        else:  # 2input
            # Tape: [program] 00 0^(max_a+1) R 0^(max_b+1) R 0... (output space)
            max_a = int(np.max(self.inputs_a))
            max_b = int(np.max(self.inputs_b))
            min_window = self.prog_len + 2 + (max_a + 1) + 1 + (max_b + 1) + 1 + 20  # +20 for output buffer
        
        if self.window < min_window:
            raise ValueError(
                f"Window size ({self.window}) is too small for this configuration. "
                f"Minimum required: {min_window} "
                f"(prog_len={self.prog_len}, max_inputs={'%d' % np.max(self.inputs) if self.mode == '1input' else f'{max_a},{max_b}'}, "
                f"mode={self.mode})"
            )
    
    def _initialize_population(self):
        """Initialize population with random genomes."""
        for i in range(self.pop_size):
            rule, prog = self._random_genome()
            self.pop_rules[i] = rule
            self.pop_progs[i] = prog
    
    def _random_genome(self):
        """Generate a random genome."""
        rule = rng.integers(0, 4, 64, dtype=np.uint8)
        prog = rng.choice([0, 1, 2], size=self.prog_len, p=[0.7, 0.2, 0.1])
        return _sanitize_rule(rule), _sanitize_programme(prog)
    
    def _tournament_selection(self, fitness):
        """Simple tournament selection."""
        idx = rng.choice(self.pop_size, self.tourney_k, replace=False)
        best = idx[np.argmax(fitness[idx])]
        return self.pop_rules[best], self.pop_progs[best]
    
    def _crossover(self, rule1, prog1, rule2, prog2):
        """Separate segment crossover for rule and program arrays."""
        def segment_swap(parent_a, parent_b):
            L = len(parent_a)
            max_seg = min(8, L)
            seg_len = rng.integers(1, max_seg + 1)
            start = rng.integers(0, L - seg_len + 1)
            child = parent_a.copy()
            child[start:start+seg_len] = parent_b[start:start+seg_len]
            return child

        child_rule = _sanitize_rule(segment_swap(rule1, rule2))
        child_prog = _sanitize_programme(segment_swap(prog1, prog2))

        return child_rule, child_prog
    
    def _mutate(self, rule, prog):
        """Simple mutation."""
        # Rule mutation
        mask_r = rng.random(64) < self.mut_rule
        if mask_r.any():
            rule = rule.copy()
            rule[mask_r] = rng.integers(0, 4, mask_r.sum(), dtype=np.uint8)
            rule = _sanitize_rule(rule)
        
        # Program mutation
        mask_p = rng.random(self.prog_len) < self.mut_prog
        if mask_p.any():
            prog = prog.copy()
            prog[mask_p] = rng.choice([0, 1, 2], size=mask_p.sum(), p=[0.7, 0.2, 0.1])
            prog = _sanitize_programme(prog)
        
        return rule, prog
    
    def _evaluate_fitness(self):
        """Evaluate fitness for entire population."""
        if self.mode == "1input":
            return fitness_population(
                self.pop_rules, 
                self.pop_progs, 
                self.inputs,
                self.targets,
                self.window,
                self.max_steps,
                self.halt_thresh,
                0.0
            )
        else:  # 2input
            return fitness_population_two_inputs(
                self.pop_rules,
                self.pop_progs,
                self.inputs_a,
                self.inputs_b,
                self.targets,
                self.window,
                self.max_steps,
                self.halt_thresh,
                0.0
            )
    
    def evolve(self):
        """Main evolution loop."""
        best_fitness_history = []
        
        for gen in tqdm(range(1, self.generations + 1), desc="GA Evolution"):
            # Evaluate fitness
            fitness = self._evaluate_fitness()
            
            # Sort by fitness (descending)
            order = np.argsort(fitness)[::-1]
            self.pop_rules = self.pop_rules[order]
            self.pop_progs = self.pop_progs[order]
            fitness = fitness[order]
            
            # Track best fitness
            best_fitness_history.append(float(fitness[0]))

            # Real-time checkpoint logging
            if self.checkpoint_callback and self.checkpoint_interval:
                if gen % self.checkpoint_interval == 0:
                    try:
                        self.checkpoint_callback(gen, float(fitness[0]), self.pop_rules[0].copy(), self.pop_progs[0].copy())
                    except Exception as e:
                        print(f"⚠️  Checkpoint callback error (generation {gen}): {e}")
            
            # Progress reporting
            if gen % 50 == 0 or gen == self.generations:
                tqdm.write(f"Gen {gen:3}: best={fitness[0]:.3f}, mean={fitness.mean():.3f}")
            
            # Create next generation
            next_rules = np.empty_like(self.pop_rules)
            next_progs = np.empty_like(self.pop_progs)
            
            # Elite preservation
            next_rules[:self.n_elite] = self.pop_rules[:self.n_elite]
            next_progs[:self.n_elite] = self.pop_progs[:self.n_elite]
            
            # Generate offspring
            for i in range(self.n_elite, self.pop_size):
                # Selection and crossover
                r1, p1 = self._tournament_selection(fitness)
                r2, p2 = self._tournament_selection(fitness)
                next_rules[i], next_progs[i] = self._crossover(r1, p1, r2, p2)
                
                # Mutation
                next_rules[i], next_progs[i] = self._mutate(next_rules[i], next_progs[i])
            
            # Update population
            self.pop_rules = next_rules
            self.pop_progs = next_progs
         
        # Return best from last generation (already sorted)
        return (self.pop_rules[0], self.pop_progs[0], 
                best_fitness_history[-1], best_fitness_history)
     
    def save_best_genome(self, best_rule, best_prog, best_fitness, save_dir="dp_checkpoints"):
        """Save the best genome."""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        genome_data = {
            "best_rule": best_rule,
            "best_prog": best_prog,
            "best_fitness": best_fitness,
            "mode": self.mode
        }

        suffix = "" if self.mode == "1input" else "_two_input"
        filename = f"best_genome{suffix}.pkl"
        
        with open(save_path / filename, 'wb') as f:
            pickle.dump(genome_data, f)
        
        print(f"Best genome saved to {save_path / filename}")


def train_model(config, checkpoint_callback: Optional[Callable[[int, float], None]] = None, checkpoint_interval: int = 0):
    """
    Train a model using GA.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (best_rule, best_prog, best_fitness, history)
    """
    ga = EM43GA(config, checkpoint_callback=checkpoint_callback, checkpoint_interval=checkpoint_interval)
    best_rule, best_prog, best_fitness, history = ga.evolve()
    ga.save_best_genome(best_rule, best_prog, best_fitness)
    return best_rule, best_prog, best_fitness, history


if __name__ == "__main__":
    # Quick test with default config
    import yaml
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Simple config conversion
    config_dict = {
        section: {k: v['value'] for k, v in params.items()}
        for section, params in config.items() 
        if section != 'config_file'
    }
    
    train_model(config_dict) 