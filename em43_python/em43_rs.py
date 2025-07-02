"""
EM-4/3 Random Search - fast vectorised edition (May 2025)
===========================================================
- Numba-parallel population evaluation: evaluates the whole
  population inside a single nopython `prange` loop - no Python calls
  per genome -> ~5-6 x speed-up on 8-core CPU.
- Pure Random Search: no evolution, just random sampling each generation
- Telemetry: average Hamming distance every
  `N_COMPLEX_TELEMETRY` gens (no plotting).

Drop-in usage: `python em43_demo.py --enable_rs true`

this code has not been checked - may still present unexpected behaviours
"""

from __future__ import annotations
import numpy as np, math, pickle
from pathlib import Path
from tqdm import tqdm
import csv
import uuid
from datetime import datetime
import pandas as pd

from em43_numba import _sanitize_rule, _sanitize_programme, fitness_population

class RandomSearch:
    def __init__(self, args):
        # Population and Search
        self.POP_SIZE = args.pop_size
        self.GENERATIONS = args.generations
        self.PROG_LEN = args.prog_len
        
        # Random search specific
        if args.rs_seed == -1:
            # Use truly random seed (system entropy)
            self.RS_SEED = None
            self.rng = np.random.default_rng()  # No seed = random
            self.actual_seed = "random"  # For logging purposes
        else:
            # Use specified seed for reproducibility
            self.RS_SEED = args.rs_seed
            self.rng = np.random.default_rng(self.RS_SEED)
            self.actual_seed = self.RS_SEED
        
        # Diversity parameters (for logging consistency, but not used in Random Search)
        self.ENABLE_STOCHASTIC_TOURNAMENT = getattr(args, 'enable_stochastic_tournament', 'false').lower() == 'true'
        self.INITIAL_TEMPERATURE = getattr(args, 'initial_temperature', 2.0)
        self.COOLING_RATE = getattr(args, 'cooling_rate', 0.98)
        self.MIN_TEMPERATURE = getattr(args, 'min_temperature', 0.1)
        
        # Regularization and Diversity
        self.LAMBDA_P = args.lambda_p
        self.COMPLEX_TELEMETRY = args.complex_telemetry
        
        # Simulation parameters
        # Evaluate input and target expressions
        self.INPUT_SET = eval(args.input_set, {'np': np})
        self.TARGET_OUT = eval(args.target_out, {'np': np})
        self.WINDOW = args.window
        self.MAX_STEPS = args.max_steps
        self.HALT_THRESH = args.halt_thresh
        
        # Checkpointing
        self.CHECK_EVERY = args.check_every
        self.SAVE_DIR = Path(args.save_dir) if args.save_dir else Path(__file__).parent.parent / Path("dp_checkpoints")
        
        # Task tracking
        self.RUN_LOG_FILE = "log.csv"  # Fixed unified log file
        self.RUN_ID = str(uuid.uuid4())[:8]  # Short unique run ID
        
        # Interactive task selection
        self.TASK_ID, self.TASK_DESCRIPTION = self._interactive_task_selection()
        
        # Initialize population arrays
        self.pop_rules = np.empty((self.POP_SIZE, 64), np.uint8)
        self.pop_progs = np.empty((self.POP_SIZE, self.PROG_LEN), np.uint8)
        
        # Track global best across all generations
        self.global_best_rule = None
        self.global_best_prog = None
        self.global_best_fitness = -np.inf
        
        # Initialize history
        self.best_curve = []
        self.mean_curve = []
        
        # Initialize CSV logging
        self._initialize_csv_log()
        self.initial_fitness = None  # Will be set after first evaluation

    # ───────────────── Task Selection methods ──────────────────────────
    def _load_task_descriptions(self):
        """Load task descriptions from CSV file."""
        csv_path = Path("task_descriptions.csv")
        if not csv_path.exists():
            # Create default CSV if it doesn't exist
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['task_id', 'task_description'])
                writer.writerow([0, 'undefined'])
                writer.writerow([1, 'multiply by 2'])
                writer.writerow([2, 'multiply by 3'])
        
        try:
            df = pd.read_csv(csv_path)
            # Convert to dict for easy lookup
            return dict(zip(df['task_id'].astype(int), df['task_description']))
        except Exception as e:
            print(f"Error reading task descriptions: {e}")
            # Return default tasks
            return {0: 'undefined', 1: 'multiply by 2', 2: 'multiply by 3'}
    
    def _interactive_task_selection(self):
        """Interactively ask user to select a task."""
        task_dict = self._load_task_descriptions()
        
        print("\n" + "=" * 60)
        print("TASK SELECTION")
        print("=" * 60)
        print("Available tasks:")
        for task_id, description in sorted(task_dict.items()):
            print(f"  {task_id}: {description}")
        print()
        
        while True:
            try:
                user_input = input("Enter task ID (or press Enter for default task 0): ").strip()
                
                if user_input == "":
                    # Default to task 0
                    task_id = 0
                    task_description = task_dict.get(0, 'undefined')
                    print(f"Defaulting to task 0: '{task_description}'")
                    return str(task_id), task_description
                
                # Try to parse as integer
                task_id = int(user_input)
                
                if task_id in task_dict:
                    task_description = task_dict[task_id]
                    print(f"\nSelected task {task_id}: '{task_description}'")
                    
                    # Ask for confirmation
                    confirm = input("Confirm this task? (y/N): ").strip().lower()
                    if confirm.startswith('y'):
                        return str(task_id), task_description
                    else:
                        print("Task selection cancelled. Please select again.\n")
                        continue
                else:
                    print(f"Task ID {task_id} not found. Available IDs: {list(task_dict.keys())}")
                    continue
                    
            except ValueError:
                print("Please enter a valid integer task ID or press Enter for default.")
                continue
            except KeyboardInterrupt:
                print("\nCancelled. Defaulting to task 0: 'undefined'")
                return "0", "undefined"

    # ───────────────── CSV Logging methods ─────────────────────────────
    def _initialize_csv_log(self):
        """Initialize CSV log file with headers if it doesn't exist."""
        log_path = Path(self.RUN_LOG_FILE)
        file_exists = log_path.exists()
        
        if not file_exists:
            with open(log_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['datetime_utc', 'run_id', 'checkpoint', 'task_id', 'task_description', 
                               'initial_fitness', 'final_fitness', 'program', 'rule'])
    
    def _log_run_data(self, checkpoint: int, final_fitness: float, best_rule: np.ndarray, best_prog: np.ndarray):
        """Log run data to CSV file."""
        log_path = Path(self.RUN_LOG_FILE)
        
        # Get current UTC timestamp
        utc_timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        
        # Convert arrays to comma-separated strings
        rule_str = ','.join(map(str, best_rule.tolist()))
        prog_str = ','.join(map(str, best_prog.tolist()))
        
        with open(log_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                utc_timestamp,
                self.RUN_ID,
                checkpoint,
                self.TASK_ID,
                self.TASK_DESCRIPTION,
                self.initial_fitness if self.initial_fitness is not None else 'N/A',
                final_fitness,
                prog_str,
                rule_str
            ])

    # ───────────────── Random Search methods ───────────────────────────
    def random_genome(self) -> tuple[np.ndarray, np.ndarray]:
        rule = self.rng.integers(0, 4, 64, dtype=np.uint8)
        prog = self.rng.choice([0, 1, 2], size=self.PROG_LEN, p=[0.7, 0.2, 0.1])
        return _sanitize_rule(rule), _sanitize_programme(prog)

    def generate_random_population(self):
        """Generate a completely new random population."""
        for i in range(self.POP_SIZE):
            r, p = self.random_genome()
            self.pop_rules[i], self.pop_progs[i] = r, p

    def avg_pairwise_hamming(self, flat: np.ndarray) -> float:
        P = flat.shape[0]
        total = 0
        for i in range(P-1):
            diff = np.count_nonzero(flat[i+1:] != flat[i], axis=1)
            total += diff.sum()
        return total / (P*(P-1)//2)

    # ───────────────── Random Search main loop ─────────────────────────
    def run(self):
        """
        Run Random Search.

        Returns:
        --------
        tuple[np.ndarray, np.ndarray, float]
            (best_rule, best_prog, best_fitness)
        """
        print(f"Starting Random Search with ID: {self.RUN_ID}")
        print(f"Task: {self.TASK_ID} - {self.TASK_DESCRIPTION}")
        if self.RS_SEED is None:
            print(f"Random seed: random (system entropy)")
        else:
            print(f"Random seed: {self.RS_SEED} (reproducible)")
        print(f"Logging to: {self.RUN_LOG_FILE}")
        print(f"UTC Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Note about diversity parameters (not applicable to Random Search)
        if self.ENABLE_STOCHASTIC_TOURNAMENT:
            print(f"Note: Stochastic Tournament settings detected but not used in Random Search")

        for gen in tqdm(range(1, self.GENERATIONS+1), ncols=80, desc="Random Search"):
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

            if gen % self.COMPLEX_TELEMETRY == 0:
                flat = np.concatenate((self.pop_rules, self.pop_progs), axis=1)
                ham = self.avg_pairwise_hamming(flat)
                tqdm.write(f"Gen {gen:3}  best={self.global_best_fitness:.3f}  current_best={current_best_fitness:.3f}  mean={fit.mean():.3f}  ham={ham:.1f}")
            else:
                tqdm.write(f"Gen {gen:3}  best={self.global_best_fitness:.3f}  current_best={current_best_fitness:.3f}  mean={fit.mean():.3f}")

            # Check-point
            if gen % self.CHECK_EVERY == 0 or gen == self.GENERATIONS:
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
                
                # Log run data to CSV
                self._log_run_data(gen, self.global_best_fitness, self.global_best_rule, self.global_best_prog)
                tqdm.write(f"Logged checkpoint {gen} data to {self.RUN_LOG_FILE}")

        # Save the best genome
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