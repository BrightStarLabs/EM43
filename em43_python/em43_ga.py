"""
EM-4/3 GA - fast vectorised edition (May 2025)
===================================================
- Numba-parallel population evaluation: evaluates the whole
  population inside a single nopython `prange` loop - no Python calls
  per genome -> ~5-6 x speed-up on 8-core CPU.
- Random-Immigrant Strategy: unchanged (keeps diversity).
- Telemetry: average Hamming distance every
  `N_COMPLEX_TELEMETRY` gens (no plotting).

Drop-in usage: `python em43_numba_ga.py`

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

rng = np.random.default_rng()

class GenomeAlgorithm:
    def __init__(self, args):
        # Population and Evolution
        self.POP_SIZE = args.pop_size
        self.GENERATIONS = args.generations
        self.ELITE_FRAC = args.elite_frac
        self.TOURNEY_K = args.tourney_k
        
        # Mutation probabilities
        self.P_MUT_RULE = args.mut_rule
        self.P_MUT_PROG = args.mut_prog
        self.PROG_LEN = args.prog_len
        
        # Regularization and Diversity
        self.LAMBDA_P = args.lambda_p
        self.RANDOM_IMMIGRANTS = args.random_immigrants
        self.COMPLEX_TELEMETRY = args.complex_telemetry
        
        # Stochastic Tournament and Temperature Cooling
        self.ENABLE_STOCHASTIC_TOURNAMENT = getattr(args, 'enable_stochastic_tournament', 'false').lower() == 'true'
        self.INITIAL_TEMPERATURE = getattr(args, 'initial_temperature', 2.0)
        self.COOLING_RATE = getattr(args, 'cooling_rate', 0.98)
        self.MIN_TEMPERATURE = getattr(args, 'min_temperature', 0.1)
        
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
        
        # Derived parameters
        self.N_ELITE = int(math.ceil(self.ELITE_FRAC * self.POP_SIZE))
        self.N_IMM = max(1, int(self.RANDOM_IMMIGRANTS * self.POP_SIZE))
        
        # Initialize population arrays
        self.pop_rules = np.empty((self.POP_SIZE, 64), np.uint8)
        self.pop_progs = np.empty((self.POP_SIZE, self.PROG_LEN), np.uint8)
        
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

    # ───────────────── GA methods ──────────────────────────────────────
    def random_genome(self) -> tuple[np.ndarray, np.ndarray]:
        rule = rng.integers(0, 4, 64, dtype=np.uint8)
        prog = rng.choice([0, 1, 2], size=self.PROG_LEN, p=[0.7, 0.2, 0.1])
        return _sanitize_rule(rule), _sanitize_programme(prog)

    def tournament(self, pop_rules, pop_progs, fit):
        idx = rng.choice(self.POP_SIZE, self.TOURNEY_K, replace=False)
        best = idx[np.argmax(fit[idx])]
        return pop_rules[best], pop_progs[best]
    
    def stochastic_tournament(self, pop_rules, pop_progs, fit, temperature=1.0):
        """
        Stochastic tournament selection with temperature control.
        
        Args:
            pop_rules: Population rule arrays
            pop_progs: Population program arrays 
            fit: Fitness values
            temperature: Selection temperature (higher = less greedy)
            
        Returns:
            Selected rule and program arrays
        """
        idx = rng.choice(self.POP_SIZE, self.TOURNEY_K, replace=False)
        tournament_fitness = fit[idx]
        
        # If temperature is very low, revert to deterministic tournament
        if temperature <= 0.01:
            best = idx[np.argmax(tournament_fitness)]
        else:
            # Apply temperature scaling for stochastic selection
            scaled_fitness = tournament_fitness / temperature
            
            # Softmax selection (numerical stability with max subtraction)
            exp_fitness = np.exp(scaled_fitness - np.max(scaled_fitness))
            probabilities = exp_fitness / np.sum(exp_fitness)
            
            # Sample based on probabilities
            selected_idx = rng.choice(len(idx), p=probabilities)
            best = idx[selected_idx]
        
        return pop_rules[best], pop_progs[best]
    
    def get_current_temperature(self, generation):
        """
        Calculate current temperature with exponential cooling.
        
        Args:
            generation: Current generation number
            
        Returns:
            Current temperature (asymptotically approaches MIN_TEMPERATURE)
        """
        temp = self.INITIAL_TEMPERATURE * (self.COOLING_RATE ** generation)
        return max(temp, self.MIN_TEMPERATURE)
    
    def select_parent(self, pop_rules, pop_progs, fit, generation):
        """
        Select a parent using the configured selection method.
        
        Args:
            pop_rules: Population rule arrays
            pop_progs: Population program arrays
            fit: Fitness values
            generation: Current generation number
            
        Returns:
            Selected rule and program arrays
        """
        if self.ENABLE_STOCHASTIC_TOURNAMENT:
            current_temp = self.get_current_temperature(generation)
            return self.stochastic_tournament(pop_rules, pop_progs, fit, current_temp)
        else:
            return self.tournament(pop_rules, pop_progs, fit)

    def crossover(self, rule1, prog1, rule2, prog2):
        vec1 = np.concatenate((rule1, prog1))
        vec2 = np.concatenate((rule2, prog2))
        cut = rng.integers(1, vec1.size)
        child = np.concatenate((vec1[:cut], vec2[cut:]))
        return _sanitize_rule(child[:64]), _sanitize_programme(child[64:64+self.PROG_LEN])

    def mutate(self, rule, prog):
        # LUT entries
        mask_r = rng.random(64) < self.P_MUT_RULE
        if mask_r.any():
            rule = rule.copy()
            rule[mask_r] = rng.integers(0, 4, mask_r.sum(), dtype=np.uint8)
            rule = _sanitize_rule(rule)
        # Programme cells
        mask_p = rng.random(self.PROG_LEN) < self.P_MUT_PROG
        if mask_p.any():
            prog = prog.copy()
            prog[mask_p] = rng.choice([0,1,2], size=mask_p.sum(), p=[0.7,0.2,0.1])
            prog = _sanitize_programme(prog)
        return rule, prog

    def avg_pairwise_hamming(self, flat: np.ndarray) -> float:
        P = flat.shape[0]
        total = 0
        for i in range(P-1):
            diff = np.count_nonzero(flat[i+1:] != flat[i], axis=1)
            total += diff.sum()
        return total / (P*(P-1)//2)

    # ───────────────── GA main loop ────────────────────────────────────
    def run(self):
        """
        Run Genetic Algorithm.

        Returns:
        --------
        tuple[np.ndarray, np.ndarray, float]
            (best_rule, best_prog, best_fitness)
        """
        print(f"Starting run with ID: {self.RUN_ID}")
        print(f"Task: {self.TASK_ID} - {self.TASK_DESCRIPTION}")
        print(f"Logging to: {self.RUN_LOG_FILE}")
        print(f"UTC Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Log diversity settings
        if self.ENABLE_STOCHASTIC_TOURNAMENT:
            print(f"Stochastic Tournament: ENABLED")
            print(f"  Initial Temperature: {self.INITIAL_TEMPERATURE}")
            print(f"  Cooling Rate: {self.COOLING_RATE}")
            print(f"  Min Temperature: {self.MIN_TEMPERATURE}")
        else:
            print(f"Stochastic Tournament: DISABLED (using standard tournament)")
        
        # Initialize population
        for i in range(self.POP_SIZE):
            r, p = self.random_genome()
            self.pop_rules[i], self.pop_progs[i] = r, p

        for gen in tqdm(range(1, self.GENERATIONS+1), ncols=80, desc="GA"):
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

            if gen % self.COMPLEX_TELEMETRY == 0:
                flat = np.concatenate((self.pop_rules, self.pop_progs), axis=1)
                ham = self.avg_pairwise_hamming(flat)
                if self.ENABLE_STOCHASTIC_TOURNAMENT:
                    current_temp = self.get_current_temperature(gen)
                    tqdm.write(f"Gen {gen:3}  best={fit[0]:.3f}  mean={fit.mean():.3f}  ham={ham:.1f}  temp={current_temp:.3f}")
                else:
                    tqdm.write(f"Gen {gen:3}  best={fit[0]:.3f}  mean={fit.mean():.3f}  ham={ham:.1f}")
            else:
                if self.ENABLE_STOCHASTIC_TOURNAMENT:
                    current_temp = self.get_current_temperature(gen)
                    tqdm.write(f"Gen {gen:3}  best={fit[0]:.3f}  mean={fit.mean():.3f}  temp={current_temp:.3f}")
                else:
                    tqdm.write(f"Gen {gen:3}  best={fit[0]:.3f}  mean={fit.mean():.3f}")

            # Check-point
            if gen % self.CHECK_EVERY == 0 or gen == self.GENERATIONS:
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
                
                # Log run data to CSV
                self._log_run_data(gen, float(fit[0]), self.pop_rules[0], self.pop_progs[0])
                tqdm.write(f"Logged checkpoint {gen} data to {self.RUN_LOG_FILE}")

            # Next generation
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

        # save the best genome
        best = {
                    "gen": gen,
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
