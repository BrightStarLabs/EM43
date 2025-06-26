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

Author: Giacomo Bocchese - with the help of ChatGPT

this code has not been checked - may still present unexpected behaviours
"""

from __future__ import annotations
import numpy as np, math, pickle
from pathlib import Path
from tqdm import tqdm

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
        
        # Derived parameters
        self.N_ELITE = int(math.ceil(self.ELITE_FRAC * self.POP_SIZE))
        self.N_IMM = max(1, int(self.RANDOM_IMMIGRANTS * self.POP_SIZE))
        
        # Initialize population arrays
        self.pop_rules = np.empty((self.POP_SIZE, 64), np.uint8)
        self.pop_progs = np.empty((self.POP_SIZE, self.PROG_LEN), np.uint8)
        
        # Initialize history
        self.best_curve = []
        self.mean_curve = []


    # ───────────────── GA methods ──────────────────────────────────────
    def random_genome(self) -> tuple[np.ndarray, np.ndarray]:
        rule = rng.integers(0, 4, 64, dtype=np.uint8)
        prog = rng.choice([0, 1, 2], size=self.PROG_LEN, p=[0.7, 0.2, 0.1])
        return _sanitize_rule(rule), _sanitize_programme(prog)

    def tournament(self, pop_rules, pop_progs, fit):
        idx = rng.choice(self.POP_SIZE, self.TOURNEY_K, replace=False)
        best = idx[np.argmax(fit[idx])]
        return pop_rules[best], pop_progs[best]

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

            self.best_curve.append(float(fit[0]))
            self.mean_curve.append(float(fit.mean()))

            if gen % self.COMPLEX_TELEMETRY == 0:
                flat = np.concatenate((self.pop_rules, self.pop_progs), axis=1)
                ham = self.avg_pairwise_hamming(flat)
                tqdm.write(f"Gen {gen:3}  best={fit[0]:.3f}  mean={fit.mean():.3f}  ham={ham:.1f}")
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

            # Next generation
            next_rules = np.empty_like(self.pop_rules)
            next_progs = np.empty_like(self.pop_progs)

            # Elite preservation
            next_rules[:self.N_ELITE] = self.pop_rules[:self.N_ELITE]
            next_progs[:self.N_ELITE] = self.pop_progs[:self.N_ELITE]

            # Fill remaining with tournament selection and crossover
            for i in range(self.N_ELITE, self.POP_SIZE - self.N_IMM):
                r1, p1 = self.tournament(self.pop_rules, self.pop_progs, fit)
                r2, p2 = self.tournament(self.pop_rules, self.pop_progs, fit)
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
