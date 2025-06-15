"""
em43_div2_ga.py - Train EM43 to divide by 2
===========================================

Starting from the simplest possible mathematical operation:
division by 2 (the inverse of the original multiply by 2).

Author: Akshaj Devireddy
"""

from __future__ import annotations
import numpy as np, math, pickle, time
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numba as nb

nb.set_num_threads(nb.config.NUMBA_NUM_THREADS)
print(f"Numba using {nb.get_num_threads()} threads")

from em43_numba import _simulate, _sanitize_rule, _sanitize_programme

# ───────────────── Training Parameters ────────────────────────────────
POP_SIZE = 5000          # Reasonable population size
N_GENERATIONS = 200      # Moderate training time
ELITE_FRAC = 0.1         # Keep top 10%
TOURNEY_K = 3            # Tournament selection

P_MUT_RULE = 0.03        # Conservative mutation rates
P_MUT_PROG = 0.08        
L_PROG = 12              # Slightly longer programs

LAMBDA_P = 0.01          # Sparsity penalty
EPS_RANDOM_IMMIGRANTS = 0.15  # Some diversity

# ───────────────── Training Data ────────────────────────────────────
# Start with even numbers only (easier division)
INPUT_SET = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30], dtype=np.int64)
TARGET_OUT = INPUT_SET // 2  # Integer division by 2

print(f"Training divide-by-2 on {len(INPUT_SET)} even numbers")
print(f"Input range: {INPUT_SET[0]} to {INPUT_SET[-1]}")
print("Sample mappings:")
for i in range(min(8, len(INPUT_SET))):
    print(f"  {INPUT_SET[i]} ÷ 2 = {TARGET_OUT[i]}")

# ───────────────── Simulation Parameters ────────────────────────────────
WINDOW = 300             # Generous window
MAX_STEPS = 1000         # Plenty of steps
HALT_THRESH = 0.50       # Standard halting

# ───────────────── Checkpointing ────────────────────────────────────
CHECK_EVERY = 50
SAVE_DIR = Path("div2_checkpoints")
SAVE_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng()

@nb.njit(parallel=True, fastmath=True, cache=True)
def fitness_population(rules: np.ndarray, progs: np.ndarray) -> np.ndarray:
    """Compute fitness for divide-by-2 function."""
    P = rules.shape[0]
    fitness = np.empty(P, dtype=np.float32)
    
    for i in nb.prange(P):
        outs = _simulate(rules[i], progs[i], INPUT_SET, WINDOW, MAX_STEPS, HALT_THRESH)
        
        # Calculate accuracy and penalties
        correct = 0
        total_error = 0.0
        convergence_penalty = 0.0
        
        for j in range(len(outs)):
            predicted = outs[j]
            target = TARGET_OUT[j]
            
            if predicted == -10:  # Non-convergent
                convergence_penalty += 10.0
            elif predicted == target:
                correct += 1
            else:
                # Penalty proportional to error
                error = abs(predicted - target)
                total_error += error * 2.0
        
        accuracy = correct / len(outs)
        avg_error = total_error / len(outs)
        avg_convergence_penalty = convergence_penalty / len(outs)
        sparsity = np.count_nonzero(progs[i]) / len(progs[i])
        
        # Fitness: reward accuracy, penalize errors and complexity
        fitness[i] = accuracy * 100.0 - avg_error - avg_convergence_penalty - LAMBDA_P * sparsity
    
    return fitness

def random_genome():
    """Generate random genome."""
    rule = rng.integers(0, 4, 64, dtype=np.uint8)
    prog = rng.choice([0, 1, 2], size=L_PROG, p=[0.7, 0.2, 0.1])
    return _sanitize_rule(rule), _sanitize_programme(prog)

def tournament(pop_rules, pop_progs, fit):
    """Tournament selection."""
    idx = rng.choice(POP_SIZE, TOURNEY_K, replace=False)
    best = idx[np.argmax(fit[idx])]
    return pop_rules[best], pop_progs[best]

def crossover(rule1, prog1, rule2, prog2):
    """Crossover operation."""
    vec1 = np.concatenate((rule1, prog1))
    vec2 = np.concatenate((rule2, prog2))
    cut = rng.integers(1, vec1.size)
    child = np.concatenate((vec1[:cut], vec2[cut:]))
    return _sanitize_rule(child[:64]), _sanitize_programme(child[64:64+L_PROG])

def mutate(rule, prog):
    """Mutation operation."""
    # Rule mutations
    mask_r = rng.random(64) < P_MUT_RULE
    if mask_r.any():
        rule = rule.copy()
        rule[mask_r] = rng.integers(0, 4, mask_r.sum(), dtype=np.uint8)
        rule = _sanitize_rule(rule)
    
    # Program mutations
    mask_p = rng.random(L_PROG) < P_MUT_PROG
    if mask_p.any():
        prog = prog.copy()
        prog[mask_p] = rng.choice([0,1,2], size=mask_p.sum(), p=[0.7,0.2,0.1])
        prog = _sanitize_programme(prog)
    
    return rule, prog

def test_generalization(rule, prog):
    """Test on some unseen even numbers."""
    test_inputs = np.array([32, 34, 36, 38, 40], dtype=np.int64)
    test_targets = test_inputs // 2
    
    test_outs = _simulate(rule, prog, test_inputs, WINDOW, MAX_STEPS, HALT_THRESH)
    
    correct = 0
    for predicted, target in zip(test_outs, test_targets):
        if predicted == target:
            correct += 1
    
    return correct / len(test_targets) * 100

def run_div2_ga():
    """Run genetic algorithm for divide-by-2 function."""
    
    # Initialize population
    pop_rules = np.empty((POP_SIZE, 64), np.uint8)
    pop_progs = np.empty((POP_SIZE, L_PROG), np.uint8)
    
    for i in range(POP_SIZE):
        r, p = random_genome()
        pop_rules[i], pop_progs[i] = r, p
    
    best_curve = []
    mean_curve = []
    test_curve = []
    
    n_elite = int(ELITE_FRAC * POP_SIZE)
    n_imm = max(1, int(EPS_RANDOM_IMMIGRANTS * POP_SIZE))
    
    for gen in tqdm(range(1, N_GENERATIONS + 1), desc="Training div2"):
        # Evaluate fitness
        fit = fitness_population(pop_rules, pop_progs)
        order = np.argsort(fit)[::-1]
        pop_rules, pop_progs, fit = pop_rules[order], pop_progs[order], fit[order]
        
        best_curve.append(float(fit[0]))
        mean_curve.append(float(fit.mean()))
        
        # Test generalization every 25 generations
        if gen % 25 == 0:
            test_acc = test_generalization(pop_rules[0], pop_progs[0])
            test_curve.append(test_acc)
            
            # Detailed evaluation of best genome
            train_outs = _simulate(pop_rules[0], pop_progs[0], INPUT_SET, 
                                 WINDOW, MAX_STEPS, HALT_THRESH)
            train_correct = np.sum(train_outs == TARGET_OUT)
            train_acc = train_correct / len(TARGET_OUT) * 100
            convergent = np.sum(train_outs != -10)
            
            tqdm.write(f"Gen {gen:3}: fit={fit[0]:.1f} train={train_acc:.1f}% "
                      f"test={test_acc:.1f}% conv={convergent}/{len(TARGET_OUT)}")
            
            # Show some examples
            if gen % 50 == 0:
                tqdm.write("Examples:")
                for i in range(min(8, len(INPUT_SET))):
                    inp, target, pred = INPUT_SET[i], TARGET_OUT[i], train_outs[i]
                    status = "✓" if pred == target else "✗"
                    tqdm.write(f"  {inp:2d} ÷ 2 = {target:2d} → {pred:3d} {status}")
        
        # Checkpoint
        if gen % CHECK_EVERY == 0:
            checkpoint = {
                "gen": gen,
                "best_rule": pop_rules[0],
                "best_prog": pop_progs[0],
                "fitness": float(fit[0]),
                "best_curve": best_curve,
                "mean_curve": mean_curve,
                "test_curve": test_curve
            }
            with open(SAVE_DIR / f"div2_gen{gen:04d}.pkl", "wb") as f:
                pickle.dump(checkpoint, f)
        
        # Evolution
        next_rules = pop_rules[:n_elite].copy()
        next_progs = pop_progs[:n_elite].copy()
        
        while next_rules.shape[0] < POP_SIZE:
            r1, p1 = tournament(pop_rules, pop_progs, fit)
            r2, p2 = tournament(pop_rules, pop_progs, fit)
            child_r, child_p = mutate(*crossover(r1, p1, r2, p2))
            next_rules = np.vstack((next_rules, child_r))
            next_progs = np.vstack((next_progs, child_p))
        
        # Random immigrants
        for _ in range(n_imm):
            idx = rng.integers(n_elite, POP_SIZE)
            next_rules[idx], next_progs[idx] = random_genome()
        
        pop_rules, pop_progs = next_rules, next_progs
    
    # Final evaluation
    final_test_acc = test_generalization(pop_rules[0], pop_progs[0])
    train_outs = _simulate(pop_rules[0], pop_progs[0], INPUT_SET, 
                          WINDOW, MAX_STEPS, HALT_THRESH)
    final_train_acc = np.sum(train_outs == TARGET_OUT) / len(TARGET_OUT) * 100
    
    print(f"\nFinal Results:")
    print(f"Training accuracy: {final_train_acc:.1f}%")
    print(f"Test accuracy: {final_test_acc:.1f}%")
    print(f"Generalization gap: {final_train_acc - final_test_acc:.1f}%")
    
    # Save best model
    Path("models").mkdir(exist_ok=True)
    with open("models/best_div2_genome.pkl", "wb") as f:
        pickle.dump({
            "operation": "div2",
            "rule": pop_rules[0],
            "prog": pop_progs[0],
            "fitness": best_curve[-1],
            "train_accuracy": final_train_acc,
            "test_accuracy": final_test_acc
        }, f)
    
    # Also save in the format expected by em43_infer.py
    with open("best_genome.pkl", "wb") as f:
        pickle.dump({
            "rule": pop_rules[0],
            "prog": pop_progs[0],
            "fitness": best_curve[-1]
        }, f)
    
    print(f"Best div2 model saved with {final_train_acc:.1f}% training accuracy")
    return pop_rules[0], pop_progs[0]

if __name__ == "__main__":
    t0 = time.time()
    run_div2_ga()
    print(f"Training completed in {time.time()-t0:.1f}s")
