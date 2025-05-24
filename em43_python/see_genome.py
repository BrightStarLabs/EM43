"""
see_genome.py  -  Dump a trained EM-4/3 genome as pure numbers
===========================================================================
• Reads best_genome.pkl from the current directory.
• Prints:

    1. Programme slice as a string of digits (0/1/2)
    2. Full lookup-table in numeric form “LCR:N”

Run:
    python see_genome.py

Author: Giacomo Bocchese - with the help of ChatGPT

this code has not been checked - may still present unexpected behaviours
"""

import sys, pickle
from pathlib import Path
import numpy as np
from em43_numba import _sanitize_rule, _sanitize_programme

CKPT = Path("best_genome.pkl")
if not CKPT.exists():
    sys.exit("best_genome.pkl not found - train a model first.")

# Robust loader for various pickle layouts
def load_genome(path="best_genome.pkl") -> tuple[np.ndarray,np.ndarray]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    # layout A: {'rule':..., 'prog':...}
    if {"rule","prog"}.issubset(data):
        return data['rule'], data['prog']
    # layout B: {'genome': (rule,prog)}
    if 'genome' in data:
        return data['genome']
    # layout C: {'best': [(rule,prog), ...]}
    if 'best' in data and isinstance(data['best'], list):
        return data['best'][0]
    raise KeyError("Cannot find genome in pickle file")

# Load and sanitize
rule_raw, prog_raw = load_genome(CKPT)
rule = _sanitize_rule(np.array(rule_raw, dtype=np.uint8))
prog = _sanitize_programme(np.array(prog_raw, dtype=np.uint8))

# Print programme slice
print("\nProgramme slice (numeric):")
print('   ', ''.join(str(int(v)) for v in prog))
print(f"   length = {len(prog)}\n")

# Print rule table
print("Rule table (numeric LCR:N)\n")
for left in range(4):
    for centre in range(4):
        row = []
        for right in range(4):
            idx = (left << 4) | (centre << 2) | right
            val = int(rule[idx])
            row.append(f"{left}{centre}{right}:{val}")
        print('  '.join(row))
    print()