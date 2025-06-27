"""
em43_infer.py  -  One-off inference for a trained EM-4/3 genome
===============================================================

• Always loads **best_genome.pkl** from the current directory.
• Takes one optional positional argument (an integer input n).
  If omitted, uses a small default list (1…10).

Example
-------
    python em43_infer.py          # tests 1..10
    python em43_infer.py 17       # tests just 17


this code has not been checked - may still present unexpected behaviours
"""

from __future__ import annotations
import sys, pickle, numpy as np
from pathlib import Path
from em43_numba import EM43Batch

# ───── choose inputs ────────────────────────────────────────────────
if len(sys.argv) > 1:
    try:
        TEST_INPUTS = [int(sys.argv[1])]
    except ValueError:
        sys.exit("Input must be an integer.")
else:
    TEST_INPUTS = list(range(1, 11))  # default 1..10

# ───── load genome ──────────────────────────────────────────────────
CKPT = Path("best_genome.pkl")
if not CKPT.exists():
    sys.exit("best_genome.pkl not found - run the GA first.")

with CKPT.open("rb") as f:
    data = pickle.load(f)
# handle multiple pickle layouts
if "genome" in data:
    rule, prog = data["genome"]
elif {"rule", "prog"}.issubset(data):
    rule, prog = data["rule"], data["prog"]
else:
    # checkpoint style
    rule, prog = data.get("best", [(None, None)])[0]

print(f"Loaded genome (prog len={len(prog)}) from {CKPT.name}")

# ───── setup simulator ──────────────────────────────────────────────
# window must be ≥ prog_len + beacon space; pick a generous cap
WINDOW     = 1000
MAX_STEPS  = 1500
HALT_THRESH= 0.50

batcher = EM43Batch((rule, prog),
                    window=WINDOW,
                    max_steps=MAX_STEPS,
                    halt_thresh=HALT_THRESH)

# ───── evaluate inputs ──────────────────────────────────────────────
outputs = batcher.run(TEST_INPUTS)

for n, out in zip(TEST_INPUTS, outputs):
    status = "halt" if out >= 0 else "no-halt"
    steps  = batcher.max_steps  # EM43Batch does not record step count
    print(f"n={n:3d}  →  output={out:5d}   ({status})")

# Note: EM43Batch doesn’t track the halt step or full tape evolution.
# If you need to visualise steps or count exactly when it halted,
# use `infer_best_model.py`’s `plot_inference_steps` logic instead.
