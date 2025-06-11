"""
Extract the actual trained divide-by-2 genome for the HTML viewer.
"""

import pickle
import numpy as np

# Load the trained model
with open("best_genome.pkl", "rb") as f:
    data = pickle.load(f)

rule = data["rule"]
prog = data["prog"]

print("Trained Divide-by-2 Genome:")
print("=" * 40)

print("\nProgram (length {}):".format(len(prog)))
prog_str = "".join(str(x) for x in prog)
print(f'const PROG=[..."{prog_str}"].map(Number);')

print("\nRule (length {}):".format(len(rule)))
rule_chunks = []
for i in range(0, len(rule), 16):
    chunk = rule[i:i+16]
    chunk_str = ",".join(str(x) for x in chunk)
    rule_chunks.append(f"  {chunk_str}")

print("const RULE=Uint8Array.from([")
print(",\n".join(rule_chunks))
print("]);")

print(f"\nFitness: {data.get('fitness', 'unknown')}")

# Test the genome on a few examples
print("\nTesting the genome:")
from em43_numba import _simulate

test_inputs = np.array([2, 4, 6, 8, 10, 12], dtype=np.int64)
outputs = _simulate(rule, prog, test_inputs, 300, 1000, 0.5)

for inp, out in zip(test_inputs, outputs):
    expected = inp // 2
    status = "✓" if out == expected else "✗"
    print(f"{inp} ÷ 2 = {expected} → {out} {status}")
