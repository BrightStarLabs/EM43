"""
em43_numba.py  -  Numba-accelerated batched EM-4/3 simulator
==================================================================
Drop-in replacement for the original **em43_parallel.py**.  
Public API unchanged:

    from em43_numba import EM43Batch, _sanitize_rule, _sanitize_programme

Key details
-----------
* 1-D CA, 4 states, radius-1, open boundary, 2-cell separator “BB”.
* Evaluates **B inputs in parallel** for a single genome.
* Core simulation loop is compiled with Numba (`@njit(cache=True)`).
* First call takes a few 100 ms to compile, then runs 5-10x faster.

No bit-packing; all arrays are `uint8`.  First & last columns stay blank.

Author: Giacomo Bocchese - with the help of ChatGPT

this code has not been checked - may still present unexpected behaviours
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np
import numba as nb

# ────────────────── helpers & constants ──────────────────────────────
def lut_idx(l: int, c: int, r: int) -> int:              # 3-tuple → 0..63
    return (l << 4) | (c << 2) | r


SEPARATOR = np.array([3, 3], dtype=np.uint8)             # BB

_IMMUTABLE = {                                           # hard-wired LUT rows
    lut_idx(0, 0, 0): 0,
    lut_idx(0, 2, 0): 2,
    lut_idx(0, 0, 2): 0,
    lut_idx(2, 0, 0): 0,
    lut_idx(0, 3, 3): 3,
    lut_idx(3, 3, 0): 3,
    lut_idx(0, 0, 3): 0,
    lut_idx(3, 0, 0): 0,
}

def _sanitize_rule(rule: np.ndarray) -> np.ndarray:
    """Overwrite immutable LUT entries; clip to 0-3."""
    rule = rule.astype(np.uint8, copy=True)
    for k, v in _IMMUTABLE.items():
        rule[k] = v
    rule[rule > 3] &= 3
    return rule

def _sanitize_programme(prog: np.ndarray) -> np.ndarray:
    """Remove accidental blue cells from programme."""
    prog = prog.astype(np.uint8, copy=True)
    prog[prog == 3] = 0
    return prog


# ────────────────── Numba simulation kernel ──────────────────────────
@nb.njit(cache=True)
def _simulate(rule: np.ndarray,
              prog: np.ndarray,
              inputs: np.ndarray,
              window: int,
              max_steps: int,
              halt_th: float) -> np.ndarray:
    """
    Parameters
    ----------
    rule    : (64,) uint8
    prog    : (L,)  uint8
    inputs  : (B,) int64     (values 1..30)
    Returns
    -------
    outputs : (B,) int32     (-10 on failure)
    """
    L = prog.shape[0]
    B = inputs.shape[0]
    N = window

    state   = np.zeros((B, N), np.uint8)
    halted  = np.zeros(B, np.bool_)
    frozen  = np.zeros_like(state)
    output  = np.full(B, -10, np.int32)

    # write programme & separator
    for b in range(B):
        for j in range(L):
            state[b, j] = prog[j]
        state[b, L    ] = 3     # B
        state[b, L + 1] = 3     # B

    # write beacons 0^(n+1) R 0
    for b in range(B):
        r_idx = L + 2 + inputs[b] + 1
        state[b, r_idx] = 2

    # main loop
    for _ in range(max_steps):
        active_any = False
        for b in range(B):
            if halted[b]:
                continue
            active_any = True
            nxt = np.zeros(N, np.uint8)
            for x in range(1, N - 1):
                idx = (state[b, x-1] << 4) | (state[b, x] << 2) | state[b, x+1]
                nxt[x] = rule[idx]
            state[b] = nxt

            # halting check
            live = blue = 0
            for x in range(N):
                v = nxt[x]
                if v != 0:
                    live += 1
                    if v == 3:
                        blue += 1
            if live > 0 and blue / live >= halt_th:
                halted[b] = True
                frozen[b] = nxt

        if not active_any:
            break

    # decode outputs
    for b in range(B):
        if not halted[b]:
            continue
        rpos = -1
        for x in range(N - 1, -1, -1):
            if frozen[b, x] == 2:
                rpos = x
                break
        if rpos != -1:
            output[b] = rpos - (L + 3)          # (sep=2)+1 zeros before R

    return output

# ───────────────── Vectorised fitness via Numba ────────────────────
@nb.njit(parallel=True, fastmath=True, cache=True, )
def fitness_population(rules: np.ndarray, progs: np.ndarray, inputs: np.ndarray, target_out: np.ndarray, window: int, max_steps: int, halt_th: float, lambda_p: float) -> np.ndarray:
    """Compute fitness for every genome in parallel.
    rules  : (P,64)  uint8
    progs  : (P,L)   uint8
    window : int
    max_steps : int
    halt_th : float
    lambda_p : float
    returns: (P,)    float32
    """
    P = rules.shape[0]
    fitness = np.empty(P, dtype=np.float32)
    for i in nb.prange(P):
        outs = _simulate(rules[i], progs[i], inputs, window,
                        max_steps, halt_th)
        avg_err = np.abs(outs - target_out).mean()
        sparsity = np.count_nonzero(progs[i]) / progs.shape[1]
        fitness[i] = -avg_err - lambda_p * sparsity
    return fitness

# ────────────────── OO wrapper (same API) ────────────────────────────
class EM43Batch:
    """
    Evaluate a single genome on B inputs in parallel (Numba backend).

    Parameters
    ----------
    genome      : (rule_array, programme_array)
    window      : int   tape length
    max_steps   : int
    halt_thresh : float
    """

    def __init__(self,
                 genome: Tuple[np.ndarray, np.ndarray],
                 window: int = 500,
                 max_steps: int = 256,
                 halt_thresh: float = 0.50):
        rule, prog = genome
        self.rule  = _sanitize_rule(rule)
        self.prog  = _sanitize_programme(prog)
        self.L     = len(self.prog)

        if self.L + 5 >= window:             # L + BB + 0 R 0  needs ≥5 extra
            raise ValueError(f"window {window} too small for given programme length {self.L}")

        self.N           = window
        self.max_steps   = max_steps
        self.halt_thresh = halt_thresh

    # -----------------------------------------------------------------
    def run(self, inputs: List[int]) -> np.ndarray:
        """Compute doubling outputs for the given input list."""
        return _simulate(self.rule,
                         self.prog,
                         np.asarray(inputs, dtype=np.int64),
                         self.N,
                         self.max_steps,
                         self.halt_thresh)


# ────────────────── quick demo ───────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng()
    rule = rng.integers(0, 4, 64, dtype=np.uint8)
    prog = rng.choice([0, 1, 2], size=32, p=[0.7, 0.2, 0.1])
    sim  = EM43Batch((rule, prog), window=300, max_steps=256)
    print("outputs 1..30:", sim.run(list(range(1, 31))))