"""
EM43 - Emergent Model with 4 States
===================================

A class-based implementation of the EM43 model featuring:
- 4 distinct states (0-3)
- Radius-1 neighborhood (3 cells)
- Numba-accelerated parallel processing
- Configurable parameters through YAML

Author: Giacomo Bocchese - with the help of ChatGPT
"""

from __future__ import annotations
import pickle
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
from em43_numba import EM43Batch


class EM43:
    """Emergent Model with 4 States (EM43) implementation"""
    
    def __init__(
        self,
        pop_rule: Optional[np.ndarray] = None,
        pop_prog: Optional[np.ndarray] = None,
        window: int = 1000,
        max_steps: int = 1500,
        halt_thresh: float = 0.50
    ):
        """
        Initialize the EM43 model.
        
        Args:
            pop_rule: The rule array for the model
            pop_prog: The program array for the model
            window: Tape length for simulation
            max_steps: Maximum simulation steps
            halt_thresh: Threshold for halting condition
        """
        self.pop_rule = pop_rule
        self.pop_prog = pop_prog
        self.WINDOW = window
        self.MAX_STEPS = max_steps
        self.HALT_THRESH = halt_thresh
        self.batcher = None
        
        # Initialize simulator only if rule and prog are provided
        if self.pop_rule is not None and self.pop_prog is not None:
            self._initialize_simulator()

    def _initialize_simulator(self) -> None:
        """Initialize the EM43Batch simulator"""
        self.batcher = EM43Batch(
            (self.pop_rule, self.pop_prog),
            window=self.WINDOW,
            max_steps=self.MAX_STEPS,
            halt_thresh=self.HALT_THRESH
        )

    def load_genome(self, checkpoint_path: Path=Path("dp_checkpoints/best_genome.pkl")) -> None:
        """
        Load a genome from a checkpoint file.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        with checkpoint_path.open("rb") as f:
            data = pickle.load(f)
            
        # Handle different pickle layouts
        if "genome" in data:
            self.pop_rule, self.pop_prog = data["genome"]
        elif {"best_rule", "best_prog", "target_out"}.issubset(data):
            self.pop_rule, self.pop_prog = data["best_rule"], data["best_prog"]
        else:
            self.pop_rule, self.pop_prog = data.get("best", [(None, None)])[0]

        self._initialize_simulator()
        print(f"Loaded genome (prog len={len(self.pop_prog)}) from {checkpoint_path.name}")

    def run(self, inputs: Optional[List[int]] = None) -> List[int]:
        """
        Run the model on a list of inputs.
        
        Args:
            inputs: List of integer inputs to process
            
        Returns:
            List of outputs corresponding to each input
        """
        if self.batcher is None:
            raise ValueError("Model not initialized. Load a genome first: `em43.load_genome()`.")

        if not inputs:  # if no inputs are provided
            inputs = list(range(1, 11))  # default 1..10    
        outputs = self.batcher.run(inputs)
        return outputs

    def evaluate(self, inputs: Optional[List[int]] = None) -> None:
        """
        Evaluate the model on a list of inputs and print results.
        
        Args:
            inputs: List of integer inputs to process
        """
        if not inputs:  # if no inputs are provided
            inputs = list(range(1, 11))  # default 1..10

        outputs = self.run(inputs)
        
        for n, out in zip(inputs, outputs):
            status = "halt" if out >= 0 else "no-halt"
            steps = self.MAX_STEPS  # EM43Batch does not record step count
            print(f"n={n:3d}  →  output={out:5d}   ({status})")

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path) -> "EM43":
        """
        Create an EM43 instance from a checkpoint file.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            Initialized EM43 instance
        """
        instance = cls()
        instance.load_genome(checkpoint_path)
        return instance

    @classmethod
    def default(cls) -> "EM43":
        """
        Create an EM43 instance with default parameters.
        
        Returns:
            EM43 instance with default parameters
        """
        return cls(
            WINDOW=1000,
            MAX_STEPS=1500,
            HALT_THRESH=0.50
        )
