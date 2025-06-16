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
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List
from em43_numba import EM43Batch
from matplotlib.colors import ListedColormap
import sys

# ───────────── symbols & cmap ─────────────
SYMBOLS = {0: "·", 1: "P", 2: "R", 3: "B"}
CMAP    = ListedColormap(["white", "black", "red", "blue"])

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
        self.expected = None
        self.fit = None
        
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

    def load_genome(self, full_path: Path = None) -> None:
        """Load a genome from a checkpoint file.
        
        Args:
            full_path: Path to the checkpoint file relative to the project root
        """
        if full_path is None:
            project_root = Path(__file__).parent  # Go up two levels from this file
            full_path = project_root / Path("dp_checkpoints/best_genome.pkl")
        
        # Create the directory if it doesn't exist
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with full_path.open("rb") as f:
                data = pickle.load(f)
                
            # Handle different pickle layouts
            if "genome" in data:
                self.pop_rule, self.pop_prog = data["genome"]
            elif {"best_rule", "best_prog", "target_out","best_fitness"}.issubset(data):
                self.pop_rule, self.pop_prog = data["best_rule"], data["best_prog"]
                self.expected = data["target_out"]
                self.fit = data["best_fitness"]
            else:
                self.pop_rule, self.pop_prog = data.get("best", [(None, None)])[0]

            print(data)
            self._initialize_simulator()
            print(f"Loaded genome (prog len={len(self.pop_prog)}) from {full_path.name}")
        except FileNotFoundError:
            print(f"Error: Genome file not found at {full_path}")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading genome: {e}")
            sys.exit(1)

    def infer(self, inputs: Optional[List[int]] = None, print_results: bool = True) -> List[int]:
        """
        Infer from the best genome and print results.
        
        Args:
            inputs: List of integer inputs to process
            print_results: Whether to print results
            
        Returns:
            List of outputs corresponding to each input
        """
        if self.batcher is None:
            raise ValueError("Model not initialized. Load a genome first: `em43.load_genome()`.")

        if not inputs:  # if no inputs are provided
            inputs = list(range(1, 11))  # default 1..10    

        outputs = self.batcher.run(inputs)

        if print_results:
            for n, out in zip(inputs, outputs):
                status = "halt" if out >= 0 else "no-halt"
                print(f"n={n:3d}  →  output={out:5d}   ({status})")

        return outputs

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

    @staticmethod
    def prog_str(p: np.ndarray) -> str: 
        """Convert a programme array to a string."""
        return "".join(SYMBOLS[x] for x in p)
    
    @staticmethod
    def prog_colormap(p: np.ndarray) -> None:
        """Display a programme array as a colormap."""
        plt.figure(figsize=(12,2)); plt.imshow([p], cmap=CMAP, aspect="auto")
        plt.axis("off"); plt.title("Programme"); plt.savefig("program_colors.png", dpi=150, bbox_inches="tight"); plt.close()

    def evaluate(self, inputs: Optional[List[int|float]] = None, verbose: bool = True, plot: bool = True) -> List[int|float]:
        """
        Evaluate the model on a list of inputs.
        
        Args:
            inputs: List of integer or float inputs to process
            print_results: Whether to print results
            
        Returns:
            List of outputs corresponding to each input
        """
        if self.batcher is None:
            raise ValueError("Model not initialized. Load a genome first: `em43.load_genome()`.")
        if self.expected is None:
            raise ValueError("Expected values not initialized. Load a genome first: `em43.load_genome()`.")
        print(self.expected)
        if not inputs:  # if no inputs are provided
            inputs = list(range(1, len(self.expected)+1))  # default 1..10    

        outputs = self.batcher.run(inputs)
        errs = np.abs(outputs-self.expected)
        avg_err = float(errs.mean())
        success = float((errs<0.1).mean()*100)
        accuracy = float((outputs==self.expected).mean()*100)

        for n, out in zip(inputs, outputs):
            status = "halt" if out >= 0 else "no-halt"
            print(f"n={n:3d}  →  output={out:5d}   ({status})")

        if verbose:
            print("\nBest EM-4/3 genome")
            print("="*60)
            print(f"Stored fitness : {self.fit:.3f}")
            print(f"Avg |err|      : {avg_err:.3f}")
            print(f"Success rate  : {success:.1f}%  (|err|<0.1)")
            print(f"Accuracy      : {accuracy:.1f}%  (exact)")
            print("\nProgramme:\n"+self.prog_str(self.pop_prog))
            print("\nInput | Out | Exp | Err")
            print("-"*38)
            for n,o,e,er in zip(inputs, outputs, self.expected, errs):
                print(f"{n:5d} | {o:4f} | {e:4f} | {er:4f}")

        if plot:
            plt.figure(figsize=(8,6))
            plt.plot(inputs,self.expected,"b-",lw=2,label="Expected")
            plt.plot(inputs,outputs,"r--",lw=2,label="Predicted"); plt.scatter(inputs,outputs,color="red")
            plt.grid(alpha=0.3); plt.xlabel("Input"); plt.ylabel("Output"); plt.title("Expected vs Predicted")
            plt.legend(); plt.tight_layout(); 
            self.prog_colormap(self.pop_prog)
            plt.savefig("prediction_plot.png",dpi=150); 

        return outputs