"""
EM43 Unified Wrapper - Single Interface for All EM43 Operations
===============================================================

A unified class interface for the refactored EM43 system that handles:
- Automatic mode detection (1-input vs 2-input)
- Seamless config.yaml integration
- Training, evaluation, and inference
- Visualization and result management
- Genome loading/saving with format auto-detection

Usage:
    # Simple 1-input training
    model = EM43Wrapper(task_id=1)
    model.train()
    model.evaluate()
    outputs = model.infer([1, 2, 3, 4, 5])
    
    # 2-input training
    model = EM43Wrapper(task_id=20)
    model.train()
    outputs = model.infer([1, 2, 3], [4, 5, 6])
    
    # Custom configuration
    model = EM43Wrapper(task_id=1, config_overrides={'population': {'pop_size': 100}})
"""

import numpy as np
import pickle
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple
import copy
import time

# Import refactored components
from tasks_config import get_dataset, TASKS
from em43_ga import train_model
from em43_numba import EM43Batch, EM43TwoInputBatch
from matplotlib.colors import ListedColormap

# Constants
SYMBOLS = {0: "·", 1: "P", 2: "R", 3: "B"}
CMAP = ListedColormap(["white", "black", "red", "blue"])


class EM43Wrapper:
    """Unified wrapper for EM43 training, evaluation, and inference."""
    
    def __init__(
        self,
        task_id: int,
        config_path: str = "config.yaml",
        config_overrides: Optional[Dict[str, Any]] = None,
        save_dir: str = "dp_checkpoints"
    ):
        """
        Initialize the EM43 wrapper.
        
        Args:
            task_id: Task ID from tasks_config.py
            config_path: Path to configuration file
            config_overrides: Dictionary of config overrides
            save_dir: Directory for saving checkpoints and results
        """
        self.task_id = task_id
        self.config_path = config_path
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and validate task
        if task_id not in TASKS:
            raise ValueError(f"Task {task_id} not found. Available tasks: {list(TASKS.keys())}")
        
        self.task = get_dataset(task_id)
        self.mode = self.task.mode
        
        # Load configuration
        self.config = self._load_config(config_overrides)
        
        # Initialize state
        self.best_rule = None
        self.best_prog = None
        self.best_fitness = None
        self.training_history = None
        self.is_trained = False
        
        # Create plots directory
        self.plots_dir = Path("plots")
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🧠 EM43Wrapper initialized:")
        print(f"   Task {task_id}: {self.task.description} ({self.mode})")
        print(f"   Training samples: {len(self.task.targets)}")
        if self.task.valid_inputs is not None:
            print(f"   Validation samples: {len(self.task.valid_targets)}")
        print(f"   Save directory: {self.save_dir}")

    def _load_config(self, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Load configuration from YAML with optional overrides."""
        # Resolve possible locations for the configuration file
        candidate_paths: List[Path] = []

        # 1) Path provided by user (absolute or relative to CWD)
        candidate_paths.append(Path(self.config_path))

        # 2) Same directory as this file (module directory)
        module_dir = Path(__file__).resolve().parent
        candidate_paths.append(module_dir / self.config_path)

        # 3) Project root (one level up from module_dir)
        candidate_paths.append(module_dir.parent / self.config_path)

        config_data: Optional[Dict[str, Any]] = None
        found_path: Optional[Path] = None

        for path in candidate_paths:
            if path.is_file():
                with open(path, 'r') as f:
                    config_data = yaml.safe_load(f)
                found_path = path
                break

        if config_data is None:
            # Build helpful error message with attempted locations
            attempted = " | ".join(str(p) for p in candidate_paths)
            raise FileNotFoundError(f"Config file not found. Tried: {attempted}")

        # Update internal path reference to the resolved one (for transparency)
        self.config_path = str(found_path)
        config = config_data
        
        # Apply task_id to config
        config['input_output']['task_id']['value'] = self.task_id
        
        # Apply overrides
        if overrides:
            config = self._deep_update(config, overrides)
        
        return config

    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> Dict:
        """Recursively update nested dictionary."""
        result = copy.deepcopy(base_dict)
        for key, value in update_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_update(result[key], value)
            else:
                result[key] = value
        return result

    def train(self, save_genome: bool = True, verbose: bool = True, *, checkpoint_callback=None, checkpoint_interval: int = 0):
        """
        Train the EM43 model using genetic algorithm.
        
        Args:
            save_genome: Whether to save the best genome
            verbose: Whether to print training progress
            
        Returns:
            Tuple of (best_rule, best_prog, best_fitness, history)
        """
        if verbose:
            print(f"\n🚀 Training EM43 for Task {self.task_id}")
            print("=" * 60)
            print(f"Mode: {self.mode}")
            print(f"Population: {self.config['population']['pop_size']['value']}")
            print(f"Generations: {self.config['population']['generations']['value']}")
            print(f"Window: {self.config['model']['window']['value']}")
            print("=" * 60)
        
        start_time = time.time()
        
        # Train using the refactored train_model function with optional callback
        self.best_rule, self.best_prog, self.best_fitness, self.training_history = train_model(
            self.config,
            checkpoint_callback=checkpoint_callback,
            checkpoint_interval=checkpoint_interval,
        )
        
        training_time = time.time() - start_time
        self.is_trained = True
        
        if verbose:
            print(f"\n✅ Training completed in {training_time:.1f}s")
            print(f"Best fitness: {self.best_fitness:.3f}")
            print(f"Final generation fitness: {self.training_history[-1]:.3f}")
            if len(self.training_history) > 1:
                improvement = self.training_history[-1] - self.training_history[0]
                print(f"Fitness improvement: {improvement:.3f}")
        
        # Save genome if requested
        if save_genome:
            self.save_genome(verbose=verbose)
        
        return self.best_rule, self.best_prog, self.best_fitness, self.training_history

    def evaluate(self, inputs: Optional[Union[List[int], Tuple[List[int], List[int]]]] = None, 
                verbose: bool = True, plot: bool = True) -> Dict[str, Any]:
        """
        Evaluate the model on validation data or custom inputs.
        
        Args:
            inputs: Custom inputs (1-input: list, 2-input: tuple of lists)
            verbose: Whether to print detailed results
            plot: Whether to generate plots
            
        Returns:
            Dictionary with evaluation results
        """
        if not self.is_trained and self.best_rule is None:
            # Try to load genome
            if not self.load_genome(verbose=False):
                raise ValueError("No trained model available. Train first or load a genome.")
        
        if verbose:
            print(f"\n📊 Evaluating Task {self.task_id}")
            print("=" * 50)
        
        # Determine inputs to use
        if inputs is None:
            # Use validation set if available, otherwise training set
            if self.task.valid_inputs is not None:
                if self.mode == "1input":
                    test_inputs = self.task.valid_inputs[:, 0].tolist()
                    test_targets = self.task.valid_targets.tolist()
                    data_type = "validation"
                else:
                    test_inputs = (self.task.valid_inputs[:, 0].tolist(), 
                                 self.task.valid_inputs[:, 1].tolist())
                    test_targets = self.task.valid_targets.tolist()
                    data_type = "validation"
            else:
                if self.mode == "1input":
                    test_inputs = self.task.inputs[:, 0].tolist()
                    test_targets = self.task.targets.tolist()
                    data_type = "training"
                else:
                    test_inputs = (self.task.inputs[:, 0].tolist(), 
                                 self.task.inputs[:, 1].tolist())
                    test_targets = self.task.targets.tolist()
                    data_type = "training"
        else:
            test_inputs = inputs
            test_targets = None
            data_type = "custom"
        
        # Run inference
        outputs = self.infer(test_inputs, verbose=False)
        
        # Calculate metrics
        results = {
            'outputs': outputs,
            'inputs': test_inputs,
            'targets': test_targets,
            'data_type': data_type,
            'task_id': self.task_id,
            'mode': self.mode
        }
        
        if test_targets is not None:
            errors = np.abs(np.array(outputs) - np.array(test_targets))
            results.update({
                'errors': errors.tolist(),
                'mean_error': float(np.mean(errors)),
                'max_error': float(np.max(errors)),
                'accuracy': float(np.mean(errors == 0)),
                'success_rate': float(np.mean(errors < 0.1)),
                'failed_simulations': int(np.sum(np.array(outputs) == -10))
            })
        
        if verbose:
            self._print_evaluation_results(results)
        
        if plot and test_targets is not None:
            self._plot_evaluation_results(results)
        
        return results

    def infer(self, inputs: Union[List[int], Tuple[List[int], List[int]]], 
             verbose: bool = True) -> List[int]:
        """
        Run inference on given inputs.
        
        Args:
            inputs: Input data (1-input: list, 2-input: tuple of lists)
            verbose: Whether to print results
            
        Returns:
            List of outputs
        """
        if not self.is_trained and self.best_rule is None:
            if not self.load_genome(verbose=False):
                raise ValueError("No trained model available. Train first or load a genome.")
        
        # Get evaluation window settings
        eval_window = self.config['evaluation']['window']['value']
        eval_max_steps = self.config['evaluation']['max_steps']['value']
        
        # Create simulator
        genome = (self.best_rule, self.best_prog)
        
        if self.mode == "1input":
            if not isinstance(inputs, list):
                raise ValueError("For 1-input mode, provide a list of integers")
            
            sim = EM43Batch(genome, window=eval_window, max_steps=eval_max_steps)
            outputs = sim.run(inputs)
            
            if verbose:
                print(f"\n🔮 1-Input Inference Results:")
                print("Input → Output (Status)")
                print("-" * 25)
                for inp, out in zip(inputs, outputs):
                    status = "✓" if out != -10 else "✗"
                    print(f"{inp:5d} → {out:5d} ({status})")
        
        else:  # 2input
            if not isinstance(inputs, tuple) or len(inputs) != 2:
                raise ValueError("For 2-input mode, provide a tuple of (list_a, list_b)")
            
            inputs_a, inputs_b = inputs
            if len(inputs_a) != len(inputs_b):
                raise ValueError("Input lists must have the same length")
            
            sim = EM43TwoInputBatch(genome, window=eval_window, max_steps=eval_max_steps)
            outputs = sim.run(inputs_a, inputs_b)
            
            if verbose:
                print(f"\n🔮 2-Input Inference Results:")
                print("Input A | Input B → Output (Status)")
                print("-" * 35)
                for a, b, out in zip(inputs_a, inputs_b, outputs):
                    status = "✓" if out != -10 else "✗"
                    print(f"{a:7d} | {b:7d} → {out:5d} ({status})")
        
        return outputs.tolist() if isinstance(outputs, np.ndarray) else outputs

    def save_genome(self, filename: Optional[str] = None, verbose: bool = True) -> Path:
        """
        Save the current best genome.
        
        Args:
            filename: Custom filename (optional)
            verbose: Whether to print save information
            
        Returns:
            Path to saved file
        """
        if self.best_rule is None or self.best_prog is None:
            raise ValueError("No genome to save. Train a model first.")
        
        if filename is None:
            filename = f"best_genome_task_{self.task_id}.pkl"
        
        filepath = self.save_dir / filename
        
        # Create genome data structure
        genome_data = {
            'best_rule': self.best_rule,
            'best_prog': self.best_prog,
            'best_fitness': self.best_fitness,
            'task_id': self.task_id,
            'mode': self.mode,
            'config': self.config,
            'training_history': self.training_history,
            'task_description': self.task.description
        }
        
        # Add task-specific data
        if self.mode == "1input":
            genome_data.update({
                'inputs': self.task.inputs[:, 0].tolist(),
                'targets': self.task.targets.tolist()
            })
        else:
            genome_data.update({
                'inputs_a': self.task.inputs[:, 0].tolist(),
                'inputs_b': self.task.inputs[:, 1].tolist(),
                'targets': self.task.targets.tolist()
            })
        
        # Save to file
        with open(filepath, 'wb') as f:
            pickle.dump(genome_data, f)
        
        if verbose:
            print(f"💾 Genome saved to: {filepath}")
        
        return filepath

    def load_genome(self, filepath: Optional[Union[str, Path]] = None, verbose: bool = True) -> bool:
        """
        Load a genome from file.
        
        Args:
            filepath: Path to genome file (optional, auto-detects)
            verbose: Whether to print load information
            
        Returns:
            True if successful, False otherwise
        """
        if filepath is None:
            # Try to find existing genome files (task-specific first, then mode-specific)
            candidates = [
                self.save_dir / f"best_genome_task_{self.task_id}.pkl",
                self.save_dir / f"best_genome_{self.mode}.pkl"
            ]
            
            # Only check for generic files if task_id is not specified or -1 (custom task)
            if self.task_id == -1:
                candidates.append(self.save_dir / "best_genome.pkl")
            
            filepath = None
            for candidate in candidates:
                if candidate.exists():
                    # Verify task compatibility if loading a genome
                    try:
                        with open(candidate, 'rb') as f:
                            data = pickle.load(f)
                        
                        # Check if the genome is compatible with current task
                        if 'task_id' in data and data['task_id'] != self.task_id and self.task_id != -1:
                            if verbose:
                                print(f"⚠️  Skipping incompatible genome for task {data['task_id']} (current: {self.task_id})")
                            continue
                        
                        if 'mode' in data and data['mode'] != self.mode:
                            if verbose:
                                print(f"⚠️  Skipping incompatible genome mode {data['mode']} (current: {self.mode})")
                            continue
                        
                        filepath = candidate
                        break
                        
                    except Exception as e:
                        if verbose:
                            print(f"⚠️  Error checking genome compatibility: {e}")
                        continue
            
            if filepath is None:
                if verbose:
                    print("❌ No compatible genome file found")
                return False
        
        filepath = Path(filepath)
        if not filepath.exists():
            if verbose:
                print(f"❌ Genome file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # Extract genome data (handle different formats)
            if 'best_rule' in data and 'best_prog' in data:
                self.best_rule = data['best_rule']
                self.best_prog = data['best_prog']
                self.best_fitness = data.get('best_fitness', None)
                self.training_history = data.get('training_history', None)
            else:
                raise ValueError("Invalid genome file format")
            
            self.is_trained = True
            
            if verbose:
                print(f"✅ Genome loaded from: {filepath}")
                if self.best_fitness is not None:
                    print(f"   Fitness: {self.best_fitness:.3f}")
                print(f"   Rule shape: {self.best_rule.shape}")
                print(f"   Program shape: {self.best_prog.shape}")
            
            return True
            
        except Exception as e:
            if verbose:
                print(f"❌ Error loading genome: {e}")
            return False

    def _print_evaluation_results(self, results: Dict[str, Any]) -> None:
        """Print detailed evaluation results."""
        print(f"Data type: {results['data_type']}")
        print(f"Samples: {len(results['outputs'])}")
        
        if results['targets'] is not None:
            print(f"Mean error: {results['mean_error']:.3f}")
            print(f"Max error: {results['max_error']:.3f}")
            print(f"Accuracy: {results['accuracy']:.1%} (exact matches)")
            print(f"Success rate: {results['success_rate']:.1%} (within ±0.1)")
            if results['failed_simulations'] > 0:
                print(f"Failed simulations: {results['failed_simulations']}")
        
        # Show sample results
        print(f"\nSample results:")
        if self.mode == "1input":
            inputs = results['inputs']
            outputs = results['outputs']
            targets = results['targets'] if results['targets'] is not None else [None] * len(outputs)
            
            print("Input → Output (Expected) [Status]")
            print("-" * 35)
            for i in range(min(10, len(outputs))):
                inp = inputs[i]
                out = outputs[i]
                exp = targets[i] if targets[i] is not None else "N/A"
                status = "✓" if targets[i] is not None and out == targets[i] else "✗" if targets[i] is not None else "?"
                print(f"{inp:5d} → {out:5d} ({exp:>3}) [{status}]")
        
        else:  # 2input
            inputs_a, inputs_b = results['inputs']
            outputs = results['outputs']
            targets = results['targets'] if results['targets'] is not None else [None] * len(outputs)
            
            print("Input A | Input B → Output (Expected) [Status]")
            print("-" * 45)
            for i in range(min(10, len(outputs))):
                inp_a = inputs_a[i]
                inp_b = inputs_b[i]
                out = outputs[i]
                exp = targets[i] if targets[i] is not None else "N/A"
                status = "✓" if targets[i] is not None and out == targets[i] else "✗" if targets[i] is not None else "?"
                print(f"{inp_a:7d} | {inp_b:7d} → {out:5d} ({exp:>3}) [{status}]")

    def _plot_evaluation_results(self, results: Dict[str, Any]) -> None:
        """Generate evaluation plots."""
        if results['targets'] is None:
            return
        
        outputs = np.array(results['outputs'])
        targets = np.array(results['targets'])
        
        if self.mode == "1input":
            self._plot_1input_results(results, outputs, targets)
        else:
            self._plot_2input_results(results, outputs, targets)

    def _plot_1input_results(self, results: Dict[str, Any], outputs: np.ndarray, targets: np.ndarray) -> None:
        """Plot 1-input evaluation results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Expected vs Predicted
        inputs = results['inputs']
        ax1.plot(inputs, targets, 'b-', linewidth=2, label='Expected', marker='o')
        ax1.plot(inputs, outputs, 'r--', linewidth=2, label='Predicted', marker='s')
        ax1.set_xlabel('Input')
        ax1.set_ylabel('Output')
        ax1.set_title(f'Task {self.task_id}: {self.task.description}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add performance metrics as text
        if 'mean_error' in results:
            ax1.text(0.02, 0.98, f"Mean Error: {results['mean_error']:.3f}", 
                    transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            ax1.text(0.02, 0.90, f"Accuracy: {results['accuracy']:.1%}", 
                    transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # Plot 2: Error distribution
        if 'errors' in results:
            errors = np.array(results['errors'])
            ax2.bar(range(len(errors)), errors, alpha=0.7, color='red')
            ax2.set_xlabel('Sample Index')
            ax2.set_ylabel('Absolute Error')
            ax2.set_title('Error Distribution')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plots_dir / f"evaluation_task_{self.task_id}_{results['data_type']}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"📊 Plot saved to: {plot_path}")
        plt.show()

    def _plot_2input_results(self, results: Dict[str, Any], outputs: np.ndarray, targets: np.ndarray) -> None:
        """Plot 2-input evaluation results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        inputs_a, inputs_b = results['inputs']
        
        # Plot 1: Expected vs Predicted (line plot)
        ax1 = axes[0, 0]
        sample_indices = range(len(targets))
        ax1.plot(sample_indices, targets, 'b-', linewidth=2, label='Expected', marker='o')
        ax1.plot(sample_indices, outputs, 'r--', linewidth=2, label='Predicted', marker='s')
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Output')
        ax1.set_title(f'Task {self.task_id}: {self.task.description}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Input space visualization
        ax2 = axes[0, 1]
        scatter = ax2.scatter(inputs_a, inputs_b, c=targets, cmap='viridis', s=100, alpha=0.7)
        ax2.set_xlabel('Input A')
        ax2.set_ylabel('Input B')
        ax2.set_title('Input Space (colored by target)')
        plt.colorbar(scatter, ax=ax2)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Error distribution
        ax3 = axes[1, 0]
        if 'errors' in results:
            errors = np.array(results['errors'])
            ax3.bar(range(len(errors)), errors, alpha=0.7, color='red')
            ax3.set_xlabel('Sample Index')
            ax3.set_ylabel('Absolute Error')
            ax3.set_title('Error Distribution')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Performance summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        if 'mean_error' in results:
            summary_text = f"""
Performance Summary:
• Mean Error: {results['mean_error']:.3f}
• Max Error: {results['max_error']:.3f}
• Accuracy: {results['accuracy']:.1%}
• Success Rate: {results['success_rate']:.1%}
• Samples: {len(outputs)}
• Failed Simulations: {results['failed_simulations']}

Program: {self.prog_str(self.best_prog)}
            """
            ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plots_dir / f"evaluation_task_{self.task_id}_{results['data_type']}_2input.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"📊 Plot saved to: {plot_path}")
        plt.show()

    @staticmethod
    def prog_str(prog: np.ndarray) -> str:
        """Convert program array to string representation."""
        return "".join(SYMBOLS[x] for x in prog)

    def get_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the wrapper state."""
        info = {
            'task_id': self.task_id,
            'task_description': self.task.description,
            'mode': self.mode,
            'training_samples': len(self.task.targets),
            'is_trained': self.is_trained,
            'config_path': str(self.config_path),
            'save_dir': str(self.save_dir)
        }
        
        if self.task.valid_inputs is not None:
            info['validation_samples'] = len(self.task.valid_targets)
        
        if self.is_trained:
            info.update({
                'best_fitness': self.best_fitness,
                'rule_shape': self.best_rule.shape if self.best_rule is not None else None,
                'prog_shape': self.best_prog.shape if self.best_prog is not None else None,
                'prog_string': self.prog_str(self.best_prog) if self.best_prog is not None else None
            })
        
        return info

    def __repr__(self) -> str:
        """String representation of the wrapper."""
        status = "trained" if self.is_trained else "untrained"
        return f"EM43Wrapper(task_id={self.task_id}, mode='{self.mode}', status='{status}')" 