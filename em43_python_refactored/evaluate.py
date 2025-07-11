"""
EM43 Evaluation Script - Validation Set Testing
===============================================

Evaluates trained EM43 genomes on out-of-distribution validation sets.
Supports very large inputs and windows for comprehensive testing.

Usage:
    python evaluate.py [--genome PATH] [--window SIZE] [--max-steps STEPS]
"""

import numpy as np
import pickle
import argparse
import yaml
from pathlib import Path
from tasks_config import get_dataset, TASKS
from em43_numba import EM43Batch, EM43TwoInputBatch


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    return config_data


def load_genome(genome_path):
    """Load a saved genome from pickle file."""
    genome_path = Path(genome_path)
    if not genome_path.exists():
        raise FileNotFoundError(f"Genome file not found: {genome_path}")
    
    with open(genome_path, 'rb') as f:
        genome_data = pickle.load(f)
    
    required_keys = ['best_rule', 'best_prog']
    for key in required_keys:
        if key not in genome_data:
            raise ValueError(f"Invalid genome file: missing '{key}'")
    
    return genome_data


def evaluate_genome_on_task(genome_data, task_id, large_window=8000, max_steps=1500):
    """
    Evaluate genome on a specific task's validation set.
    
    Args:
        genome_data: Loaded genome dictionary
        task_id: Task ID to evaluate
        large_window: Window size for simulation (supports very large inputs)
        max_steps: Maximum simulation steps
        
    Returns:
        Dictionary with evaluation results
    """
    # Get task
    task = get_dataset(task_id)
    
    if task.valid_inputs is None:
        return None
    
    # Extract genome components
    rule = genome_data['best_rule']
    prog = genome_data['best_prog'] 
    genome = (rule, prog)
    
    print(f"  Running simulation (window={large_window}, max_steps={max_steps})...")
    
    # Run evaluation based on task mode
    if task.mode == "1input":
        sim = EM43Batch(genome, window=large_window, max_steps=max_steps)
        valid_inputs_1d = task.valid_inputs[:, 0]
        outputs = sim.run(valid_inputs_1d.tolist())
    else:  # 2input
        sim = EM43TwoInputBatch(genome, window=large_window, max_steps=max_steps)
        valid_a = task.valid_inputs[:, 0]
        valid_b = task.valid_inputs[:, 1]
        outputs = sim.run(valid_a.tolist(), valid_b.tolist())
    
    # Calculate comprehensive metrics
    valid_targets = task.valid_targets
    errors = np.abs(outputs - valid_targets)
    relative_errors = np.abs(errors / (valid_targets + 1e-8))  # Avoid division by zero
    
    # Failure analysis
    failed_outputs = outputs == -10  # -10 indicates simulation failure
    successful_outputs = outputs != -10
    
    results = {
        'task_id': task_id,
        'task_description': task.description,
        'mode': task.mode,
        'num_samples': len(valid_targets),
        'outputs': outputs,
        'targets': valid_targets,
        'errors': errors,
        'failed_simulations': np.sum(failed_outputs),
        'successful_simulations': np.sum(successful_outputs),
        'success_rate_sim': np.mean(successful_outputs),
        'mean_error': np.mean(errors[successful_outputs]) if np.any(successful_outputs) else np.inf,
        'max_error': np.max(errors[successful_outputs]) if np.any(successful_outputs) else np.inf,
        'mean_relative_error': np.mean(relative_errors[successful_outputs]) if np.any(successful_outputs) else np.inf,
        'exact_accuracy': np.mean(outputs[successful_outputs] == valid_targets[successful_outputs]) if np.any(successful_outputs) else 0,
        'tolerance_01': np.mean(errors[successful_outputs] <= 0.1) if np.any(successful_outputs) else 0,
        'tolerance_1': np.mean(errors[successful_outputs] <= 1.0) if np.any(successful_outputs) else 0,
        'input_range': (np.min(task.valid_inputs), np.max(task.valid_inputs)),
        'target_range': (np.min(valid_targets), np.max(valid_targets)),
    }
    
    return results


def print_evaluation_results(results):
    """Print detailed evaluation results."""
    if results is None:
        print("  No validation set available")
        return
    
    print(f"  Samples: {results['num_samples']}")
    print(f"  Input range: {results['input_range']}")
    print(f"  Target range: {results['target_range']}")
    print(f"  Successful simulations: {results['successful_simulations']}/{results['num_samples']} ({results['success_rate_sim']:.1%})")
    
    if results['successful_simulations'] > 0:
        print(f"  Mean error: {results['mean_error']:.3f}")
        print(f"  Max error: {results['max_error']:.3f}")
        print(f"  Mean relative error: {results['mean_relative_error']:.1%}")
        print(f"  Exact accuracy: {results['exact_accuracy']:.1%}")
        print(f"  Within ±0.1: {results['tolerance_01']:.1%}")
        print(f"  Within ±1.0: {results['tolerance_1']:.1%}")
    else:
        print("  ⚠️  All simulations failed!")
    
    # Show examples
    print("  Examples:")
    num_examples = min(8, len(results['targets']))
    for i in range(num_examples):
        if results['mode'] == "1input":
            task = get_dataset(results['task_id'])
            inp = task.valid_inputs[i, 0]
            output = results['outputs'][i]
            target = results['targets'][i]
            status = "✓" if output == target else "✗" if output != -10 else "⚠"
            print(f"    {status} {inp} → {output} (expected {target})")
        else:
            task = get_dataset(results['task_id'])
            inp_a, inp_b = task.valid_inputs[i, 0], task.valid_inputs[i, 1]
            output = results['outputs'][i]
            target = results['targets'][i]
            status = "✓" if output == target else "✗" if output != -10 else "⚠"
            print(f"    {status} ({inp_a},{inp_b}) → {output} (expected {target})")


def main():
    """Main evaluation function."""
    # Load config to get default values
    try:
        config = load_config()
        default_window = config['evaluation']['window']['value']
        default_max_steps = config['evaluation']['max_steps']['value']
    except Exception as e:
        print(f"Warning: Could not load config, using hardcoded defaults: {e}")
        default_window = 10000
        default_max_steps = 3000
    
    parser = argparse.ArgumentParser(description="Evaluate EM43 genome on validation sets")
    parser.add_argument("--genome", default="dp_checkpoints/best_genome.pkl", 
                       help="Path to saved genome file")
    parser.add_argument("--window", type=int, default=default_window,
                       help=f"Simulation window size (default from config: {default_window})")
    parser.add_argument("--max-steps", type=int, default=default_max_steps,
                       help=f"Maximum simulation steps (default from config: {default_max_steps})")
    parser.add_argument("--task", type=int, default=None,
                       help="Evaluate specific task ID only")
    parser.add_argument("--config", default="config.yaml",
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    print("EM43 Validation Set Evaluation")
    print("=" * 60)
    
    # Load genome
    try:
        genome_data = load_genome(args.genome)
        print(f"✓ Loaded genome: {args.genome}")
        print(f"  Fitness: {genome_data.get('best_fitness', 'unknown')}")
        print(f"  Mode: {genome_data.get('mode', 'unknown')}")
        print(f"  Rule size: {len(genome_data['best_rule'])}")
        print(f"  Program size: {len(genome_data['best_prog'])}")
    except Exception as e:
        print(f"✗ Error loading genome: {e}")
        return
    
    print(f"\nEvaluation Parameters:")
    print(f"  Window size: {args.window}")
    print(f"  Max steps: {args.max_steps}")
    
    # Determine tasks to evaluate
    if args.task is not None:
        task_ids = [args.task] if args.task in TASKS else []
        if not task_ids:
            print(f"✗ Task {args.task} not found")
            return
    else:
        # Evaluate all tasks with validation sets
        task_ids = [tid for tid in sorted(TASKS.keys()) 
                   if tid != -1 and TASKS[tid].valid_inputs is not None]
    
    print(f"\nEvaluating {len(task_ids)} tasks with validation sets...")
    print("=" * 60)
    
    # Run evaluations
    all_results = []
    for task_id in task_ids:
        task = TASKS[task_id]
        print(f"\nTask {task_id}: {task.description} ({task.mode})")
        print("-" * 50)
        
        try:
            results = evaluate_genome_on_task(genome_data, task_id, args.window, args.max_steps)
            print_evaluation_results(results)
            if results:
                all_results.append(results)
        except Exception as e:
            print(f"  ✗ Error evaluating task {task_id}: {e}")
    
    # Overall summary
    if all_results:
        print(f"\n{'='*60}")
        print("OVERALL SUMMARY")
        print(f"{'='*60}")
        
        total_samples = sum(r['num_samples'] for r in all_results)
        total_successful = sum(r['successful_simulations'] for r in all_results)
        mean_errors = [r['mean_error'] for r in all_results if r['mean_error'] != np.inf]
        exact_accuracies = [r['exact_accuracy'] for r in all_results]
        
        print(f"Tasks evaluated: {len(all_results)}")
        print(f"Total samples: {total_samples}")
        print(f"Overall simulation success: {total_successful}/{total_samples} ({total_successful/total_samples:.1%})")
        if mean_errors:
            print(f"Average error across tasks: {np.mean(mean_errors):.3f}")
        if exact_accuracies:
            print(f"Average exact accuracy: {np.mean(exact_accuracies):.1%}")
        
        print(f"\nBest performing tasks:")
        sorted_results = sorted(all_results, key=lambda x: x['exact_accuracy'], reverse=True)
        for r in sorted_results[:3]:
            print(f"  Task {r['task_id']}: {r['exact_accuracy']:.1%} accuracy ({r['task_description']})")
    
    print(f"\n✓ Evaluation completed!")


if __name__ == "__main__":
    main() 