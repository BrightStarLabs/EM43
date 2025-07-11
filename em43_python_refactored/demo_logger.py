#!/usr/bin/env python3
"""
EM43 Unified Demo with Distributed Logging - Complete Training, Evaluation & Inference Pipeline
==============================================================================================

Enhanced version of demo.py with automatic distributed logging to cloud API.
Features automatic registration, health checks, and graceful offline fallback.

Key Features:
- Automatic API health check on startup
- Automatic user registration if not registered
- Graceful offline mode if API is unavailable
- Logs initial state, checkpoints at intervals, and best genome
- Logs evaluation metrics (accuracy, errors, success rate)
- Complete rule and program arrays saved for best genome
- All original demo.py functionality preserved

Logging Behavior:
- Initial state logged at generation 0
- Checkpoints logged at intervals defined in config (default: every 100 generations)
- Final best genome logged if not already at checkpoint
- Evaluation results logged when evaluation stage is run

Usage Examples:
    # Basic training with logging
    python demo_logger.py --task 1 --stage train      # Train with automatic logging
    
    # Full pipeline with logging
    python demo_logger.py --task 2 --stage all       # Train → Evaluate → Infer with logging
    
    # Works offline if API is unavailable
    python demo_logger.py --task 1 --stage train      # Continues without logging if offline
    
    # All original demo.py functionality works:
    python demo_logger.py --interactive               # Browse and select tasks
    python demo_logger.py --task 1 --pop-size 1000 --generations 50
    python demo_logger.py --task 1 --stage infer --inputs "1,2,3,4,5"
    python demo_logger.py --task 20 --stage infer --inputs "1,2,3" --inputs-b "4,5,6"
    python demo_logger.py --task 1 --stage evaluate --checkpoint best_genome_task_1.pkl
    
Logged Data Includes:
- Initial state (t=0)
- Checkpoints at configured intervals (e.g., every 100 generations)
- Final best genome with complete rule[64] and program[N] arrays
- Evaluation metrics (accuracy, mean error, max error, success rate)
- Task information (ID, description, mode)
- Training parameters (population size, mutation rate, etc.)
"""

import argparse
import sys
import time
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def arr_to_str(arr):
    """Convert a NumPy array / list of ints into a compact digit string.

    This helper is used when logging genomes (rule/program) so that the payload
    remains lightweight while still conveying the complete information.
    Accepts a NumPy array, list, or tuple and returns a string such as
    "0013012". If *arr* is None it returns None so the caller can decide how
    to handle missing data.
    """
    if arr is None:
        return None

    # Convert numpy arrays to Python list for consistent handling
    if hasattr(arr, "tolist"):
        arr = arr.tolist()

    # Ensure we iterate over a flat sequence of ints and cast to str
    return "".join(str(int(x)) for x in arr)

# Import our unified components
from em43_wrapper import EM43Wrapper
from tasks_config import TASKS


# Logger initialization with comprehensive error handling
def initialize_logger():
    """Initialize logger with health checks and automatic registration."""
    logger_available = False
    logger_instance = None
    
    try:
        # Import logger module
        from logger.log import EM43Logger
        from logger.register import UserRegistration
        
        # Create logger instance
        logger_instance = EM43Logger()
        
        # Check logger status
        status = logger_instance.get_status()
        
        print("\n🌐 Distributed Logging System Check")
        print("=" * 50)
        
        # 1. API Health Check
        print("📡 Checking API connection...")
        try:
            import requests
            api_endpoint = logger_instance.config['logging']['api_endpoint']
            response = requests.get(f"{api_endpoint}/health", timeout=5)
            if response.status_code == 200:
                print("✅ API is healthy")
            else:
                print(f"⚠️  API returned status {response.status_code}")
                raise Exception("API unhealthy")
        except Exception as e:
            print(f"⚠️  API health check failed: {e}")
            print("📴 Continuing in OFFLINE mode")
            return False, None
        
        # 2. Check if user is registered
        if not status['user_registered']:
            print("\n👤 User not registered. Starting automatic registration...")
            try:
                # Try to run registration
                registrar = UserRegistration()
                if registrar.run_registration():
                    print("✅ Registration successful!")
                    # Reload logger with new user config
                    logger_instance = EM43Logger()
                    status = logger_instance.get_status()
                else:
                    print("❌ Registration failed")
                    raise Exception("Registration failed")
            except Exception as e:
                print(f"⚠️  Registration error: {e}")
                print("📴 Continuing in OFFLINE mode")
                return False, None
        
        # 3. Verify user is valid
        if status['user_registered']:
            print(f"\n🔍 Verifying user: {status['username']}")
            if logger_instance.test_connection():
                print("✅ User verified and ready")
                logger_available = True
            else:
                print("⚠️  User verification failed")
                print("📴 Continuing in OFFLINE mode")
                return False, None
        
        # 4. Final status
        if logger_available:
            print(f"\n✅ Logging ENABLED")
            print(f"   Username: {status['username']}")
            print(f"   Run ID: {logger_instance.get_run_id()}")
            print("=" * 50)
            return True, logger_instance
        
    except ImportError:
        print("⚠️  Logger module not found")
        print("📴 Continuing in OFFLINE mode")
        return False, None
    except Exception as e:
        print(f"⚠️  Logger initialization error: {e}")
        print("📴 Continuing in OFFLINE mode")
        return False, None
    
    return False, None


# Global logger initialization
LOGGER_AVAILABLE, LOGGER = initialize_logger()


def safe_log(log_func_name: str, *args, **kwargs):
    """Safely call a logging function with error handling."""
    if not LOGGER_AVAILABLE or not LOGGER:
        return False
    
    try:
        # Get the function from logger instance
        log_func = getattr(LOGGER, log_func_name, None)
        if log_func:
            return log_func(*args, **kwargs)
        return False
    except Exception as e:
        print(f"⚠️  Logging error: {e}. Continuing offline.")
        return False


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="EM43 Unified Demo - Training, Evaluation & Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --task 1 --stage train            # Train multiply-by-2 (1-input)
  %(prog)s --task 20 --stage evaluate        # Evaluate summation (2-input)
  %(prog)s --interactive                     # Browse available tasks
  %(prog)s --task 2 --stage all             # Full pipeline
  %(prog)s --task 1 --pop-size 500          # Custom population size
  %(prog)s --task 1 --stage evaluate --checkpoint best_genome_task_1.pkl  # Evaluate with checkpoint
        """
    )
    
    # Task selection
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument(
        "--task", "-t", type=int, 
        help="Task ID from tasks_config.py (use --list-tasks to see all)"
    )
    task_group.add_argument(
        "--interactive", "-i", action="store_true",
        help="Interactive task selection mode"
    )
    task_group.add_argument(
        "--list-tasks", "-l", action="store_true",
        help="List all available tasks and exit"
    )
    
    # Execution stages
    parser.add_argument(
        "--stage", "-s", choices=["train", "evaluate", "infer", "all"],
        default="train", help="Execution stage (default: train)"
    )
    
    # Checkpoint loading
    parser.add_argument(
        "--checkpoint", "-c", type=str,
        help="Checkpoint file to load for evaluation/inference (e.g., best_genome_task_1.pkl)"
    )
    
    # Configuration overrides
    config_group = parser.add_argument_group("Configuration Overrides")
    config_group.add_argument(
        "--pop-size", type=int, help="Population size"
    )
    config_group.add_argument(
        "--generations", type=int, help="Number of generations"
    )
    config_group.add_argument(
        "--prog-len", type=int, help="Program length"
    )
    config_group.add_argument(
        "--window", type=int, help="Simulation window size"
    )
    config_group.add_argument(
        "--max-steps", type=int, help="Maximum simulation steps"
    )
    
    # Inference inputs
    inference_group = parser.add_argument_group("Inference Options")
    inference_group.add_argument(
        "--inputs", type=str, 
        help="Comma-separated inputs (1-input mode) or first inputs (2-input mode)"
    )
    inference_group.add_argument(
        "--inputs-b", type=str,
        help="Second inputs for 2-input mode (comma-separated)"
    )
    
    # General options
    parser.add_argument(
        "--config", default="config.yaml", help="Configuration file path"
    )
    parser.add_argument(
        "--save-dir", default="dp_checkpoints", help="Save directory for genomes"
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Disable plotting"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Minimal output"
    )
    
    return parser


def list_tasks() -> None:
    """List all available tasks with descriptions."""
    print("🧠 Available EM43 Tasks:")
    print("=" * 80)
    
    # Group tasks by mode
    one_input_tasks = [(tid, task) for tid, task in TASKS.items() 
                      if tid != -1 and task.mode == "1input"]
    two_input_tasks = [(tid, task) for tid, task in TASKS.items() 
                      if tid != -1 and task.mode == "2input"]
    
    print("\n📊 1-Input Tasks:")
    print("-" * 40)
    for task_id, task in sorted(one_input_tasks):
        validation_info = f" (+ {len(task.valid_targets)} validation)" if task.valid_inputs is not None else ""
        print(f"  {task_id:2d}: {task.description:<25} ({len(task.targets)} samples{validation_info})")
    
    print("\n🔢 2-Input Tasks:")
    print("-" * 40)
    for task_id, task in sorted(two_input_tasks):
        validation_info = f" (+ {len(task.valid_targets)} validation)" if task.valid_inputs is not None else ""
        print(f"  {task_id:2d}: {task.description:<25} ({len(task.targets)} samples{validation_info})")
    
    if -1 in TASKS:
        print(f"\n⚙️  Custom Task:")
        print("-" * 40)
        print(f"  -1: {TASKS[-1].description}")
    
    print(f"\nTotal: {len([t for t in TASKS.keys() if t != -1])} predefined tasks")


def interactive_task_selection() -> int:
    """Interactive task selection interface."""
    print("🎯 Interactive Task Selection")
    print("=" * 50)
    
    while True:
        print("\nChoose a category:")
        print("  1. 1-Input Tasks (functions of single numbers)")
        print("  2. 2-Input Tasks (operations on pairs of numbers)")
        print("  3. List all tasks")
        print("  4. Exit")
        
        try:
            choice = input("\nEnter choice (1-4): ").strip()
            
            if choice == "1":
                return select_from_category("1input")
            elif choice == "2":
                return select_from_category("2input")
            elif choice == "3":
                list_tasks()
                continue
            elif choice == "4":
                print("👋 Goodbye!")
                sys.exit(0)
            else:
                print("❌ Invalid choice. Please enter 1-4.")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            sys.exit(0)


def select_from_category(mode: str) -> int:
    """Select task from specific category."""
    tasks = [(tid, task) for tid, task in TASKS.items() 
            if tid != -1 and task.mode == mode]
    
    if not tasks:
        print(f"❌ No {mode} tasks available")
        return interactive_task_selection()
    
    print(f"\n📋 {mode.upper()} Tasks:")
    print("-" * 50)
    
    for i, (task_id, task) in enumerate(sorted(tasks), 1):
        validation_info = f" (+ validation)" if task.valid_inputs is not None else ""
        print(f"  {i:2d}. Task {task_id:2d}: {task.description} ({len(task.targets)} samples{validation_info})")
    
    while True:
        try:
            choice = input(f"\nSelect task (1-{len(tasks)}) or 'b' to go back: ").strip()
            
            if choice.lower() == 'b':
                return interactive_task_selection()
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(tasks):
                selected_task_id = sorted(tasks)[choice_idx][0]
                selected_task = TASKS[selected_task_id]
                
                print(f"\n✅ Selected Task {selected_task_id}: {selected_task.description}")
                print(f"   Mode: {selected_task.mode}")
                print(f"   Training samples: {len(selected_task.targets)}")
                if selected_task.valid_inputs is not None:
                    print(f"   Validation samples: {len(selected_task.valid_targets)}")
                
                confirm = input("\nProceed with this task? (y/n): ").strip().lower()
                if confirm in ['y', 'yes', '']:
                    return selected_task_id
                else:
                    continue
            else:
                print(f"❌ Invalid choice. Please enter 1-{len(tasks)} or 'b'.")
        
        except ValueError:
            print("❌ Invalid input. Please enter a number or 'b'.")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            sys.exit(0)


def parse_inputs(inputs_str: str) -> List[int]:
    """Parse comma-separated input string."""
    try:
        return [int(x.strip()) for x in inputs_str.split(",")]
    except ValueError:
        raise ValueError(f"Invalid input format: '{inputs_str}'. Use comma-separated integers.")


def build_config_overrides(args) -> dict:
    """Build configuration overrides from command-line arguments."""
    overrides = {}
    
    if args.pop_size is not None:
        overrides.setdefault('population', {})['pop_size'] = {'value': args.pop_size}
    
    if args.generations is not None:
        overrides.setdefault('population', {})['generations'] = {'value': args.generations}
    
    if args.prog_len is not None:
        overrides.setdefault('model', {})['prog_len'] = {'value': args.prog_len}
    
    if args.window is not None:
        overrides.setdefault('model', {})['window'] = {'value': args.window}
    
    if args.max_steps is not None:
        overrides.setdefault('model', {})['max_steps'] = {'value': args.max_steps}
    
    return overrides


def stage_train(model: EM43Wrapper, args) -> bool:
    """Execute training stage with logging at checkpoint intervals."""
    if not args.quiet:
        print("\n🚀 TRAINING STAGE")
        print("=" * 60)
    
    try:
        # Build task config for logging
        task_config = {
            'task_id': model.task_id,
            'description': model.task.description,
            'EDH': 'unknown',
            'num_inputs': 1 if model.mode == '1input' else 2
        }
        
        # Log initial state (generation 0)
        safe_log('log_initial',
            task_config=task_config,
            initial_fitness=None,
            population_size=model.config['population']['pop_size']['value']
        )
        
        # Define checkpoint callback for real-time logging
        checkpoint_interval = model.config.get('general', {}).get('check_every', {}).get('value', 0)
        def arr_to_str(arr):
            return "".join(map(str, arr.tolist()))

        def checkpoint_cb(gen:int, best_fit:float, rule_arr, prog_arr):
            if not args.quiet and LOGGER_AVAILABLE:
                print(f"   📌 (live) Logging checkpoint at generation {gen}")
            safe_log('log_checkpoint',
                generation=gen,
                best_fitness=best_fit,
                task_config=task_config,
                population_size=model.config['population']['pop_size']['value'],
                additional_data={
                    'rule': arr_to_str(rule_arr),
                    'program': arr_to_str(prog_arr)
                }
            )

        # Train model with live logging
        model.train(verbose=not args.quiet,
                    checkpoint_callback=checkpoint_cb if checkpoint_interval>0 else None,
                    checkpoint_interval=checkpoint_interval)

        history = model.training_history or []
        
        # Always log best genome at the end
        final_gen = len(history)
        if not args.quiet and LOGGER_AVAILABLE:
            print(f"   📌 Logging final best genome at generation {final_gen}")
        safe_log('log_checkpoint',
            generation=final_gen,
            best_fitness=model.best_fitness,
            task_config=task_config,
            population_size=model.config['population']['pop_size']['value'],
            additional_data={
                'rule': arr_to_str(model.best_rule) if model.best_rule is not None else None,
                'program': arr_to_str(model.best_prog) if model.best_prog is not None else None,
                'tag': 'best_genome'
            }
        )

        return True
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False


def stage_evaluate(model: EM43Wrapper, args) -> bool:
    """Execute evaluation stage with logging."""
    if not args.quiet:
        print("\n📊 EVALUATION STAGE")
        print("=" * 60)
    
    try:
        # Load checkpoint if specified
        if args.checkpoint:
            if not args.quiet:
                print(f"📂 Loading checkpoint: {args.checkpoint}")
            
            checkpoint_path = Path(args.checkpoint)
            if not checkpoint_path.is_absolute():
                # Try relative to save directory first
                checkpoint_path = Path(args.save_dir) / args.checkpoint
                if not checkpoint_path.exists():
                    # Try relative to current directory
                    checkpoint_path = Path(args.checkpoint)
            
            success = model.load_genome(checkpoint_path, verbose=not args.quiet)
            if not success:
                print(f"❌ Failed to load checkpoint: {args.checkpoint}")
                return False
        
        # Run evaluation without plotting first
        results = model.evaluate(verbose=not args.quiet, plot=False)
        
        # Log evaluation results
        if results.get('targets') is not None:
            task_config = {
                'task_id': model.task_id,
                'description': model.task.description,
                'EDH': 'unknown',
                'num_inputs': 1 if model.mode == '1input' else 2
            }
            
            if not args.quiet and LOGGER_AVAILABLE:
                print(f"\n📊 Logging evaluation results")
            
            safe_log('log_evaluation',
                generation=model.config['population']['generations']['value'],
                best_fitness=model.best_fitness if model.best_fitness else 0.0,
                avg_fitness=None,
                task_config=task_config,
                eval_fitness=-results['mean_error'],
                additional_data={
                    'accuracy': results['accuracy'],
                    'mean_error': results['mean_error'],
                    'max_error': results['max_error'],
                    'success_rate': results['success_rate'],
                    'num_samples': len(results['targets']),
                    'rule': arr_to_str(model.best_rule) if model.best_rule is not None else None,
                    'program': arr_to_str(model.best_prog) if model.best_prog is not None else None
                }
            )
            
        if not args.quiet:
            print(f"\n📈 Evaluation Summary:")
            print(f"   Accuracy: {results['accuracy']:.1%}")
            print(f"   Mean Error: {results['mean_error']:.3f}")
            print(f"   Success Rate: {results['success_rate']:.1%}")
        
        # After logging, generate plot if requested
        if not args.no_plot and results.get('targets') is not None:
            model._plot_evaluation_results(results)

        return True
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return False


def stage_infer(model: EM43Wrapper, args) -> bool:
    """Execute inference stage."""
    if not args.quiet:
        print("\n🔮 INFERENCE STAGE")
        print("=" * 60)
    
    try:
        # Load checkpoint if specified and not already loaded
        if args.checkpoint and not model.is_trained:
            if not args.quiet:
                print(f"📂 Loading checkpoint: {args.checkpoint}")
            
            checkpoint_path = Path(args.checkpoint)
            if not checkpoint_path.is_absolute():
                # Try relative to save directory first
                checkpoint_path = Path(args.save_dir) / args.checkpoint
                if not checkpoint_path.exists():
                    # Try relative to current directory
                    checkpoint_path = Path(args.checkpoint)
            
            success = model.load_genome(checkpoint_path, verbose=not args.quiet)
            if not success:
                print(f"❌ Failed to load checkpoint: {args.checkpoint}")
                return False
        
        # Determine inputs
        if model.mode == "1input":
            if args.inputs:
                inputs = parse_inputs(args.inputs)
            else:
                # Default 1-input test
                inputs = list(range(1, 11))
                if not args.quiet:
                    print("Using default inputs: 1-10")
            
            outputs = model.infer(inputs, verbose=not args.quiet)
        
        else:  # 2input
            if args.inputs and args.inputs_b:
                inputs_a = parse_inputs(args.inputs)
                inputs_b = parse_inputs(args.inputs_b)
            elif args.inputs:
                # Parse single string for both inputs (assume same values)
                inputs_a = parse_inputs(args.inputs)
                inputs_b = inputs_a.copy()
                if not args.quiet:
                    print(f"Using same inputs for both: {inputs_a}")
            else:
                # Default 2-input test
                inputs_a = [1, 2, 3, 4, 5]
                inputs_b = [1, 2, 3, 4, 5]
                if not args.quiet:
                    print("Using default inputs: (1,1), (2,2), ..., (5,5)")
            
            outputs = model.infer((inputs_a, inputs_b), verbose=not args.quiet)
        
        if not args.quiet:
            print(f"\n✅ Inference completed: {len(outputs)} outputs generated")
        
        return True
    
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return False


def main():
    """Main demo function."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle special modes
    if args.list_tasks:
        list_tasks()
        sys.exit(0)
    
    # Determine task ID
    if args.interactive:
        task_id = interactive_task_selection()
    else:
        task_id = args.task
    
    # Validate task ID
    if task_id not in TASKS:
        print(f"❌ Task {task_id} not found. Use --list-tasks to see available tasks.")
        sys.exit(1)
    
    # Build config overrides
    config_overrides = build_config_overrides(args)
    
    # Initialize wrapper
    if not args.quiet:
        print("🧠 EM43 Unified Demo")
        print("=" * 60)
    
    try:
        model = EM43Wrapper(
            task_id=task_id,
            config_path=args.config,
            config_overrides=config_overrides if config_overrides else None,
            save_dir=args.save_dir
        )
    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
        sys.exit(1)
    
    # Execute stages
    success = True
    start_time = time.time()
    
    if args.stage in ["train", "all"]:
        success &= stage_train(model, args)
        if not success and args.stage == "all":
            print("❌ Stopping pipeline due to training failure")
            sys.exit(1)
    
    if args.stage in ["evaluate", "all"]:
        success &= stage_evaluate(model, args)
        if not success and args.stage == "all":
            print("❌ Stopping pipeline due to evaluation failure")
    
    if args.stage in ["infer", "all"]:
        success &= stage_infer(model, args)
    
    # Final summary
    total_time = time.time() - start_time
    
    if not args.quiet:
        print("\n" + "=" * 60)
        print("DEMO SUMMARY")
        print("=" * 60)
        print(f"Task: {task_id} - {model.task.description} ({model.mode})")
        print(f"Stage(s): {args.stage}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Status: {'✅ SUCCESS' if success else '❌ FAILED'}")
        
        if model.is_trained:
            print(f"Best fitness: {model.best_fitness:.3f}")
            print(f"Program: {model.prog_str(model.best_prog)}")
        
        # Logging status
        if LOGGER_AVAILABLE:
            print(f"\n🌐 Logging: ✅ ENABLED")
            print(f"   Run ID: {LOGGER.get_run_id()}")
        else:
            print(f"\n🌐 Logging: 📴 OFFLINE")
        
        print("\n🎉 Demo completed!")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1) 