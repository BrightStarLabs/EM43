#!/usr/bin/env python3
"""
EM43 Unified System - Usage Examples
===================================

This script demonstrates how to use the unified EM43 system for:
- Training models on different tasks
- Evaluating performance 
- Running inference
- Both 1-input and 2-input modes

Run this script to see the unified system in action!
"""

import sys
from pathlib import Path

# Add parent directory to Python path to import EM43 modules
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from em43_wrapper import EM43Wrapper

def example_1input_task():
    """Example: Training and using a 1-input task (multiply by 2)."""
    print("🔢 Example 1: 1-Input Task (Multiply by 2)")
    print("=" * 50)
    
    # Create wrapper for Task 1 (multiply by 2)
    model = EM43Wrapper(
        task_id=1,
        config_overrides={
            'population': {'pop_size': {'value': 100}, 'generations': {'value': 20}},
            'model': {'prog_len': {'value': 10}}
        }
    )
    
    # Train the model
    print("\n🚀 Training...")
    model.train()
    
    # Evaluate on validation set
    print("\n📊 Evaluating...")
    results = model.evaluate()
    
    # Run custom inference
    print("\n🔮 Custom inference...")
    test_inputs = [11, 12, 13, 14, 15]  # Out-of-training range
    outputs = model.infer(test_inputs)
    
    print(f"\nResults summary:")
    print(f"- Training accuracy: {results['accuracy']:.1%}")
    print(f"- Custom inference: {dict(zip(test_inputs, outputs))}")
    
    return model


def example_2input_task():
    """Example: Training and using a 2-input task (summation)."""
    print("\n🔢 Example 2: 2-Input Task (Summation)")
    print("=" * 50)
    
    # Create wrapper for Task 20 (summation)
    model = EM43Wrapper(
        task_id=20,
        config_overrides={
            'population': {'pop_size': {'value': 100}, 'generations': {'value': 20}},
            'model': {'prog_len': {'value': 12}}
        }
    )
    
    # Train the model
    print("\n🚀 Training...")
    model.train()
    
    # Evaluate on validation set
    print("\n📊 Evaluating...")
    results = model.evaluate()
    
    # Run custom inference
    print("\n🔮 Custom inference...")
    test_a = [6, 7, 8, 9, 10]
    test_b = [4, 3, 2, 1, 0]
    outputs = model.infer((test_a, test_b))
    
    print(f"\nResults summary:")
    print(f"- Training accuracy: {results['accuracy']:.1%}")
    print(f"- Custom inference:")
    for a, b, out in zip(test_a, test_b, outputs):
        print(f"  {a} + {b} = {out} (expected: {a+b})")
    
    return model


def example_quick_evaluation():
    """Example: Quick evaluation of multiple tasks."""
    print("\n🔢 Example 3: Quick Multi-Task Evaluation")
    print("=" * 50)
    
    # Test several tasks quickly
    test_tasks = [1, 2, 6, 7, 20, 21]  # Mix of 1-input and 2-input
    quick_config = {
        'population': {'pop_size': {'value': 50}, 'generations': {'value': 10}},
        'model': {'prog_len': {'value': 8}}
    }
    
    results_summary = []
    
    for task_id in test_tasks:
        print(f"\n🧠 Testing Task {task_id}...")
        
        try:
            model = EM43Wrapper(task_id=task_id, config_overrides=quick_config)
            model.train(verbose=False)
            eval_results = model.evaluate(verbose=False, plot=False)
            
            results_summary.append({
                'task_id': task_id,
                'description': model.task.description,
                'mode': model.mode,
                'accuracy': eval_results['accuracy'],
                'fitness': model.best_fitness
            })
            
            print(f"   ✅ {model.task.description}: {eval_results['accuracy']:.1%} accuracy")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    # Summary table
    print(f"\n📊 Summary of {len(results_summary)} tasks:")
    print("-" * 70)
    print(f"{'Task':<6} {'Mode':<8} {'Description':<20} {'Accuracy':<10} {'Fitness':<10}")
    print("-" * 70)
    for r in results_summary:
        print(f"{r['task_id']:<6} {r['mode']:<8} {r['description']:<20} {r['accuracy']:<10.1%} {r['fitness']:<10.3f}")


def example_advanced_usage():
    """Example: Advanced usage patterns."""
    print("\n🔢 Example 4: Advanced Usage Patterns")
    print("=" * 50)
    
    # Load existing genome and run inference
    try:
        # Try to load a previously saved genome
        model = EM43Wrapper(task_id=1)
        if model.load_genome(verbose=False):
            print("✅ Loaded existing genome")
            
            # Get model info
            info = model.get_info()
            print(f"   Task: {info['task_description']}")
            print(f"   Program: {info.get('prog_string', 'N/A')}")
            print(f"   Fitness: {info.get('best_fitness', 'N/A')}")
            
            # Quick inference
            outputs = model.infer([100, 200, 300], verbose=False)
            print(f"   Large input test: {outputs}")
        else:
            print("❌ No existing genome found - training new one...")
            model.train(verbose=False)
            
    except Exception as e:
        print(f"❌ Advanced example failed: {e}")


def main():
    """Run all examples."""
    print("🧠 EM43 Unified System - Usage Examples")
    print("=" * 60)
    print("This script demonstrates the key features of the refactored EM43 system.")
    print("Each example shows different aspects of training, evaluation, and inference.")
    print("=" * 60)
    
    try:
        # Run examples
        model1 = example_1input_task()
        model2 = example_2input_task()
        example_quick_evaluation()
        example_advanced_usage()
        
        print("\n" + "=" * 60)
        print("🎉 All examples completed successfully!")
        print("=" * 60)
        print("\nKey takeaways:")
        print("✅ Unified interface for both 1-input and 2-input tasks")
        print("✅ Automatic mode detection based on task_id")
        print("✅ Easy configuration overrides")
        print("✅ Seamless train → evaluate → infer pipeline")
        print("✅ Comprehensive visualization and reporting")
        print("✅ Robust genome save/load functionality")
        
        print(f"\nTo explore more:")
        print(f"• python demo.py --list-tasks          # See all available tasks")
        print(f"• python demo.py --interactive         # Interactive task selection")
        print(f"• python demo.py --task 1 --stage all  # Full pipeline for Task 1")
        
    except KeyboardInterrupt:
        print("\n\n👋 Examples interrupted by user")
    except Exception as e:
        print(f"\n💥 Example failed with error: {e}")
        print("This might indicate an issue with the system setup.")


if __name__ == "__main__":
    main() 