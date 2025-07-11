"""
EM43 Inference Script - Interactive Model Testing
================================================

Interactive inference with trained EM43 genomes.
Supports very large inputs (up to 10000+) with automatic window sizing.

Usage:
    python inference.py [--genome PATH] [--window SIZE] [--max-steps STEPS]
"""

import numpy as np
import pickle
import argparse
import yaml
from pathlib import Path
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


def calculate_optimal_window(inputs, prog_len, mode="1input", base_window=5000):
    """
    Calculate optimal window size based on input values.
    
    Args:
        inputs: Input values (single value or pair)
        prog_len: Program length
        mode: "1input" or "2input"
        base_window: Base window size
        
    Returns:
        Recommended window size
    """
    if mode == "1input":
        max_input = max(inputs) if isinstance(inputs, (list, tuple)) else inputs
        # Structure: [program] 00 0^(input+1) R 0... (output_space)
        min_required = prog_len + 2 + (max_input + 1) + 1 + 50
    else:  # 2input
        max_a = max(inputs[0]) if isinstance(inputs[0], (list, tuple)) else inputs[0]
        max_b = max(inputs[1]) if isinstance(inputs[1], (list, tuple)) else inputs[1]
        # Structure: [program] 00 0^(a+1) R 0^(b+1) R 0... (output_space)
        min_required = prog_len + 2 + (max_a + 1) + 1 + (max_b + 1) + 1 + 50
    
    # Use larger of base_window or minimum required + 20% buffer
    return max(base_window, int(min_required * 1.2))


def run_inference_1input(genome, inputs, window_size, max_steps):
    """Run inference for 1-input mode."""
    sim = EM43Batch(genome, window=window_size, max_steps=max_steps)
    outputs = sim.run(inputs if isinstance(inputs, list) else [inputs])
    return outputs


def run_inference_2input(genome, inputs_a, inputs_b, window_size, max_steps):
    """Run inference for 2-input mode."""
    sim = EM43TwoInputBatch(genome, window=window_size, max_steps=max_steps)
    
    # Ensure inputs are lists
    if not isinstance(inputs_a, list):
        inputs_a = [inputs_a]
    if not isinstance(inputs_b, list):
        inputs_b = [inputs_b]
    
    outputs = sim.run(inputs_a, inputs_b)
    return outputs


def interactive_inference(genome_data, base_window=10000, max_steps=2000):
    """Interactive inference session."""
    rule = genome_data['best_rule']
    prog = genome_data['best_prog']
    genome = (rule, prog)
    mode = genome_data.get('mode', '1input')
    prog_len = len(prog)
    
    print(f"\n🔬 Interactive Inference Mode: {mode.upper()}")
    print("=" * 55)
    print("Enter inputs to get predictions from the evolved model")
    print("Supports very large values (tested up to 10000+)")
    print("Type 'quit', 'exit', or 'q' to stop")
    print()
    
    if mode == "1input":
        print("📝 Format: single integer")
        print("   Examples: 42, 1000, 5000, 10000")
        
        while True:
            try:
                user_input = input("\n🔢 Input: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not user_input:
                    continue
                
                value = int(user_input)
                
                # Calculate optimal window for this input
                window = calculate_optimal_window(value, prog_len, mode, base_window)
                
                print(f"   Processing... (window={window}, max_steps={max_steps})")
                outputs = run_inference_1input(genome, [value], window, max_steps)
                
                if outputs[0] == -10:
                    print(f"⚠️  Simulation failed (try increasing window or max_steps)")
                else:
                    print(f"📤 Output: {outputs[0]}")
                
            except ValueError:
                print("❌ Please enter a valid integer")
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error during inference: {e}")
    
    else:  # 2input mode
        print("📝 Format: two integers separated by comma or space")
        print("   Examples: 42,37  or  1000 500  or  10000,8000")
        
        while True:
            try:
                user_input = input("\n🔢 Input (a,b): ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not user_input:
                    continue
                
                # Parse input (comma or space separated)
                if ',' in user_input:
                    parts = user_input.split(',')
                else:
                    parts = user_input.split()
                
                if len(parts) != 2:
                    print("❌ Please enter exactly two values separated by comma or space")
                    continue
                
                value_a = int(parts[0].strip())
                value_b = int(parts[1].strip())
                
                # Calculate optimal window for these inputs
                window = calculate_optimal_window([value_a, value_b], prog_len, mode, base_window)
                
                print(f"   Processing... (window={window}, max_steps={max_steps})")
                outputs = run_inference_2input(genome, [value_a], [value_b], window, max_steps)
                
                if outputs[0] == -10:
                    print(f"⚠️  Simulation failed (try increasing window or max_steps)")
                else:
                    print(f"📤 Output: {outputs[0]}")
                
            except ValueError:
                print("❌ Please enter valid integers")
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error during inference: {e}")


def batch_inference(genome_data, base_window=10000, max_steps=2000):
    """Batch inference with predefined large test cases."""
    rule = genome_data['best_rule']
    prog = genome_data['best_prog']
    genome = (rule, prog)
    mode = genome_data.get('mode', '1input')
    prog_len = len(prog)
    
    print(f"\n🧪 Batch Inference Mode: {mode.upper()}")
    print("=" * 55)
    print("Testing with predefined large inputs...")
    
    if mode == "1input":
        # Test cases with progressively larger values
        test_inputs = [1, 5, 10, 50, 100, 500, 1000, 2500, 5000, 10000]
        
        print(f"\n📊 Testing {len(test_inputs)} input values:")
        print("-" * 40)
        
        for inp in test_inputs:
            try:
                window = calculate_optimal_window(inp, prog_len, mode, base_window)
                outputs = run_inference_1input(genome, [inp], window, max_steps)
                
                if outputs[0] == -10:
                    status = "⚠️  FAILED"
                else:
                    status = f"📤 {outputs[0]}"
                
                print(f"  {inp:>5} → {status}")
                
            except Exception as e:
                print(f"  {inp:>5} → ❌ Error: {e}")
    
    else:  # 2input mode
        # Test cases with progressively larger value pairs
        test_pairs = [
            (1, 1), (5, 3), (10, 20), (25, 15), (50, 75), 
            (100, 200), (500, 300), (1000, 800), (2500, 1500), 
            (5000, 3000), (10000, 7000)
        ]
        
        print(f"\n📊 Testing {len(test_pairs)} input pairs:")
        print("-" * 50)
        
        for inp_a, inp_b in test_pairs:
            try:
                window = calculate_optimal_window([inp_a, inp_b], prog_len, mode, base_window)
                outputs = run_inference_2input(genome, [inp_a], [inp_b], window, max_steps)
                
                if outputs[0] == -10:
                    status = "⚠️  FAILED"
                else:
                    status = f"📤 {outputs[0]}"
                
                print(f"  ({inp_a:>4},{inp_b:>4}) → {status}")
                
            except Exception as e:
                print(f"  ({inp_a:>4},{inp_b:>4}) → ❌ Error: {e}")
    
    print(f"\n✅ Batch testing completed!")


def benchmark_inference(genome_data, base_window=15000, max_steps=3000):
    """Benchmark inference with extremely large inputs."""
    rule = genome_data['best_rule']
    prog = genome_data['best_prog']
    genome = (rule, prog)
    mode = genome_data.get('mode', '1input')
    prog_len = len(prog)
    
    print(f"\n🚀 Benchmark Mode: {mode.upper()} (Extreme Large Inputs)")
    print("=" * 60)
    print("Testing computational limits with very large inputs...")
    
    if mode == "1input":
        # Extreme test cases
        extreme_inputs = [10000, 25000, 50000, 100000]
        
        print(f"\n🔥 Testing {len(extreme_inputs)} extreme input values:")
        print("-" * 50)
        
        for inp in extreme_inputs:
            try:
                window = calculate_optimal_window(inp, prog_len, mode, base_window)
                print(f"  Testing {inp} (window={window})... ", end="", flush=True)
                
                outputs = run_inference_1input(genome, [inp], window, max_steps)
                
                if outputs[0] == -10:
                    print("⚠️  SIMULATION FAILED")
                else:
                    print(f"✅ {outputs[0]}")
                
            except Exception as e:
                print(f"❌ ERROR: {e}")
    
    else:  # 2input mode
        # Extreme test pairs
        extreme_pairs = [
            (10000, 5000), (25000, 15000), (50000, 30000), (100000, 75000)
        ]
        
        print(f"\n🔥 Testing {len(extreme_pairs)} extreme input pairs:")
        print("-" * 60)
        
        for inp_a, inp_b in extreme_pairs:
            try:
                window = calculate_optimal_window([inp_a, inp_b], prog_len, mode, base_window)
                print(f"  Testing ({inp_a},{inp_b}) (window={window})... ", end="", flush=True)
                
                outputs = run_inference_2input(genome, [inp_a], [inp_b], window, max_steps)
                
                if outputs[0] == -10:
                    print("⚠️  SIMULATION FAILED")
                else:
                    print(f"✅ {outputs[0]}")
                
            except Exception as e:
                print(f"❌ ERROR: {e}")
    
    print(f"\n🏁 Benchmark completed!")


def main():
    """Main inference function."""
    # Load config to get default values
    try:
        config = load_config()
        default_window = config['evaluation']['window']['value']
        default_max_steps = config['evaluation']['max_steps']['value']
    except Exception as e:
        print(f"Warning: Could not load config, using hardcoded defaults: {e}")
        default_window = 10000
        default_max_steps = 3000
    
    parser = argparse.ArgumentParser(description="Interactive EM43 inference")
    parser.add_argument("--genome", default="dp_checkpoints/best_genome.pkl", 
                       help="Path to saved genome file")
    parser.add_argument("--window", type=int, default=default_window,
                       help=f"Base window size (default from config: {default_window}, auto-adjusted for large inputs)")
    parser.add_argument("--max-steps", type=int, default=default_max_steps,
                       help=f"Maximum simulation steps (default from config: {default_max_steps})")
    parser.add_argument("--mode", choices=["interactive", "batch", "benchmark"], 
                       default="interactive", help="Inference mode")
    parser.add_argument("--config", default="config.yaml",
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    print("🧠 EM43 Interactive Inference")
    print("=" * 40)
    
    # Load genome
    try:
        genome_data = load_genome(args.genome)
        print(f"✅ Loaded genome: {args.genome}")
        print(f"   Fitness: {genome_data.get('best_fitness', 'unknown')}")
        print(f"   Mode: {genome_data.get('mode', 'unknown')}")
        print(f"   Program length: {len(genome_data['best_prog'])}")
    except Exception as e:
        print(f"❌ Error loading genome: {e}")
        return
    
    print(f"\n⚙️  Parameters:")
    print(f"   Base window: {args.window}")
    print(f"   Max steps: {args.max_steps}")
    print(f"   Mode: {args.mode}")
    
    # Run selected inference mode
    try:
        if args.mode == "interactive":
            interactive_inference(genome_data, args.window, args.max_steps)
        elif args.mode == "batch":
            batch_inference(genome_data, args.window, args.max_steps)
        elif args.mode == "benchmark":
            benchmark_inference(genome_data, args.window, args.max_steps)
        
    except KeyboardInterrupt:
        print("\n\n👋 Inference stopped by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    
    print("\n✨ Inference session completed!")


if __name__ == "__main__":
    main() 