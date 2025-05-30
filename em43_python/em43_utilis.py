import argparse
import yaml
from pathlib import Path
import sys
import numpy as np

# Type conversion dictionary
TYPE_MAP = {
    'int': int,
    'float': float,
    'str': str,
    'list': list
}

# Load configuration from YAML file
CONFIG_FILE = Path(__file__).parent / 'config.yaml'

try:
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print(f"Error: Configuration file {CONFIG_FILE} not found")
    sys.exit(1)
except yaml.YAMLError as e:
    print(f"Error parsing YAML configuration: {e}")
    sys.exit(1)

# Convert YAML config to the original dictionary format
ARGUMENTS = {}
for section, params in config.items():
    if section == 'config_file':
        continue
    for param_name, param_config in params.items():
        # Convert type string to actual Python type
        param_config['type'] = TYPE_MAP[param_config['type']]
        param_config['section'] = section.title()
        ARGUMENTS[param_name] = param_config


def get_args():
    parser = argparse.ArgumentParser(description='EM-4/3 GA with Numba optimization')
    
    for name, config_dict in ARGUMENTS.items():
        parser.add_argument(
            config_dict['short'],
            f"--{name}",
            type=config_dict['type'],
            default=config_dict['default'],
            help=config_dict['help']
        )
    
    parser.add_argument(
        '--config',
        type=str,
        default=str(CONFIG_FILE),
        help=f'Path to configuration file (default: {CONFIG_FILE})'
    )
    
    args = parser.parse_args()

    try:
        if len(eval(args.input_set, {'np': np})) != len(eval(args.target_out, {'np': np})):
            raise ValueError("Input set and target output must have the same length") 
    except Exception as e:
        print(f"Error missing input set or target output: {e}")
        sys.exit(1)
    
    if args.config != str(CONFIG_FILE):
        config_file = Path(args.config)
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    _ = yaml.safe_load(f)
            except yaml.YAMLError as e:
                print(f"Error parsing YAML configuration: {e}")
                sys.exit(1)
        else:
            print(f"Warning: Config file {config_file} not found, using default values")
    
    return args