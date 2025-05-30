import argparse
import yaml
from pathlib import Path
import sys
import numpy as np
from datetime import datetime

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

def get_args():
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

    # Create argument parser
    parser = argparse.ArgumentParser(description='EM-4/3 GA with Numba optimization')
    
    # Add all arguments from the dictionary
    for name, config_dict in ARGUMENTS.items():
        parser.add_argument(
            config_dict['short'],
            f"--{name}",
            type=config_dict['type'],
            default=config_dict['value'],
            help=config_dict['help']
        )
    
    parser.add_argument(
        '--config',
        type=str,
        default=str(CONFIG_FILE),
        help=f'Path to configuration file (default: config.yaml)'
    )
    
    args = parser.parse_args()

    try:
        # Check input and target output lengths
        if len(eval(args.input_set, {'np': np})) != len(eval(args.target_out, {'np': np})):
            raise ValueError("Input set and target output must have the same length") 
    except Exception as e:
        print(f"Error missing input set or target output: {e}")
        sys.exit(1)
    
    # Create timestamped copy of config
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    copy_filename = f"setup_{timestamp}.yaml"
    
    # Create a new config structure that matches the original format
    updated_config = {}
    
    # First copy the config file path
    updated_config['config_file'] = config['config_file']
    
    # Then update each section in order
    for section, params in config.items():
        if section == 'config_file':
            continue
            
        updated_section = {}
        for param_name, param_config in params.items():
            # Get the value from command line arguments
            value = getattr(args, param_name)
            print(value)
            
            # Convert the type back to string using TYPE_MAP
            type_str = next(key for key, val in TYPE_MAP.items() if val == type(value))
            
            # Create a new parameter dictionary with the same structure
            updated_param = {
                'short': param_config['short'],
                'type': type_str,
                'value': value,
                'help': param_config['help']
            }
            
            updated_section[param_name] = updated_param
        
        updated_config[section] = updated_section
        print(updated_config)
    # Write the updated config to the timestamped file
    with open(copy_filename, 'w') as f:
        yaml.dump(updated_config, f, 
                 default_flow_style=False,
                 sort_keys=False,
                 allow_unicode=True)
    print(f"\nSetup parameters saved to {copy_filename}")

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