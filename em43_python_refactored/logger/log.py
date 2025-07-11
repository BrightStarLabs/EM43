#!/usr/bin/env python3
"""
EM43 Training Data Logger
Handles logging of training data to the distributed API system
"""

import json
import requests
import sys
import os
from datetime import datetime
from pathlib import Path
import yaml
import uuid
import time
import traceback

class EM43Logger:
    def __init__(self, config_path="config_log.yaml"):
        """Initialize the logger"""
        self.config_path = Path(__file__).parent / config_path
        self.user_config_path = Path(__file__).parent / "user_config.json"
        self.config = self.load_config()
        self.user_data = self.load_user_config()
        self.current_run_id = None
        
    def load_config(self):
        """Load logger configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config
        except FileNotFoundError:
            print(f"⚠️  Warning: Config file {self.config_path} not found!")
            print("Using default configuration.")
            return self.get_default_config()
        except yaml.YAMLError as e:
            print(f"⚠️  Warning: Error parsing config file: {e}")
            print("Using default configuration.")
            return self.get_default_config()
    
    def get_default_config(self):
        """Get default configuration"""
        return {
            'logging': {
                'enabled': True,
                'api_endpoint': 'https://jkk4nk8j9g.execute-api.us-east-1.amazonaws.com/api',
                'timeout': 20,
                'default_model': 'EM43',
                'auto_detect_mode': True,
                'log_checkpoints': True,
                'log_initial': True,
                'log_evaluation': False
            }
        }
    
    def load_user_config(self):
        """Load user configuration"""
        try:
            with open(self.user_config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("⚠️  User not registered! Please run register.py first.")
            return None
        except json.JSONDecodeError:
            print("⚠️  Invalid user config file! Please run register.py again.")
            return None
    
    def is_enabled(self):
        """Check if logging is enabled"""
        return self.config['logging']['enabled'] and self.user_data is not None
    
    def generate_run_id(self):
        """Generate unique run ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = str(uuid.uuid4())[:8]
        return f"run_{timestamp}_{random_suffix}"
    
    def get_run_id(self):
        """Get current run ID, generate if needed"""
        if self.current_run_id is None:
            self.current_run_id = self.generate_run_id()
        return self.current_run_id
    
    def detect_mode(self, task_config=None, task_id=None, num_inputs=None):
        """
        Detect if task is 1-input or 2-input mode
        
        Args:
            task_config: Task configuration dictionary
            task_id: Task ID number
            num_inputs: Number of inputs (if known)
        
        Returns:
            str: "1input" or "2input"
        """
        # If explicitly provided
        if num_inputs is not None:
            return "2input" if num_inputs == 2 else "1input"
        
        # Check task_config
        if task_config:
            # Look for indicators of 2-input mode
            if any(key in task_config for key in ['input1', 'input2', 'two_inputs', 'num_inputs']):
                num_inputs = task_config.get('num_inputs', 1)
                return "2input" if num_inputs == 2 else "1input"
            
            # Check task description for common 2-input patterns
            description = task_config.get('description', '').lower()
            two_input_patterns = ['sum', 'add', 'addition', 'plus', 'combine', 'merge', 'two input']
            if any(pattern in description for pattern in two_input_patterns):
                return "2input"
        
        # Check by task_id (common patterns)
        if task_id is not None:
            # Assume task_id > 10 might be 2-input (this is a heuristic)
            # This can be adjusted based on your specific task numbering
            if task_id > 10:
                return "2input"
        
        # Default to 1-input
        return "1input"
    
    def extract_task_info(self, task_config):
        """Extract task information from config"""
        if not task_config:
            return {
                'task_id': None,
                'task_description': 'Unknown task',
                'EDH': 'unknown',
                'mode': '1input'
            }
        
        task_id = task_config.get('task_id', task_config.get('id', None))
        description = task_config.get('description', task_config.get('task_description', 'Unknown task'))
        
        # Try to extract EDH from different possible locations
        edh = task_config.get('EDH', 
                             task_config.get('edh', 
                                           task_config.get('evolution_description', 'unknown')))
        
        mode = self.detect_mode(task_config, task_id)
        
        return {
            'task_id': task_id,
            'task_description': description,
            'EDH': edh,
            'mode': mode
        }
    
    def log_training_data(self, generation=0, best_fitness=None, avg_fitness=None, 
                         task_config=None, checkpoint_type="training", 
                         eval_fitness=None, additional_data=None,
                         population_size=None, mutation_rate=None, 
                         crossover_rate=None, selection_method=None,
                         training_params=None):
        """
        Log training data to the API
        
        Args:
            generation: Current generation number
            best_fitness: Best fitness value
            avg_fitness: Average fitness value
            task_config: Task configuration dictionary
            checkpoint_type: Type of checkpoint ("initial", "training", "evaluation")
            eval_fitness: Evaluation fitness (optional)
            additional_data: Additional data to log
            population_size: Population size
            mutation_rate: Mutation rate
            crossover_rate: Crossover rate
            selection_method: Selection method
            training_params: Additional training parameters
        """
        if not self.is_enabled():
            return False
        
        try:
            # Extract task information
            task_info = self.extract_task_info(task_config)
            
            # Prepare data structure
            data = {
                "model": self.config['logging']['default_model'],
                "mode": task_info['mode'],
                "EDH": task_info['EDH'],
                "task_id": task_info['task_id'],
                "task_description": task_info['task_description'],
                "run_id": self.get_run_id(),
                "generation": generation,
                "checkpoint_type": checkpoint_type,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add fitness data
            if best_fitness is not None:
                data["best_fitness"] = float(best_fitness)
            if avg_fitness is not None:
                data["avg_fitness"] = float(avg_fitness)
            if eval_fitness is not None:
                data["eval_fitness"] = float(eval_fitness)
            
            # Add training parameters
            if population_size is not None:
                data["population_size"] = population_size
            if mutation_rate is not None:
                data["mutation_rate"] = mutation_rate
            if crossover_rate is not None:
                data["crossover_rate"] = crossover_rate
            if selection_method is not None:
                data["selection_method"] = selection_method
            
            # Add training parameters dictionary
            if training_params:
                data["training_params"] = training_params
            
            # Add additional data
            if additional_data:
                data.update(additional_data)
            
            # Prepare API request
            payload = {
                "user_id": self.user_data['user_id'],
                "username": self.user_data['username'],
                "data": data
            }
            
            # Send to API
            return self.send_to_api(payload)
            
        except Exception as e:
            print(f"⚠️  Error preparing training data: {e}")
            if self.config['logging'].get('debug', False):
                traceback.print_exc()
            return False
    
    def send_to_api(self, payload):
        """Send data to the API"""
        api_endpoint = self.config['logging']['api_endpoint']
        timeout = self.config['logging']['timeout']
        
        try:
            response = requests.post(
                f"{api_endpoint}/log",
                json=payload,
                timeout=timeout,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                checkpoint_time = result.get('checkpoint_timestamp', 'unknown')
                print(f"✅ Training data logged successfully [{checkpoint_time}]")
                return True
            else:
                print(f"⚠️  API logging failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            print(f"⚠️  API request timed out after {timeout} seconds")
            return False
        except requests.exceptions.ConnectionError:
            print("⚠️  Cannot connect to API - continuing without logging")
            return False
        except requests.exceptions.RequestException as e:
            print(f"⚠️  API request error: {e}")
            return False
    
    def log_initial(self, task_config, initial_fitness=None, population_size=None, 
                   training_params=None):
        """Log initial training state (t=0)"""
        if not self.config['logging']['log_initial']:
            return True
            
        return self.log_training_data(
            generation=0,
            best_fitness=initial_fitness,
            avg_fitness=initial_fitness,
            task_config=task_config,
            checkpoint_type="initial",
            population_size=population_size,
            training_params=training_params
        )
    
    def log_checkpoint(self, generation, best_fitness, task_config,
                      population_size=None, training_params=None, additional_data=None):
        """Log checkpoint during training"""
        if not self.config['logging']['log_checkpoints']:
            return True
            
        return self.log_training_data(
            generation=generation,
            best_fitness=best_fitness,
            avg_fitness=None,
            task_config=task_config,
            checkpoint_type="training",
            population_size=population_size,
            training_params=training_params,
            additional_data=additional_data
        )
    
    def log_evaluation(self, generation, best_fitness, avg_fitness, task_config,
                      eval_fitness, additional_data=None):
        """Log evaluation results"""
        if not self.config['logging']['log_evaluation']:
            return True
            
        return self.log_training_data(
            generation=generation,
            best_fitness=best_fitness,
            avg_fitness=avg_fitness,
            task_config=task_config,
            checkpoint_type="evaluation",
            eval_fitness=eval_fitness,
            additional_data=additional_data
        )
    
    def test_connection(self):
        """Test API connection"""
        if not self.user_data:
            print("❌ No user data available. Please register first.")
            return False
            
        api_endpoint = self.config['logging']['api_endpoint']
        timeout = self.config['logging']['timeout']
        
        try:
            # Test health endpoint
            response = requests.get(f"{api_endpoint}/health", timeout=timeout)
            if response.status_code == 200:
                print("✅ API health check passed")
                
                # Test user verification
                user_id = self.user_data['user_id']
                response = requests.get(f"{api_endpoint}/user/{user_id}/verify", timeout=timeout)
                if response.status_code == 200:
                    print("✅ User verification passed")
                    return True
                else:
                    print(f"❌ User verification failed: {response.status_code}")
                    return False
            else:
                print(f"❌ API health check failed: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ API connection test failed: {e}")
            return False
    
    def get_status(self):
        """Get logger status"""
        status = {
            'enabled': self.is_enabled(),
            'user_registered': self.user_data is not None,
            'config_loaded': self.config is not None,
            'current_run_id': self.current_run_id,
            'api_endpoint': self.config['logging']['api_endpoint']
        }
        
        if self.user_data:
            status['username'] = self.user_data['username']
            status['user_id'] = self.user_data['user_id']
        
        return status

# Global logger instance
_logger_instance = None

def get_logger():
    """Get global logger instance"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = EM43Logger()
    return _logger_instance

def log_training_data(*args, **kwargs):
    """Convenience function for logging training data"""
    logger = get_logger()
    return logger.log_training_data(*args, **kwargs)

def log_initial(*args, **kwargs):
    """Convenience function for logging initial state"""
    logger = get_logger()
    return logger.log_initial(*args, **kwargs)

def log_checkpoint(*args, **kwargs):
    """Convenience function for logging checkpoints"""
    logger = get_logger()
    return logger.log_checkpoint(*args, **kwargs)

def log_evaluation(*args, **kwargs):
    """Convenience function for logging evaluation"""
    logger = get_logger()
    return logger.log_evaluation(*args, **kwargs)

def test_connection():
    """Test API connection"""
    logger = get_logger()
    return logger.test_connection()

def get_status():
    """Get logger status"""
    logger = get_logger()
    return logger.get_status()

if __name__ == "__main__":
    # Test the logger
    logger = EM43Logger()
    print("🧪 Testing EM43 Logger")
    print("=" * 30)
    
    # Print status
    status = logger.get_status()
    print(f"Logger enabled: {status['enabled']}")
    print(f"User registered: {status['user_registered']}")
    print(f"Config loaded: {status['config_loaded']}")
    
    if status['user_registered']:
        print(f"Username: {status['username']}")
        print(f"User ID: {status['user_id']}")
    
    # Test connection
    if logger.test_connection():
        print("✅ Logger is ready for use!")
    else:
        print("❌ Logger has issues. Check configuration and registration.") 