# EM43 Logger System

🧬 **Distributed Training Data Logger for EM43 Cellular Automaton Research**

This logger system provides seamless integration with the EM43 training system to log training data to a distributed API for collaborative research.

## 🚀 Quick Start

### 1. First-time Setup

**Step 1**: Register your user account
```bash
python register.py
```

**Step 2**: Test the system
```bash
python log.py
```

**Step 3**: Run the test notebook
```bash
jupyter notebook test_api_registration.ipynb
```

### 2. Integration with Training Scripts

#### **Option A: Use the Enhanced Demo Script**

The easiest way to use logging is with the enhanced `demo_logger.py`:

```bash
# Automatic registration, health checks, and logging
python demo_logger.py --task 1 --stage train

# Full pipeline with logging
python demo_logger.py --task 2 --stage all
```

Features:
- ✅ Automatic API health check
- ✅ Automatic user registration if needed
- ✅ Graceful offline fallback
- ✅ Logs all training data including complete rule/program arrays
- ✅ All original demo.py functionality preserved

#### **Option B: Manual Integration**

```python
# Import logger
from logger.log import get_logger, log_initial, log_checkpoint, log_evaluation

# Initialize logger
logger = get_logger()

# Log initial state (t=0)
log_initial(
    task_config=your_task_config,
    initial_fitness=0.0,
    population_size=200,
    training_params={
        'mutation_rate': 0.02,
        'crossover_rate': 0.8
    }
)

# Log checkpoints during training
log_checkpoint(
    generation=current_generation,
    best_fitness=best_fitness,
    avg_fitness=avg_fitness,
    task_config=your_task_config,
    population_size=200
)

# Log evaluation results
log_evaluation(
    generation=final_generation,
    best_fitness=final_best_fitness,
    avg_fitness=final_avg_fitness,
    task_config=your_task_config,
    eval_fitness=evaluation_score
)
```

## 📁 File Structure

```
logger/
├── register.py              # User registration script
├── log.py                   # Main logging module
├── config_log.yaml          # Logger configuration
├── test_api_registration.ipynb  # Test notebook
├── db_api_info.md           # API documentation
├── README.md                # This file
└── user_config.json         # Created after registration
```

## 🔧 Configuration

### config_log.yaml
```yaml
logging:
  enabled: true
  api_endpoint: "https://jkk4nk8j9g.execute-api.us-east-1.amazonaws.com/api"
  timeout: 20
  default_model: "EM43"
  auto_detect_mode: true
  log_initial: true
  log_checkpoints: true
  log_evaluation: false
  debug: false
```

### user_config.json (auto-generated)
```json
{
  "user_id": "abc-123-def-456-789",
  "username": "your_username",
  "created_at": "2024-01-15T10:30:00.000Z"
}
```

⚠️ **Important**: This file is automatically excluded from git (via .gitignore) to protect your sensitive user data. Never commit this file to version control.

## 🎯 Features

### ✅ Automatic Mode Detection
- **1-input mode**: Tasks like "multiply by 2"
- **2-input mode**: Tasks like "addition", "sum", etc.
- Smart detection based on task configuration and description

### ✅ Comprehensive Data Logging
- **Model**: Always "EM43"
- **Mode**: "1input" or "2input"
- **EDH**: Evolution Description Hash
- **Training Parameters**: Population size, mutation rate, etc.
- **Fitness Data**: Best fitness, average fitness, evaluation results
- **Timestamps**: ISO 8601 format
- **Run IDs**: Unique identifiers for each training session

### ✅ Robust Error Handling
- **Network timeouts**: 20-second timeout with graceful degradation
- **Connection failures**: Continue training without logging
- **Invalid configurations**: Fallback to defaults
- **User verification**: Mandatory registration before logging

### ✅ Flexible Integration
- **Standalone usage**: Direct API calls
- **EM43 wrapper integration**: Seamless with existing training scripts
- **Batch logging**: Multiple entries in single session
- **Checkpoint logging**: Automatic at save intervals

## 🧪 Testing

### Test Notebook
The `test_api_registration.ipynb` notebook provides comprehensive testing:

1. **Configuration Check** - Verify all files are in place
2. **API Health Check** - Test API connectivity
3. **User Registration** - Register new users or verify existing
4. **Logger Initialization** - Test logger setup
5. **Mode Detection** - Test automatic mode detection
6. **Manual Logging** - Test direct logging functions
7. **EM43 Integration** - Test with actual EM43 wrapper
8. **Data Verification** - Verify data was logged correctly

**📝 Note**: The test notebook works with or without user registration. If you haven't registered yet, most logging will be skipped (which is correct behavior). To test full functionality, run `python register.py` first.

### Command Line Testing
```bash
# Test registration
python register.py

# Test logger
python log.py

# Test API connection
python -c "from logger.log import test_connection; test_connection()"
```

## 📊 Data Structure

Each logged entry contains:

```json
{
  "user_id": "abc-123-def-456-789",
  "username": "researcher_name",
  "checkpoint_timestamp": "2024-01-15T10:30:00.000Z",
  "data": {
    "model": "EM43",
    "mode": "1input",
    "EDH": "s2,ep1,dp1,hf0.5ba",
    "task_id": 1,
    "task_description": "multiply by 2",
    "run_id": "run_20240115_103000_abc123",
    "generation": 150,
    "best_fitness": 0.8542,
    "avg_fitness": 0.6234,
    "checkpoint_type": "training",
    "population_size": 200,
    "mutation_rate": 0.02,
    "training_params": {
      "crossover_rate": 0.8,
      "selection_method": "tournament"
    }
  }
}
```

## 🔍 API Endpoints

- **Health Check**: `GET /health`
- **User Registration**: `POST /register_user`
- **User Verification**: `GET /user/{user_id}/verify`
- **Log Data**: `POST /log`
- **Get Users**: `GET /users`
- **Get Data**: `GET /data`

## 🛠️ Usage Examples

### Basic Usage
```python
from logger.log import EM43Logger

logger = EM43Logger()

# Check status
status = logger.get_status()
print(f"Logger ready: {status['enabled']}")

# Log data
logger.log_training_data(
    generation=100,
    best_fitness=0.85,
    avg_fitness=0.67,
    task_config={'task_id': 1, 'description': 'multiply by 2'}
)
```

### Advanced Usage
```python
from logger.log import get_logger

logger = get_logger()

# Test connection
if logger.test_connection():
    print("API connected successfully")

# Log with detailed parameters
logger.log_checkpoint(
    generation=50,
    best_fitness=0.75,
    avg_fitness=0.55,
    task_config=task_config,
    population_size=200,
    training_params={
        'mutation_rate': 0.02,
        'crossover_rate': 0.8,
        'selection_method': 'tournament',
        'diversity_penalty': 0.1
    },
    additional_data={
        'notes': 'Checkpoint after parameter adjustment',
        'experiment_phase': 'optimization'
    }
)
```

## 🚨 Troubleshooting

### Common Issues

1. **"User not registered"**
   ```bash
   python register.py
   ```

2. **"Config file not found"**
   - Ensure `config_log.yaml` exists in the logger directory
   - Check file permissions

3. **"API connection failed"**
   - Check internet connection
   - Verify API endpoint in config
   - Try increasing timeout value

4. **"Import errors"**
   - Ensure you're in the correct directory
   - Check Python path includes parent directory

5. **"User config missing on new machine"**
   - This is normal - `user_config.json` is not tracked by git
   - Simply run `python register.py` to set up user on new machine

6. **"Accidentally committed user_config.json"**
   - Remove from git: `git rm --cached em43_python_refactored/logger/user_config.json`
   - Verify .gitignore includes the file
   - Consider regenerating user credentials for security

### Debug Mode
Enable debug mode in `config_log.yaml`:
```yaml
logging:
  debug: true
```

## 📈 Performance

- **Lightweight**: Minimal overhead on training
- **Asynchronous**: Non-blocking API calls
- **Efficient**: Batch logging capabilities
- **Resilient**: Graceful failure handling

## 🔒 Security

- **No API keys required**: Frictionless research access
- **User verification**: Mandatory registration
- **Data validation**: Input sanitization
- **Secure transmission**: HTTPS only
- **Git exclusion**: `user_config.json` is automatically excluded from version control to protect sensitive user data

## 🎉 Production Ready

The logger system is production-ready and includes:

- ✅ **Comprehensive error handling**
- ✅ **Automatic mode detection**
- ✅ **Flexible configuration**
- ✅ **Robust testing suite**
- ✅ **Complete documentation**
- ✅ **Real-world integration examples**

**Perfect for distributed research with multiple collaborators! 🧬🚀** 

## 📦 Payload Example (Training Checkpoint)
```json
{
  "checkpoint_type": "training",
  "generation": 50,
  "best_fitness": -3.5666666031,
  "model": "EM43",
  "mode": "1input",
  "task_id": 2,
  "task_description": "multiply by 3",
  "population_size": 3000,
  "rule": "0000130222332323113111321312132303002121002021120332313313303123",
  "program": "10001000001000010110",
  "run_id": "run_20250711_170327_dbcfecb2",
  "EDH": "unknown",
  "timestamp": "2025-07-11T17:03:56.672541"
}
```

### Key Points
1. **Real-time logging** – checkpoints are pushed to the API immediately every *N* generations (`check_every` in `config.yaml`).
2. **Compact genome format** – `rule` (64 digits) and `program` (variable length) are stored as plain digit strings for easy parsing.
3. **Evaluation fitness** – `eval_fitness` is logged as **negative mean error** so lower is better (consistent with training fitness).
4. **Minimal payload** – only essential fields; no aver­age fitness or training hyper-parameters.

See `demo_logger.py` for a reference implementation that produces the structure above. 