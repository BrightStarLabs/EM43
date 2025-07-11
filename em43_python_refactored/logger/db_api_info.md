# EM Simplified Training System

🧬 **Cellular Automata-based Emergent Model training with simplified user management and logging**

A streamlined distributed system for training EM models with clean separation of user management and training data logging.

## 🚀 **Live Deployment Status**

**API URL:** `https://jkk4nk8j9g.execute-api.us-east-1.amazonaws.com/api`

**Architecture:** Simplified v3.0
- ✅ User registration and verification
- ✅ Mandatory user verification before logging
- ✅ Clean data structure (everything in data{} object)
- ✅ Raw data dumps for client processing

## 🏗️ **Simplified Architecture**

### **2-Table Design**
```
EMUsers (User Management)
├── user_id (String, UUID) - Primary Key
├── username (String, NOT unique)
└── created_at (ISO 8601 timestamp)

EMLogs (Training Data)
├── user_id (String) - Partition Key
├── checkpoint_timestamp (ISO 8601) - Sort Key
├── username (String, denormalized)
└── data (Object with all training parameters)
```

### **Key Principles**
- **Clean Separation:** Users vs Training Data
- **User Verification:** Required before any logging
- **Simple Structure:** Everything training-related goes in data{}
- **Raw Data Access:** No server-side analytics, client processes data
- **Error Enforcement:** Unregistered users cannot log data

## 🚀 **Quick Start**

### **Prerequisites**
1. **AWS Account** with CLI configured
2. **Python 3.8+** installed

### **1. Setup AWS Credentials**
```bash
# Install AWS CLI if needed
pip install awscli

# Configure credentials
aws configure
```

### **2. Deploy the System**
```bash
# Navigate to backend directory
cd backend-aws

# Step 1: Setup DynamoDB tables (clean slate)
python setup_dynamodb.py

# Step 2: Deploy Chalice API
cd em-log-api
chalice deploy --stage dev

# Note the API endpoint URL from output!
```

### **3. Test the System**
```bash
# Go back to backend directory
cd ..

# Test the deployed system
python test_api.py https://jkk4nk8j9g.execute-api.us-east-1.amazonaws.com/api
```

## 📊 **API Endpoints (6 Total)**

**Base URL:** `https://jkk4nk8j9g.execute-api.us-east-1.amazonaws.com/api`

| Endpoint | Method | Purpose | Status |
|----------|--------|---------|--------|
| `/health` | GET | System health check | ✅ Working |
| `/register_user` | POST | Create new user account | ✅ Working |
| `/user/{user_id}/verify` | GET | Verify user exists | ✅ Working |
| `/log` | POST | Log training data (requires registration) | ✅ Working |
| `/users` | GET | Get all registered users | ✅ Working |
| `/data` | GET | Get all training data (raw format) | ✅ Working |

## 🔄 **Complete User Flow**

### **1. User Registration**
```bash
# Register new user
curl -X POST https://jkk4nk8j9g.execute-api.us-east-1.amazonaws.com/api/register_user \
  -H "Content-Type: application/json" \
  -d '{"username": "researcher_name"}'

# Response:
{
  "success": true,
  "user_id": "abc-123-def-456-789",
  "username": "researcher_name",
  "created_at": "2024-01-15T10:30:00.000Z"
}
```

### **2. User Verification**
```bash
# Verify user exists
curl https://jkk4nk8j9g.execute-api.us-east-1.amazonaws.com/api/user/abc-123-def-456-789/verify

# Response (success):
{
  "exists": true,
  "username": "researcher_name",
  "created_at": "2024-01-15T10:30:00.000Z"
}

# Response (not found): 404 error
```

### **3. Training Data Logging**
```bash
# Log training data (user must be registered)
curl -X POST https://jkk4nk8j9g.execute-api.us-east-1.amazonaws.com/api/log \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "abc-123-def-456-789",
    "username": "researcher_name",
    "data": {
      "model": "EM",
      "EDH": "s2,ep1,dp1,hf0.5ba",
      "task_id": 1,
      "task_description": "multip. by 2",
      "run_id": "run_abc123",
      "generation": 150,
      "best_fitness": 0.8542,
      "avg_fitness": 0.6234,
      "population_size": 200,
      "mutation_rate": 0.02,
      "custom_fields": "any_data_you_want"
    }
  }'

# Response:
{
  "success": true,
  "checkpoint_timestamp": "2024-01-15T10:30:00.000Z",
  "message": "Training data logged successfully",
  "user_id": "abc-123-def-456-789",
  "username": "researcher_name"
}
```

### **4. Data Retrieval**
```bash
# Get all users
curl https://jkk4nk8j9g.execute-api.us-east-1.amazonaws.com/api/users

# Get all training data (raw format)
curl https://jkk4nk8j9g.execute-api.us-east-1.amazonaws.com/api/data
```

## 🔐 **Error Handling & Validation**

### **Unregistered User Protection**
```bash
# Attempt to log with unregistered user_id
curl -X POST https://jkk4nk8j9g.execute-api.us-east-1.amazonaws.com/api/log \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "fake-user-id",
    "username": "fake_user",
    "data": {"test": "data"}
  }'

# Response: 400 Bad Request
{
  "Message": "User fake-user-id is not registered. Please register first using /register_user endpoint."
}
```

### **Validation Rules**
- **Registration:** `username` required, non-empty string
- **Logging:** `user_id`, `username`, `data{}` required
- **User Verification:** Checks user exists in EMUsers table
- **Username Consistency:** Logged username must match registered username
- **Data Structure:** `data` must be an object/dictionary, not string

## 🗄️ **Database Schema Details**

### **EMUsers Table**
```json
{
  "user_id": "abc-123-def-456-789",
  "username": "researcher_name",
  "created_at": "2024-01-15T10:30:00.000Z"
}
```

**Primary Key:** `user_id` (String, UUID)
**Notes:** 
- `user_id` is unique per user
- `username` is NOT unique (multiple users can have same username)
- No complex queries needed

### **EMLogs Table**
```json
{
  "user_id": "abc-123-def-456-789",
  "username": "researcher_name",
  "checkpoint_timestamp": "2024-01-15T10:30:00.000Z",
  "data": {
    "model": "EM",
    "EDH": "s2,ep1,dp1,hf0.5ba",
    "task_id": 1,
    "task_description": "multip. by 2",
    "run_id": "run_abc123",
    "generation": 150,
    "best_fitness": 0.8542,
    "custom_fields": "unlimited_flexibility"
  }
}
```

**Primary Key:** `user_id` (Hash) + `checkpoint_timestamp` (Range)
**Notes:**
- Efficient per-user queries
- Chronological ordering via ISO 8601 timestamps
- Unlimited flexibility in `data` object
- Username denormalized for easy access

## 🎯 **Data Structure Benefits**

### **Before (Complex):**
```json
{
  "user_id": "...",
  "username": "...",
     "model": "EM",           // At root level
  "EDH": "s2,ep1,dp1,hf0.5ba",  // At root level
  "task_id": 1,              // At root level  
  "task_description": "...", // At root level
  "data": {
    "run_id": "...",
    "generation": 150,
    "best_fitness": 0.8542
  }
}
```

### **After (Simplified):**
```json
{
  "user_id": "...",
  "username": "...",
       "data": {
       "model": "EM",           // Inside data{}
       "EDH": "s2,ep1,dp1,hf0.5ba",  // Inside data{}
    "task_id": 1,              // Inside data{}
    "task_description": "...", // Inside data{}
    "run_id": "...",
    "generation": 150,
    "best_fitness": 0.8542,
    "any_custom_fields": "..."
  }
}
```

**Benefits:**
- ✅ Clean separation: User identity vs Training data
- ✅ Complete flexibility in data structure
- ✅ No default values or optional field complexity
- ✅ Client decides what to store and how

## 🔧 **Development**

### **Local Testing**
```bash
# Run Chalice locally
cd backend-aws/em-log-api
chalice local

# Test locally
cd ..
python test_api.py http://localhost:8000
```

### **Project Structure**
```
dockerino/
├── backend-aws/             # Backend AWS infrastructure
│   ├── em-log-api/          # Chalice API project
│   │   ├── app.py           # 6 simplified endpoints
│   │   └── requirements.txt # Dependencies
│   ├── setup_dynamodb.py   # 2-table setup script
│   ├── test_api.py          # Comprehensive testing
│   └── README.md            # This file
└── client/                  # (Future) Training client integration
```

## 🛡️ **Security & Features**

- **No API keys required** (frictionless research access)
- **Mandatory user verification** before any logging
- **Input validation** for all endpoints
- **Proper HTTP status codes** (400, 404, 500)
- **Raw data access** for flexible client-side processing
- **Pay-per-request** DynamoDB billing

## 💰 **Cost Estimation**

For **100,000 requests/year:**

| Component | Cost |
|-----------|------|
| Lambda execution | ~$0.00 (free tier) |
| DynamoDB writes | ~$0.13/year |
| DynamoDB storage | ~$0.30/year |
| **Total** | **~$0.43/year** |

## 🐛 **Troubleshooting**

### **Common Issues**

1. **"No credentials found"**
   ```bash
   aws configure
   ```

2. **"User not registered" error**
   ```bash
   # Register user first
   curl -X POST https://your-api/register_user -d '{"username": "your_name"}'
   ```

3. **"Username mismatch" error**
   - Use exact username from registration response

4. **Table setup issues**
   ```bash
   # Delete and recreate tables
   cd backend-aws
   python setup_dynamodb.py
   ```

### **Debugging**
```bash
# Check AWS credentials
aws sts get-caller-identity

# View API logs
cd backend-aws/em-log-api
chalice logs --stage dev

# Test individual endpoints
curl -v https://your-api/health
```

## 🔄 **Migration Notes**

If migrating from the old complex system:

1. **Data Migration:** Use `/data` endpoint to export existing data
2. **User Creation:** Bulk register users via `/register_user`
3. **Schema Update:** Move training parameters into `data{}` object
4. **Client Updates:** Update client code to register users first

## 🚀 **Next Steps**

### **Client Integration**
1. **User Registration Flow:** Add to client startup
2. **Offline Handling:** Graceful degradation when API unavailable
3. **Batch Logging:** Efficient training data submission
4. **Data Analysis:** Client-side processing of raw data dumps

### **System Enhancements**
1. **Rate Limiting:** Configure API Gateway throttling
2. **Monitoring:** CloudWatch dashboards
3. **Bulk Operations:** Batch user registration/logging
4. **Data Export:** Enhanced data retrieval formats

## 📝 **API Response Examples**

### **Health Check**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "schema_version": "3.0",
  "architecture": "simplified",
     "tables": {
     "users": "EMUsers",
     "logs": "EMLogs"
  }
}
```

### **Registration Response**
```json
{
  "success": true,
  "user_id": "abc-123-def-456-789",
  "username": "researcher_name",
  "created_at": "2024-01-15T10:30:00.000Z"
}
```

### **Data Dump Response**
```json
{
  "data": [
    {
      "user_id": "abc-123-def-456-789",
      "username": "researcher_name",
      "checkpoint_timestamp": "2024-01-15T10:30:00.000Z",
              "data": {
        "model": "EM",
        "EDH": "s2,ep1,dp1,hf0.5ba",
        "task_id": 1,
        "task_description": "multip. by 2",
        "run_id": "run_abc123",
        "generation": 150,
        "best_fitness": 0.8542,
        "population_size": 200
      }
    }
  ],
  "count": 1,
  "format": "raw_unstructured",
  "note": "Client should parse and process this data as needed"
}
```

---

## 🎉 **Production Ready!**

Your EM simplified training system is ready for research:

- ✅ **Clean Architecture:** 2 tables, 6 endpoints, clear separation
- ✅ **User Management:** Registration and verification enforced
- ✅ **Data Integrity:** No logging without registered users
- ✅ **Flexible Storage:** Everything training-related in data{} object
- ✅ **Raw Data Access:** Client processes data as needed
- ✅ **Error Handling:** Proper validation and meaningful error messages

**Perfect for distributed research with multiple collaborators! 🧬🚀** 