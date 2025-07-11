#!/usr/bin/env python3
"""
EM43 User Registration System
Handles first-time user registration with the distributed logging API
"""

import json
import uuid
import requests
import sys
import os
from datetime import datetime
from pathlib import Path
import yaml

class UserRegistration:
    def __init__(self, config_path="config_log.yaml"):
        """Initialize registration system"""
        self.config_path = Path(__file__).parent / config_path
        self.user_config_path = Path(__file__).parent / "user_config.json"
        self.config = self.load_config()
        
    def load_config(self):
        """Load logger configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"❌ Error: Config file {self.config_path} not found!")
            print("Please ensure config_log.yaml exists in the logger directory.")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"❌ Error parsing config file: {e}")
            sys.exit(1)
    
    def check_existing_user(self):
        """Check if user is already registered"""
        if self.user_config_path.exists():
            try:
                with open(self.user_config_path, 'r') as f:
                    user_data = json.load(f)
                    return user_data
            except (json.JSONDecodeError, KeyError):
                print("⚠️  Invalid user_config.json found. Will create new registration.")
                return None
        return None
    
    def get_username(self):
        """Get username from user input"""
        print("\n🧬 EM43 Training Logger - User Registration")
        print("=" * 50)
        
        while True:
            username = input("Enter your username (researcher name): ").strip()
            if len(username) == 0:
                print("❌ Username cannot be empty. Please try again.")
                continue
            if len(username) < 2:
                print("❌ Username must be at least 2 characters. Please try again.")
                continue
            if len(username) > 50:
                print("❌ Username must be less than 50 characters. Please try again.")
                continue
            
            # Confirm username
            confirm = input(f"Confirm username '{username}'? (y/n): ").strip().lower()
            if confirm in ['y', 'yes']:
                return username
            elif confirm in ['n', 'no']:
                continue
            else:
                print("Please enter 'y' or 'n'")
    
    def register_user(self, username):
        """Register user with the API"""
        api_endpoint = self.config['logging']['api_endpoint']
        timeout = self.config['logging']['timeout']
        
        print(f"\n🔄 Registering user '{username}' with API...")
        print(f"API: {api_endpoint}")
        
        try:
            # Prepare registration data
            registration_data = {
                "username": username
            }
            
            # Make API request
            response = requests.post(
                f"{api_endpoint}/register_user",
                json=registration_data,
                timeout=timeout,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                user_data = response.json()
                print("✅ Registration successful!")
                print(f"   User ID: {user_data['user_id']}")
                print(f"   Username: {user_data['username']}")
                print(f"   Created: {user_data['created_at']}")
                return user_data
            else:
                print(f"❌ Registration failed! Status: {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"❌ Request timed out after {timeout} seconds")
            print("Please check your internet connection and try again.")
            return None
        except requests.exceptions.ConnectionError:
            print("❌ Connection error - cannot reach API")
            print("Please check your internet connection and API endpoint.")
            return None
        except requests.exceptions.RequestException as e:
            print(f"❌ Request error: {e}")
            return None
    
    def verify_user(self, user_id):
        """Verify user registration with API"""
        api_endpoint = self.config['logging']['api_endpoint']
        timeout = self.config['logging']['timeout']
        
        print(f"\n🔍 Verifying user registration...")
        
        try:
            response = requests.get(
                f"{api_endpoint}/user/{user_id}/verify",
                timeout=timeout
            )
            
            if response.status_code == 200:
                verification_data = response.json()
                print("✅ User verification successful!")
                print(f"   Verified username: {verification_data['username']}")
                return True
            else:
                print(f"❌ User verification failed! Status: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Verification error: {e}")
            return False
    
    def save_user_config(self, user_data):
        """Save user configuration to local file"""
        try:
            with open(self.user_config_path, 'w') as f:
                json.dump(user_data, f, indent=2)
            print(f"✅ User configuration saved to {self.user_config_path}")
            return True
        except Exception as e:
            print(f"❌ Error saving user config: {e}")
            return False
    
    def run_registration(self):
        """Run complete registration process"""
        print("🚀 Starting EM43 User Registration Process")
        
        # Check if user already exists
        existing_user = self.check_existing_user()
        if existing_user:
            print(f"\n✅ User already registered!")
            print(f"   Username: {existing_user['username']}")
            print(f"   User ID: {existing_user['user_id']}")
            
            # Verify with API
            if self.verify_user(existing_user['user_id']):
                print("🎉 Registration verification complete! Ready to log training data.")
                return True
            else:
                print("⚠️  Verification failed. Please re-register.")
                # Continue with new registration
        
        # Get username
        username = self.get_username()
        
        # Register with API
        user_data = self.register_user(username)
        if not user_data:
            print("\n❌ Registration failed. Exiting.")
            return False
        
        # Verify registration
        if not self.verify_user(user_data['user_id']):
            print("\n❌ User verification failed. Exiting.")
            return False
        
        # Save user config
        if not self.save_user_config(user_data):
            print("\n❌ Failed to save user configuration. Exiting.")
            return False
        
        print("\n🎉 Registration complete! You can now log training data.")
        print(f"User config saved to: {self.user_config_path}")
        return True

def main():
    """Main registration function"""
    try:
        registrar = UserRegistration()
        success = registrar.run_registration()
        
        if success:
            print("\n📊 Next steps:")
            print("1. Run your training scripts - they will automatically log data")
            print("2. Use the test notebook to verify logging functionality")
            print("3. Check the API /data endpoint to view your logged data")
            sys.exit(0)
        else:
            print("\n❌ Registration failed. Please check the error messages above.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Registration cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 