import requests
import json

def get_auth_token(email, password, backend_url="http://localhost:8000"):
    """
    Get authentication token from the backend
    
    Args:
        email (str): User email
        password (str): User password
        backend_url (str): Backend API URL
    
    Returns:
        str: Authentication token or None if failed
    """
    try:
        # Login to get token
        login_data = {
            "username": email,  # FastAPI OAuth2 expects 'username' field
            "password": password
        }
        
        response = requests.post(
            f"{backend_url}/token",
            data=login_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10
        )
        
        if response.status_code == 200:
            token_data = response.json()
            token = token_data.get("access_token")
            print(f"Successfully obtained token: {token[:20]}...")
            return token
        else:
            print(f"Failed to get token: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Error getting token: {e}")
        return None

def register_user(email, password, full_name, backend_url="http://localhost:8000"):
    """
    Register a new user
    
    Args:
        email (str): User email
        password (str): User password
        full_name (str): User full name
        backend_url (str): Backend API URL
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        user_data = {
            "email": email,
            "password": password,
            "full_name": full_name
        }
        
        response = requests.post(
            f"{backend_url}/register",
            json=user_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            print(f"Successfully registered user: {email}")
            return True
        else:
            print(f"Failed to register user: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"Error registering user: {e}")
        return False

if __name__ == "__main__":
    # Configuration
    BACKEND_URL = "http://localhost:8000"
    EMAIL = "admin@csms.com"
    PASSWORD = "admin123"
    FULL_NAME = "Admin User"
    
    print("CSMS Authentication Helper")
    print("=" * 30)
    
    # Try to register user first (will fail if user already exists)
    print("Attempting to register user...")
    register_user(EMAIL, PASSWORD, FULL_NAME, BACKEND_URL)
    
    # Get authentication token
    print("\nAttempting to get authentication token...")
    token = get_auth_token(EMAIL, PASSWORD, BACKEND_URL)
    
    if token:
        print(f"\nToken obtained successfully!")
        print(f"Add this token to your realtime_detection.py script:")
        print(f"AUTH_TOKEN = '{token}'")
    else:
        print("\nFailed to get token. Please check your backend server and credentials.") 