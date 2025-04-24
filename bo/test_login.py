"""Test script for the login endpoint."""

import requests
import json

def test_login():
    """Test the login endpoint."""
    print("Testing login endpoint...")
    
    # Create form data
    data = {
        "username": "admin",
        "password": "Admin123"
    }
    
    # Make the request to the backend API
    try:
        response = requests.post(
            "http://localhost:8000/auth/token",
            data=data,
            headers={
                "Content-Type": "application/x-www-form-urlencoded"
            }
        )
        
        # Print response status code
        print(f"Response status code: {response.status_code}")
        
        # Print response headers
        print("Response headers:")
        for key, value in response.headers.items():
            print(f"  {key}: {value}")
        
        # Print response body
        try:
            response_json = response.json()
            print("Response body:")
            print(json.dumps(response_json, indent=2))
            
            if "access_token" in response_json:
                print("Login successful!")
                return True
            else:
                print("Login failed: No access token in response")
                return False
        except json.JSONDecodeError:
            print("Response body (not JSON):")
            print(response.text)
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    test_login()
