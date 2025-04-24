#!/usr/bin/env python
"""
Setup script for Agentor BackOffice.
This script creates the virtual environment and installs dependencies.
"""

import os
import sys
import subprocess
import platform

# Configuration
BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")

def create_venv():
    """Create the virtual environment for the backend."""
    print("Creating virtual environment...")
    
    venv_path = os.path.join(BACKEND_DIR, "venv")
    if os.path.exists(venv_path):
        print("Virtual environment already exists.")
        return True
    
    try:
        subprocess.check_call([sys.executable, "-m", "venv", venv_path])
        print("Virtual environment created successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating virtual environment: {e}")
        return False

def install_backend_deps():
    """Install backend dependencies."""
    print("Installing backend dependencies...")
    
    # Command to activate virtual environment
    if platform.system() == "Windows":
        activate_cmd = os.path.join(BACKEND_DIR, "venv", "Scripts", "activate")
        pip_cmd = os.path.join(BACKEND_DIR, "venv", "Scripts", "pip")
    else:
        activate_cmd = os.path.join(BACKEND_DIR, "venv", "bin", "activate")
        pip_cmd = os.path.join(BACKEND_DIR, "venv", "bin", "pip")
    
    requirements_path = os.path.join(BACKEND_DIR, "requirements.txt")
    
    try:
        if platform.system() == "Windows":
            # On Windows, we need to use a different approach
            subprocess.check_call(
                f'"{pip_cmd}" install -r "{requirements_path}"',
                shell=True
            )
        else:
            # On Unix, we can source the activate script
            subprocess.check_call(
                f'source "{activate_cmd}" && pip install -r "{requirements_path}"',
                shell=True,
                executable="/bin/bash"
            )
        
        print("Backend dependencies installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing backend dependencies: {e}")
        return False

def install_frontend_deps():
    """Install frontend dependencies."""
    print("Installing frontend dependencies...")
    
    node_modules_path = os.path.join(FRONTEND_DIR, "node_modules")
    if os.path.exists(node_modules_path):
        print("Frontend dependencies already installed.")
        return True
    
    try:
        # Check if npm is available
        try:
            subprocess.check_call(["npm", "--version"], stdout=subprocess.DEVNULL)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Error: npm not found. Please install Node.js and npm.")
            return False
        
        # Install dependencies with legacy-peer-deps flag to avoid dependency conflicts
        subprocess.check_call(
            "npm install --legacy-peer-deps",
            shell=True,
            cwd=FRONTEND_DIR
        )
        
        print("Frontend dependencies installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing frontend dependencies: {e}")
        return False

def main():
    """Main function."""
    print("Setting up Agentor BackOffice...")
    
    # Create virtual environment
    if not create_venv():
        print("Failed to create virtual environment. Exiting.")
        return
    
    # Install backend dependencies
    if not install_backend_deps():
        print("Failed to install backend dependencies. Exiting.")
        return
    
    # Install frontend dependencies
    if not install_frontend_deps():
        print("Failed to install frontend dependencies. Exiting.")
        return
    
    print("\nSetup completed successfully!")
    print("\nYou can now run the development servers:")
    print("python bo/run_dev.py")

if __name__ == "__main__":
    main()
