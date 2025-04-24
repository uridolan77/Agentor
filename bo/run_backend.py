#!/usr/bin/env python

import os
import sys
import subprocess
import platform

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
backend_dir = os.path.join(current_dir, "backend")

# Add the parent directory to Python path to make 'bo' module accessible
sys.path.insert(0, parent_dir)

# Configure backend
port = 9000

# Command to activate virtual environment
if platform.system() == "Windows":
    activate_venv = os.path.join(backend_dir, "venv", "Scripts", "activate")
    activate_cmd = f"call {activate_venv}"
else:
    activate_venv = os.path.join(backend_dir, "venv", "bin", "activate")
    activate_cmd = f"source {activate_venv}"

# Command to run backend server
backend_cmd = f"cd {backend_dir} && {activate_cmd} && set PYTHONPATH={parent_dir} && uvicorn main:app --reload --port {port}"

def main():
    """Run the backend server only."""
    print(f"Starting backend server on port {port}...")
    
    try:
        if platform.system() == "Windows":
            process = subprocess.Popen(backend_cmd, shell=True)
        else:
            process = subprocess.Popen(backend_cmd, shell=True, executable="/bin/bash")
        
        print(f"Backend server running at http://localhost:{port}")
        print("Press CTRL+C to stop the server.")
        
        # Keep the script running
        process.wait()
    
    except KeyboardInterrupt:
        print("\nShutting down backend server...")
        if platform.system() == "Windows":
            subprocess.call(['taskkill', '/F', '/T', '/PID', str(process.pid)])
        else:
            process.terminate()
        print("Backend server shut down.")
    
    except Exception as e:
        print(f"Error running backend server: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())