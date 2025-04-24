#!/usr/bin/env python

import os
import sys
import subprocess
import threading
import time
import webbrowser

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))

def run_backend():
    """Run the backend server."""
    # Set the Python path to include the current directory
    env = os.environ.copy()
    env["PYTHONPATH"] = script_dir
    
    # Change to the backend directory
    backend_dir = os.path.join(script_dir, "bo", "backend")
    
    print("Starting backend server...")
    # Run the backend server
    subprocess.run(
        ["uvicorn", "main:app", "--reload", "--port", "8000"],
        cwd=backend_dir,
        env=env
    )

def run_frontend():
    """Run the frontend server."""
    # Change to the frontend directory
    frontend_dir = os.path.join(script_dir, "bo", "frontend")
    
    print("Starting frontend server...")
    # Run the frontend server
    subprocess.run(
        ["npm", "start"],
        cwd=frontend_dir
    )

def open_browser():
    """Open the browser to the frontend."""
    time.sleep(5)  # Wait for the servers to start
    print("Opening http://localhost:3000 in browser...")
    webbrowser.open("http://localhost:3000")

def main():
    """Run the backend and frontend servers."""
    # Start the backend server in a separate thread
    backend_thread = threading.Thread(target=run_backend)
    backend_thread.daemon = True
    backend_thread.start()
    
    # Wait for the backend server to start
    time.sleep(2)
    
    # Start the frontend server in a separate thread
    frontend_thread = threading.Thread(target=run_frontend)
    frontend_thread.daemon = True
    frontend_thread.start()
    
    # Open the browser
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    print("Press Ctrl+C to stop the servers.")
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping servers...")

if __name__ == "__main__":
    main()
