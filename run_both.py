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
    # Change to the backend directory
    backend_dir = os.path.join(script_dir, "bo", "backend")
    os.chdir(backend_dir)
    
    # Add the backend directory to the Python path
    sys.path.insert(0, backend_dir)
    
    # Run the backend server
    print("Starting backend server...")
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

def run_frontend():
    """Run the frontend server."""
    # Change to the frontend directory
    frontend_dir = os.path.join(script_dir, "bo", "frontend")
    os.chdir(frontend_dir)
    
    # Run the frontend server
    print("Starting frontend server...")
    subprocess.run(["npm", "start"], shell=True)

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
    
    # Open the browser
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Run the frontend server in the main thread
    run_frontend()

if __name__ == "__main__":
    main()
