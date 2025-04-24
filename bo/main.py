"""Main entry point for the Agentor BackOffice application."""

import os
import sys
import uvicorn
import subprocess
import threading
import time
import webbrowser

def run_backend():
    """Run the backend server."""
    # Add the current directory to the Python path
    sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    
    print("Starting backend server...")
    # Run the backend server
    uvicorn.run("bo.backend.main:app", host="127.0.0.1", port=8000, reload=True)

def run_frontend():
    """Run the frontend server."""
    print("Starting frontend server...")
    # Change to the frontend directory
    frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
    
    # Run the frontend server
    process = subprocess.Popen(
        ["npm", "start"],
        cwd=frontend_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        shell=True
    )
    
    # Print the output
    for line in process.stdout:
        print(line, end="")
    
    process.wait()

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
