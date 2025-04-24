#!/usr/bin/env python

import os
import sys
import subprocess
import time
import webbrowser
import signal
import platform

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration
BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
BACKEND_PORT = 9000
FRONTEND_PORT = 3000

# Command to activate virtual environment
if platform.system() == "Windows":
    ACTIVATE_VENV = os.path.join(BACKEND_DIR, "venv", "Scripts", "activate")
    ACTIVATE_CMD = f"call {ACTIVATE_VENV}"
else:
    ACTIVATE_VENV = os.path.join(BACKEND_DIR, "venv", "bin", "activate")
    ACTIVATE_CMD = f"source {ACTIVATE_VENV}"

# Commands to run servers
# Use PYTHONPATH to ensure the parent directory is in the Python path
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_CMD = f"cd {BACKEND_DIR} && {ACTIVATE_CMD} && set PYTHONPATH={PARENT_DIR} && uvicorn main:app --reload --port {BACKEND_PORT}"
FRONTEND_CMD = f"cd {FRONTEND_DIR} && npm start"

# Global variables to store process objects
backend_process = None
frontend_process = None

def start_backend():
    """Start the backend server."""
    global backend_process
    print("Starting backend server...")

    if platform.system() == "Windows":
        backend_process = subprocess.Popen(BACKEND_CMD, shell=True)
    else:
        backend_process = subprocess.Popen(BACKEND_CMD, shell=True, executable="/bin/bash")

    # Wait for backend to start
    time.sleep(2)
    print(f"Backend server running at http://localhost:{BACKEND_PORT}")

def start_frontend():
    """Start the frontend server."""
    global frontend_process
    print("Starting frontend server...")

    if platform.system() == "Windows":
        frontend_process = subprocess.Popen(FRONTEND_CMD, shell=True)
    else:
        frontend_process = subprocess.Popen(FRONTEND_CMD, shell=True, executable="/bin/bash")

    # Wait for frontend to start
    time.sleep(5)
    print(f"Frontend server running at http://localhost:{FRONTEND_PORT}")

def open_browser():
    """Open the browser to the frontend URL."""
    url = f"http://localhost:{FRONTEND_PORT}"
    print(f"Opening {url} in browser...")
    webbrowser.open(url)

def cleanup(signum=None, frame=None):
    """Clean up processes on exit."""
    print("\nShutting down servers...")

    if backend_process:
        if platform.system() == "Windows":
            subprocess.call(['taskkill', '/F', '/T', '/PID', str(backend_process.pid)])
        else:
            backend_process.terminate()

    if frontend_process:
        if platform.system() == "Windows":
            subprocess.call(['taskkill', '/F', '/T', '/PID', str(frontend_process.pid)])
        else:
            frontend_process.terminate()

    print("Servers shut down.")
    sys.exit(0)

def main():
    """Main function."""
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        # Check if virtual environment exists
        venv_path = os.path.join(BACKEND_DIR, "venv")
        if not os.path.exists(venv_path):
            print("Virtual environment not found. Creating it now...")
            try:
                subprocess.check_call(
                    f"python -m venv venv",
                    shell=True,
                    cwd=BACKEND_DIR
                )
                print("Virtual environment created successfully.")

                # Install dependencies
                print("Installing backend dependencies...")
                if platform.system() == "Windows":
                    subprocess.check_call(
                        f"{ACTIVATE_CMD} && pip install -r requirements.txt && pip install email-validator",
                        shell=True,
                        cwd=BACKEND_DIR
                    )
                else:
                    subprocess.check_call(
                        f"{ACTIVATE_CMD} && pip install -r requirements.txt && pip install email-validator",
                        shell=True,
                        executable="/bin/bash",
                        cwd=BACKEND_DIR
                    )
                print("Backend dependencies installed successfully.")
            except subprocess.CalledProcessError as e:
                print(f"Error setting up virtual environment: {e}")
                print("Please set it up manually:")
                print(f"cd {BACKEND_DIR} && python -m venv venv")
                print(f"Then install dependencies: cd {BACKEND_DIR} && {ACTIVATE_CMD} && pip install -r requirements.txt && pip install email-validator")
                return
        else:
            # Install email-validator if it's not already installed
            print("Installing email-validator...")
            if platform.system() == "Windows":
                subprocess.check_call(
                    f"{ACTIVATE_CMD} && pip install email-validator",
                    shell=True,
                    cwd=BACKEND_DIR
                )
            else:
                subprocess.check_call(
                    f"{ACTIVATE_CMD} && pip install email-validator",
                    shell=True,
                    executable="/bin/bash",
                    cwd=BACKEND_DIR
                )
            print("email-validator installed successfully.")

        # Check if node_modules exists
        node_modules_path = os.path.join(FRONTEND_DIR, "node_modules")
        if not os.path.exists(node_modules_path):
            print("Node modules not found. Installing them now...")
            try:
                subprocess.check_call(
                    "npm install --legacy-peer-deps",
                    shell=True,
                    cwd=FRONTEND_DIR
                )
                print("Frontend dependencies installed successfully.")
            except subprocess.CalledProcessError as e:
                print(f"Error installing frontend dependencies: {e}")
                print("Please install them manually:")
                print(f"cd {FRONTEND_DIR} && npm install --legacy-peer-deps")
                return

        # Start servers
        start_backend()
        start_frontend()

        # Open browser
        open_browser()

        # Keep the script running
        print("Press Ctrl+C to stop the servers.")
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        cleanup()
    except Exception as e:
        print(f"Error: {e}")
        cleanup()

if __name__ == "__main__":
    main()
