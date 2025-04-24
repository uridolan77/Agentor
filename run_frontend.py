#!/usr/bin/env python

import os
import sys
import subprocess
import platform

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change to the frontend directory
frontend_dir = os.path.join(script_dir, "bo", "frontend")
os.chdir(frontend_dir)

# Run the frontend server
if __name__ == "__main__":
    # Use different commands based on the operating system
    if platform.system() == "Windows":
        # On Windows, use the npm.cmd executable
        subprocess.run(["cmd", "/c", "npm", "start"], shell=True)
    else:
        # On Unix-like systems, use npm directly
        subprocess.run(["npm", "start"], shell=True)
