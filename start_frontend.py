#!/usr/bin/env python

import os
import sys
import subprocess

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change to the frontend directory
frontend_dir = os.path.join(script_dir, "bo", "frontend")
os.chdir(frontend_dir)

# Run the frontend server
subprocess.run(["npm", "start"])
