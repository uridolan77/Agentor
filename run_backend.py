#!/usr/bin/env python

import os
import sys

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change to the backend directory
backend_dir = os.path.join(script_dir, "bo", "backend")
os.chdir(backend_dir)

# Add the backend directory to the Python path
sys.path.insert(0, backend_dir)

# Import and run the app
import uvicorn

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=9000, reload=True)
