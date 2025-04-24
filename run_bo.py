"""
Run the Agentor BackOffice application.
"""

import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the main module
from bo.backend.main import app
import uvicorn

if __name__ == "__main__":
    uvicorn.run("bo.backend.main:app", host="0.0.0.0", port=8000, reload=True)
