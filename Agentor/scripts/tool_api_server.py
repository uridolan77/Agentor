#!/usr/bin/env python
"""
Command-line entry point for the Tool API Server.

This script provides a command-line entry point for starting the Tool API Server
and generating OpenAPI documentation.
"""

import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agentor.agents.api.cli import main

if __name__ == "__main__":
    main()
