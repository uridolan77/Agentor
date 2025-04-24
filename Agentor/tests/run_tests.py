#!/usr/bin/env python
"""
Test runner script for the Agentor framework.

This script provides a convenient way to run tests with different configurations.
"""

import argparse
import subprocess
import sys
import os
from typing import List, Optional


def run_tests(
    test_path: Optional[str] = None,
    markers: Optional[List[str]] = None,
    coverage: bool = False,
    html_report: bool = False,
    verbose: bool = False,
    no_skip: bool = False,
    junit_xml: bool = False
) -> int:
    """Run the tests with the specified configuration.
    
    Args:
        test_path: Path to the test file or directory
        markers: List of markers to select
        coverage: Whether to generate a coverage report
        html_report: Whether to generate an HTML coverage report
        verbose: Whether to use verbose output
        no_skip: Whether to run tests marked as skipped
        junit_xml: Whether to generate a JUnit XML report
        
    Returns:
        The exit code from pytest
    """
    # Build the command
    cmd = ["pytest"]
    
    # Add the test path if specified
    if test_path:
        cmd.append(test_path)
    
    # Add markers if specified
    if markers:
        for marker in markers:
            cmd.append(f"-m")
            cmd.append(marker)
    
    # Add coverage options if specified
    if coverage:
        cmd.append("--cov=agentor")
        if html_report:
            cmd.append("--cov-report=html")
    
    # Add verbose flag if specified
    if verbose:
        cmd.append("-v")
    
    # Add no-skip flag if specified
    if no_skip:
        cmd.append("--no-skip")
    
    # Add JUnit XML report if specified
    if junit_xml:
        cmd.append("--junitxml=test-results.xml")
    
    # Print the command
    print(f"Running: {' '.join(cmd)}")
    
    # Run the command
    result = subprocess.run(cmd)
    
    return result.returncode


def main():
    """Parse arguments and run tests."""
    parser = argparse.ArgumentParser(description="Run Agentor tests")
    
    parser.add_argument(
        "test_path",
        nargs="?",
        help="Path to the test file or directory"
    )
    
    parser.add_argument(
        "-m", "--marker",
        action="append",
        dest="markers",
        help="Run tests with the specified marker (can be used multiple times)"
    )
    
    parser.add_argument(
        "-c", "--coverage",
        action="store_true",
        help="Generate a coverage report"
    )
    
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate an HTML coverage report"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Use verbose output"
    )
    
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Run tests marked as skipped"
    )
    
    parser.add_argument(
        "--junit-xml",
        action="store_true",
        help="Generate a JUnit XML report"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run tests
    exit_code = run_tests(
        test_path=args.test_path,
        markers=args.markers,
        coverage=args.coverage,
        html_report=args.html,
        verbose=args.verbose,
        no_skip=args.no_skip,
        junit_xml=args.junit_xml
    )
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
