#!/usr/bin/env python
"""
Script to start the monitoring components for the LLM Gateway Agent System.

This script initializes the health monitoring and metrics collection.
"""

import asyncio
import argparse
import logging
import sys
import os

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm_gateway.utils.health_metrics import start_health_monitoring, stop_health_monitoring
from llm_gateway.utils.llm_metrics import initialize_cost_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def main():
    """Start the monitoring components."""
    parser = argparse.ArgumentParser(description='Start monitoring for LLM Gateway Agent System')
    parser.add_argument('--health-check-interval', type=int, default=60,
                        help='Interval in seconds for health checks (default: 60)')
    args = parser.parse_args()
    
    try:
        # Initialize metrics
        logger.info("Initializing metrics...")
        initialize_cost_metrics()
        
        # Start health monitoring
        logger.info(f"Starting health monitoring with interval {args.health_check_interval}s...")
        await start_health_monitoring()
        
        logger.info("Monitoring started. Press Ctrl+C to stop.")
        
        # Keep the script running
        while True:
            await asyncio.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Stopping monitoring...")
        await stop_health_monitoring()
        logger.info("Monitoring stopped.")
    
    except Exception as e:
        logger.error(f"Error in monitoring: {str(e)}")
        await stop_health_monitoring()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
