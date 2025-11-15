#!/usr/bin/env python
"""
Application launcher using configured settings.

Usage:
    python run.py                 # Uses settings from configs + .env
    python run.py --port 9000     # Override port via CLI
    python run.py --help          # Show all options
"""

import argparse
import logging
import sys
from pathlib import Path

import uvicorn


PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import get_settings

# Get configuration
settings = get_settings()

# Configure logging based on settings
settings.setup_logging()
logger = logging.getLogger(__name__)


def main():
    """Launch the application using configured settings."""
    # Ensure project root is on sys.path so "src" imports work when running from scripts/
    PROJECT_ROOT = Path(__file__).parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    
    # Parse CLI arguments (allows overrides)
    parser = argparse.ArgumentParser(
        description="Trump Speeches NLP Chatbot API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configuration precedence (highest to lowest):
  1. Command-line arguments
  2. Environment variables
  3. .env file
  4. YAML config file (configs/<environment>.yaml)
  5. Python defaults

Examples:
  python run.py                          # Use all settings from config
  python run.py --port 9000              # Override port
  python run.py --host 0.0.0.0 --port 9000 --reload
  ENVIRONMENT=production python run.py   # Use production config
        """,
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host to bind to (default from settings)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind to (default from settings)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code changes (overrides settings)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: 1)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["critical", "error", "warning", "info", "debug"],
        default=None,
        help="Log level (default: info)",
    )
    
    args = parser.parse_args()
    
    # Import here to avoid loading all the ML models upfront
    # We only need the settings module for config reading
    from src.config.settings import get_settings
    
    # Load settings from config hierarchy
    settings = get_settings()
    
    # Determine final launch parameters (CLI > settings > defaults)
    host = args.host or settings.api.host
    port = args.port or settings.api.port
    reload = args.reload or settings.api.reload
    workers = args.workers or 1
    log_level = args.log_level or "info"
    
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)
    logger.info("Environment: %s", settings.environment)
    logger.info("Host: %s", host)
    logger.info("Port: %s", port)
    logger.info("Reload: %s", reload)
    logger.info("Workers: %s", workers)
    logger.info("Log Level: %s", log_level)
    
    # Launch uvicorn with settings
    uvicorn.run(
        "src.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_level=log_level,
    )

    logger.info("Application stopped.")


if __name__ == "__main__":
    main()
