"""
Production-ready logging configuration for Azure/Docker deployment.

Supports both JSON logging (for production/cloud) and human-readable format
(for local development). Automatically detects environment and configures
appropriate logging format.
"""

import json
import logging
import sys
from typing import Any, Dict


class JsonFormatter(logging.Formatter):
    """
    JSON formatter for structured logging in production environments.

    Outputs logs as JSON with timestamp, level, logger name, and message.
    Compatible with Azure Application Insights, CloudWatch, and other
    cloud logging systems.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string."""
        log_record: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_record.update(getattr(record, "extra_fields"))

        return json.dumps(log_record)


class ColoredFormatter(logging.Formatter):
    """
    Colored formatter for human-readable logs in local development.

    Adds color coding based on log level for better readability in terminals.
    """

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with color coding."""
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname:8}{self.RESET}"

        # Format the message
        formatted = super().format(record)

        # Reset levelname for next use
        record.levelname = levelname

        return formatted


def configure_logging(
    level: str = "INFO",
    use_json: bool = False,
    include_uvicorn: bool = True,
) -> None:
    """
    Configure application-wide logging.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        use_json: Use JSON formatting (True for production, False for development)
        include_uvicorn: Configure uvicorn loggers as well
    """
    # Determine format
    formatter: logging.Formatter
    if use_json:
        formatter = JsonFormatter(datefmt="%Y-%m-%d %H:%M:%S")
    else:
        # Human-readable format with timestamps and color
        fmt = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        formatter = ColoredFormatter(fmt=fmt, datefmt=datefmt)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Configure uvicorn loggers if requested
    if include_uvicorn:
        # Uvicorn access logs
        uvicorn_access = logging.getLogger("uvicorn.access")
        uvicorn_access.handlers.clear()
        uvicorn_access.addHandler(console_handler)
        uvicorn_access.propagate = False

        # Uvicorn error logs
        uvicorn_error = logging.getLogger("uvicorn.error")
        uvicorn_error.handlers.clear()
        uvicorn_error.addHandler(console_handler)
        uvicorn_error.propagate = False

        # Keep uvicorn at INFO level regardless of global level
        uvicorn_access.setLevel(logging.INFO)
        uvicorn_error.setLevel(logging.INFO)

    # Suppress noisy third-party loggers
    logging.getLogger("chromadb").setLevel(logging.ERROR)
    logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)  # Suppress telemetry errors
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Add filter to suppress ChromaDB telemetry errors
    class ChromaDBTelemetryFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "Failed to send telemetry event" not in record.getMessage()

    chromadb_logger = logging.getLogger("chromadb")
    chromadb_logger.addFilter(ChromaDBTelemetryFilter())

    # Log configuration
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured: level={level}, format={'JSON' if use_json else 'colored'}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.

    This is a convenience function that ensures consistent logger retrieval.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
