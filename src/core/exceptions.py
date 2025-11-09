"""
Custom exception classes for the NLP Chatbot API.

Provides domain-specific exceptions for better error handling and
more informative error messages.
"""

from typing import Any, Dict, Optional


class APIException(Exception):
    """Base exception for all API-related errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize API exception.

        Args:
            message: Error message
            status_code: HTTP status code
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class ConfigurationError(APIException):
    """Raised when there's a configuration problem."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=500, details=details)


class ModelLoadError(APIException):
    """Raised when a machine learning model fails to load."""

    def __init__(self, model_name: str, details: Optional[Dict[str, Any]] = None):
        message = f"Failed to load model: {model_name}"
        super().__init__(message, status_code=503, details=details)
        self.model_name = model_name


class LLMServiceError(APIException):
    """Raised when LLM service encounters an error."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=503, details=details)


class RAGServiceError(APIException):
    """Raised when RAG service encounters an error."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=503, details=details)


class ValidationError(APIException):
    """Raised when input validation fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=422, details=details)


class NotFoundError(APIException):
    """Raised when a requested resource is not found."""

    def __init__(self, resource: str, details: Optional[Dict[str, Any]] = None):
        message = f"Resource not found: {resource}"
        super().__init__(message, status_code=404, details=details)
        self.resource = resource
