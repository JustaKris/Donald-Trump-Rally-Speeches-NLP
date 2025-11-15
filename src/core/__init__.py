"""
Core application configuration and utilities.

This module contains the core infrastructure components including
configuration management, logging, security, custom exceptions, and constants.
"""

from .constants import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMOTION_MODEL,
    DEFAULT_EXCLUDED_VERBS,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_SENTIMENT_MODEL,
    ENTITY_QUESTION_WORDS,
    ENTITY_STOPWORDS,
)
from .exceptions import (
    APIException,
    ConfigurationError,
    LLMServiceError,
    ModelLoadError,
    RAGServiceError,
)
from .logging_config import configure_logging, get_logger

__all__ = [
    # Constants
    "ENTITY_STOPWORDS",
    "ENTITY_QUESTION_WORDS",
    "DEFAULT_EXCLUDED_VERBS",
    "DEFAULT_SENTIMENT_MODEL",
    "DEFAULT_EMOTION_MODEL",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_RERANKER_MODEL",
    # Logging
    "configure_logging",
    "get_logger",
    # Exceptions
    "APIException",
    "ConfigurationError",
    "ModelLoadError",
    "LLMServiceError",
    "RAGServiceError",
]
