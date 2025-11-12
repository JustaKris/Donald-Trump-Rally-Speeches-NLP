"""
Dependency injection for API routes.

Provides reusable dependencies for FastAPI route handlers,
including configuration, services, and model instances.
"""

import logging
from typing import Optional

from fastapi import Depends

from ..core import Settings, get_settings
from ..services import GeminiLLM, NLPService, RAGService, SentimentAnalyzer, TopicExtractionService

logger = logging.getLogger(__name__)

# Global service instances (initialized on startup)
_sentiment_analyzer: Optional[SentimentAnalyzer] = None
_rag_service: Optional[RAGService] = None
_llm_service: Optional[GeminiLLM] = None
_nlp_service: Optional[NLPService] = None
_topic_service: Optional[TopicExtractionService] = None


def set_sentiment_analyzer(analyzer: SentimentAnalyzer) -> None:
    """Set the global sentiment analyzer instance."""
    global _sentiment_analyzer
    _sentiment_analyzer = analyzer


def set_rag_service(service: RAGService) -> None:
    """Set the global RAG service instance."""
    global _rag_service
    _rag_service = service


def set_llm_service(service: Optional[GeminiLLM]) -> None:
    """Set the global LLM service instance."""
    global _llm_service
    _llm_service = service


def set_nlp_service(service: NLPService) -> None:
    """Set the global NLP service instance."""
    global _nlp_service
    _nlp_service = service


def set_topic_service(service: TopicExtractionService) -> None:
    """Set the global topic extraction service instance."""
    global _topic_service
    _topic_service = service


def get_settings_dep() -> Settings:
    """
    Dependency to get application settings.

    Returns:
        Settings instance
    """
    return get_settings()


def get_sentiment_analyzer_dep() -> Optional[SentimentAnalyzer]:
    """
    Dependency to get sentiment analyzer.

    Returns:
        SentimentAnalyzer instance or None if not loaded
    """
    return _sentiment_analyzer


def get_rag_service() -> Optional[RAGService]:
    """
    Dependency to get RAG service.

    Returns:
        RAGService instance or None if not initialized
    """
    return _rag_service


def get_llm_service() -> Optional[GeminiLLM]:
    """
    Dependency to get LLM service.

    Returns:
        GeminiLLM instance or None if not configured
    """
    return _llm_service


def get_nlp_service() -> NLPService:
    """
    Dependency to get NLP service.

    Returns:
        NLPService instance
    """
    if _nlp_service is None:
        # Create on demand if not initialized
        return NLPService()
    return _nlp_service


def get_topic_service() -> Optional[TopicExtractionService]:
    """
    Dependency to get topic extraction service.

    Returns:
        TopicExtractionService instance or None if not initialized
    """
    return _topic_service
