"""
API module for the NLP Chatbot API.

Contains route handlers organized by functionality:
- chatbot: RAG-powered question answering
- nlp: Text analysis endpoints (sentiment, frequency, topics)
- health: Health check and metadata endpoints
"""

from .dependencies import (
    get_llm_service,
    get_nlp_service,
    get_rag_service,
    get_sentiment_analyzer_dep,
    get_settings_dep,
)
from .routes_chatbot import router as chatbot_router
from .routes_health import router as health_router
from .routes_nlp import router as nlp_router

__all__ = [
    # Routers
    "chatbot_router",
    "nlp_router",
    "health_router",
    # Dependencies
    "get_settings_dep",
    "get_sentiment_analyzer_dep",
    "get_rag_service",
    "get_llm_service",
    "get_nlp_service",
]
