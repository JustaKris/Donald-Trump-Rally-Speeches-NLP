"""
Health check and metadata endpoints.

Provides system health status, version information, and configuration details.
"""

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from ..core import Settings
from ..services import EnhancedSentimentAnalyzer, RAGService
from .dependencies import get_rag_service, get_sentiment_analyzer_dep, get_settings_dep

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

# Set up templates
templates_path = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_path))


@router.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main frontend page."""
    html_file = Path(__file__).parent.parent / "templates" / "index.html"
    if html_file.exists():
        return FileResponse(html_file)

    # Fallback HTML if template not found
    return """
    <html>
        <head><title>Trump Speeches NLP Chatbot API</title></head>
        <body>
            <h1>🧠 Trump Speeches NLP Chatbot API</h1>
            <p>Production-ready NLP and RAG platform with AI Q&A, sentiment analysis, and semantic search</p>
            <ul>
                <li><a href="/docs">📚 Interactive API Documentation</a></li>
                <li><a href="/redoc">📖 ReDoc Documentation</a></li>
                <li><a href="/health">🏥 Health Check</a></li>
            </ul>
        </body>
    </html>
    """


@router.get("/health")
async def health_check(
    settings: Settings = Depends(get_settings_dep),
    sentiment_analyzer: Optional[EnhancedSentimentAnalyzer] = Depends(get_sentiment_analyzer_dep),
    rag_service: Optional[RAGService] = Depends(get_rag_service),
):
    """
    Health check endpoint.

    Returns system status and service availability.
    """
    return {
        "status": "healthy",
        "version": settings.app_version,
        "environment": settings.environment,
        "services": {
            "sentiment_analyzer": sentiment_analyzer is not None,
            "rag_service": rag_service is not None,
            "llm_configured": settings.is_llm_configured(),
        },
    }


@router.get("/config")
async def get_config(settings: Settings = Depends(get_settings_dep)):
    """
    Get public configuration information.

    Returns non-sensitive configuration details.
    """
    return {
        "app_name": settings.app_name,
        "app_version": settings.app_version,
        "environment": settings.environment,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.get_llm_model_name() if settings.is_llm_configured() else None,
        "sentiment_model": settings.sentiment_model_name,
        "embedding_model": settings.embedding_model_name,
        "features": {
            "llm_enabled": settings.llm_enabled and settings.is_llm_configured(),
            "hybrid_search": settings.use_hybrid_search,
            "reranking": settings.use_reranking,
        },
    }
