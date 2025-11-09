"""
Trump Speeches NLP Chatbot API - Production-Ready AI Platform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A comprehensive NLP and RAG platform for analyzing political speeches
with AI-powered question answering, sentiment analysis, and semantic search.

Built with FastAPI, ChromaDB, Google Gemini, and state-of-the-art NLP models.
"""

__version__ = "0.2.0"
__author__ = "JustaKris"

from .main import app

__all__ = ["app"]
