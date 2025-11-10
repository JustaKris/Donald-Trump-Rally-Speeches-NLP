"""
Utility functions for the NLP Chatbot API.

This module contains helper functions for text processing, data loading,
formatting, and other common operations.
"""

from .embeddings import batch_embeddings, cosine_similarity, normalize_scores
from .formatters import format_confidence_badge, format_sentiment_badge, format_sources
from .io_helpers import (
    extract_topics,
    get_dataset_statistics,
    get_project_root,
    get_word_frequency_stats,
    load_speeches_from_directory,
)
from .text_preprocessing import (
    calculate_word_frequency,
    chunk_text_for_bert,
    clean_text,
    extract_ngrams,
    get_stopwords,
    tokenize_text,
)

__all__ = [
    # Text preprocessing
    "clean_text",
    "tokenize_text",
    "extract_ngrams",
    "chunk_text_for_bert",
    "calculate_word_frequency",
    "get_stopwords",
    # IO helpers
    "get_project_root",
    "load_speeches_from_directory",
    "get_dataset_statistics",
    "get_word_frequency_stats",
    "extract_topics",
    # Formatters
    "format_sentiment_badge",
    "format_confidence_badge",
    "format_sources",
    # Embeddings
    "cosine_similarity",
    "normalize_scores",
    "batch_embeddings",
]
