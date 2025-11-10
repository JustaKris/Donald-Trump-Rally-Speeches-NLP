"""
Embedding utilities for vector operations.

Provides helper functions for working with embeddings and vector databases.
"""

from typing import List

import numpy as np


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score (0-1)
    """
    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)

    dot_product = np.dot(vec1_np, vec2_np)
    norm1 = np.linalg.norm(vec1_np)
    norm2 = np.linalg.norm(vec2_np)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


def normalize_scores(scores: List[float]) -> List[float]:
    """
    Normalize a list of scores to 0-1 range.

    Args:
        scores: List of raw scores

    Returns:
        Normalized scores
    """
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [0.5] * len(scores)

    return [(s - min_score) / (max_score - min_score) for s in scores]


def batch_embeddings(texts: List[str], batch_size: int = 32) -> List[List[int]]:
    """
    Process texts in batches for embedding generation.

    Args:
        texts: List of texts to process
        batch_size: Batch size for processing

    Returns:
        List of batches (each batch is a list of text indices)
    """
    batches = []
    for i in range(0, len(texts), batch_size):
        batch = list(range(i, min(i + batch_size, len(texts))))
        batches.append(batch)

    return batches
