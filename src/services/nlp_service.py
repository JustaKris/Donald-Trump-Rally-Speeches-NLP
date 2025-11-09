"""
NLP Service for text analysis and processing.

Provides word frequency analysis, topic extraction, and other NLP operations.
This service aggregates functionality from utils for cleaner API separation.
"""

import logging
from typing import Any, Dict, List

from ..utils.io_helpers import get_dataset_statistics, get_word_frequency_stats, load_speeches_from_directory
from ..utils.text_preprocessing import extract_ngrams, get_stopwords, tokenize_text

logger = logging.getLogger(__name__)


class NLPService:
    """
    Service for NLP text analysis operations.

    Provides methods for word frequency, topic extraction, n-gram analysis,
    and other text processing tasks.
    """

    def __init__(self):
        """Initialize NLP service."""
        self.stopwords = get_stopwords()
        logger.info("NLP service initialized")

    def analyze_word_frequency(self, text: str, top_n: int = 50) -> Dict[str, Any]:
        """
        Analyze word frequency in the input text.

        Args:
            text: Input text to analyze
            top_n: Number of top words to return

        Returns:
            Dictionary with word frequency statistics
        """
        stats = get_word_frequency_stats(text, top_n=top_n)
        return stats

    def extract_topics(self, text: str, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Extract key topics/themes from text.

        Args:
            text: Input text to analyze
            top_n: Number of topics to extract

        Returns:
            List of topic dictionaries with relevance scores
        """
        stats = get_word_frequency_stats(text, top_n=top_n)

        # Normalize counts to 0-1 range for relevance score
        top_words = stats["top_words"]
        if not top_words:
            return []

        max_count = top_words[0]["count"]

        return [
            {
                "topic": word_data["word"],
                "relevance": round(word_data["count"] / max_count, 3),
                "mentions": word_data["count"],
            }
            for word_data in top_words
        ]

    def extract_ngrams(self, text: str, n: int = 2, top_n: int = 20) -> Dict[str, Any]:
        """
        Extract n-grams from text.

        Args:
            text: Input text
            n: N-gram size (2 for bigrams, 3 for trigrams, etc.)
            top_n: Number of top n-grams to return

        Returns:
            Dictionary with n-gram statistics
        """
        from collections import Counter

        tokens = tokenize_text(text)
        # Remove stopwords for better n-grams
        tokens = [t for t in tokens if t not in self.stopwords and t.isalpha()]

        ngrams = extract_ngrams(tokens, n=n)

        # Count and rank n-grams
        ngram_counts = Counter(ngrams)
        top_ngrams = ngram_counts.most_common(top_n)

        return {
            "n": n,
            "total_ngrams": len(ngrams),
            "unique_ngrams": len(set(ngrams)),
            "top_ngrams": [{"ngram": ngram, "count": count} for ngram, count in top_ngrams],
        }

    def get_dataset_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the speech dataset.

        Returns:
            Dictionary with dataset statistics
        """
        return get_dataset_statistics()

    def list_speeches(self) -> Dict[str, Any]:
        """
        List all speeches in the dataset.

        Returns:
            Dictionary with speech list and count
        """
        df = load_speeches_from_directory()
        speeches = df[["filename", "location", "month", "year", "word_count"]].to_dict("records")
        return {"total": len(speeches), "speeches": speeches}
