"""
Unit tests for utility functions.

Tests data loading, statistics computation, and topic extraction.
"""

import pytest
from src.utils import (
    get_word_frequency_stats,
    extract_topics,
)


class TestWordFrequency:
    """Test suite for word frequency analysis."""

    @pytest.mark.unit
    def test_word_frequency_basic(self):
        """Test basic word frequency computation."""
        text = "hello world hello python world world"
        stats = get_word_frequency_stats(text, top_n=10)

        assert isinstance(stats, dict)
        assert "total_tokens" in stats
        assert "unique_tokens" in stats
        assert "top_words" in stats
        assert isinstance(stats["top_words"], list)

    @pytest.mark.unit
    def test_word_frequency_top_n(self):
        """Test that top_n parameter works."""
        text = " ".join(["word"] * 10 + ["other"] * 5 + ["test"] * 3)
        stats = get_word_frequency_stats(text, top_n=2)

        assert len(stats["top_words"]) <= 2
        # Most frequent word should be first
        if stats["top_words"]:
            first_word = stats["top_words"][0]
            assert "word" in str(first_word).lower()

    @pytest.mark.unit
    def test_word_frequency_empty_text(self):
        """Test word frequency with empty text."""
        stats = get_word_frequency_stats("", top_n=10)
        assert stats["total_tokens"] == 0
        assert len(stats["top_words"]) == 0

    @pytest.mark.unit
    def test_word_frequency_counts(self):
        """Test that word counts are correct."""
        text = "test test test"
        stats = get_word_frequency_stats(text, top_n=10)

        # Should have tokens
        assert stats["total_tokens"] == 3
        assert stats["unique_tokens"] == 1


class TestTopicExtraction:
    """Test suite for topic extraction."""

    @pytest.mark.unit
    def test_extract_topics_basic(self):
        """Test basic topic extraction."""
        text = "economy jobs growth economy employment jobs market economy"
        topics = extract_topics(text, top_n=5)

        assert isinstance(topics, list)
        assert len(topics) > 0
        # Check structure of topic dictionaries
        if topics:
            topic = topics[0]
            assert "topic" in topic
            assert "relevance" in topic
            assert "mentions" in topic

    @pytest.mark.unit
    def test_extract_topics_relevance_scores(self):
        """Test that relevance scores are normalized."""
        text = "test word test word test"
        topics = extract_topics(text, top_n=10)

        for topic in topics:
            assert 0 <= topic["relevance"] <= 1
            # Top topic should have relevance of 1.0
        if topics:
            assert topics[0]["relevance"] == 1.0

    @pytest.mark.unit
    def test_extract_topics_sorted(self):
        """Test that topics are sorted by relevance."""
        text = "economy economy economy jobs jobs market"
        topics = extract_topics(text, top_n=10)

        # Should be sorted by relevance (descending)
        relevances = [t["relevance"] for t in topics]
        assert relevances == sorted(relevances, reverse=True)

    @pytest.mark.unit
    def test_extract_topics_top_n(self):
        """Test that top_n parameter limits results."""
        text = " ".join([f"word{i}" for i in range(20)])
        topics = extract_topics(text, top_n=5)

        assert len(topics) <= 5

    @pytest.mark.unit
    def test_extract_topics_empty_text(self):
        """Test topic extraction with empty text."""
        topics = extract_topics("", top_n=5)
        assert isinstance(topics, list)
        assert len(topics) == 0
