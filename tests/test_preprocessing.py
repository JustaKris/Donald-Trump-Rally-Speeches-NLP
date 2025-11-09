"""
Unit tests for preprocessing utilities.

Tests the text cleaning, tokenization, and n-gram extraction functions.
"""

import pytest
from src.utils.text_preprocessing import (
    clean_text,
    tokenize_text,
    extract_ngrams,
    get_stopwords,
)


class TestTextCleaning:
    """Test suite for text cleaning functions."""

    @pytest.mark.unit
    def test_clean_text_basic(self):
        """Test basic text cleaning."""
        text = "Hello World! This is a test."
        result = clean_text(text, remove_stopwords=False)
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.unit
    def test_clean_text_removes_urls(self):
        """Test that URLs are removed from text."""
        text = "Check out https://example.com for more info!"
        result = clean_text(text, remove_stopwords=False)
        assert "https://" not in result
        assert "example.com" not in result

    @pytest.mark.unit
    def test_clean_text_removes_stopwords(self):
        """Test stopword removal."""
        text = "This is a simple test sentence."
        result = clean_text(text, remove_stopwords=True)
        # Common stopwords should be removed
        assert "this" not in result.lower()
        assert "is" not in result.lower()
        # Content words should remain
        assert "simple" in result.lower() or "test" in result.lower()

    @pytest.mark.unit
    def test_clean_text_lowercases(self):
        """Test that text is lowercased."""
        text = "HELLO WORLD"
        result = clean_text(text, remove_stopwords=False)
        assert result.islower() or result == ""

    @pytest.mark.unit
    def test_clean_text_empty_string(self):
        """Test cleaning empty string."""
        result = clean_text("", remove_stopwords=False)
        assert result == ""


class TestTokenization:
    """Test suite for tokenization functions."""

    @pytest.mark.unit
    def test_tokenize_text_basic(self):
        """Test basic tokenization."""
        text = "Hello world! How are you?"
        tokens = tokenize_text(text)
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(token, str) for token in tokens)

    @pytest.mark.unit
    def test_tokenize_text_splits_words(self):
        """Test that words are properly split."""
        text = "one two three"
        tokens = tokenize_text(text)
        assert "one" in tokens
        assert "two" in tokens
        assert "three" in tokens

    @pytest.mark.unit
    def test_tokenize_empty_string(self):
        """Test tokenizing empty string."""
        tokens = tokenize_text("")
        assert isinstance(tokens, list)
        assert len(tokens) == 0


class TestNGrams:
    """Test suite for n-gram extraction."""

    @pytest.mark.unit
    def test_extract_bigrams(self):
        """Test bigram extraction."""
        tokens = ["the", "quick", "brown", "fox"]
        bigrams = extract_ngrams(tokens, n=2)
        assert isinstance(bigrams, list)
        assert len(bigrams) == 3  # 4 tokens -> 3 bigrams
        assert "the quick" in bigrams or ("the", "quick") in bigrams

    @pytest.mark.unit
    def test_extract_trigrams(self):
        """Test trigram extraction."""
        tokens = ["the", "quick", "brown", "fox"]
        trigrams = extract_ngrams(tokens, n=3)
        assert isinstance(trigrams, list)
        assert len(trigrams) == 2  # 4 tokens -> 2 trigrams

    @pytest.mark.unit
    def test_extract_ngrams_short_input(self):
        """Test n-gram extraction with input shorter than n."""
        tokens = ["hello"]
        bigrams = extract_ngrams(tokens, n=2)
        assert len(bigrams) == 0

    @pytest.mark.unit
    def test_extract_ngrams_empty_input(self):
        """Test n-gram extraction with empty input."""
        ngrams = extract_ngrams([], n=2)
        assert isinstance(ngrams, list)
        assert len(ngrams) == 0


class TestStopwords:
    """Test suite for stopword utilities."""

    @pytest.mark.unit
    def test_get_stopwords(self):
        """Test getting stopwords list."""
        stopwords = get_stopwords()
        assert isinstance(stopwords, set)
        assert len(stopwords) > 0
        # Check for common English stopwords
        assert "the" in stopwords
        assert "is" in stopwords
        assert "and" in stopwords
