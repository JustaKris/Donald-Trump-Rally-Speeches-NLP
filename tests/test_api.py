"""
Integration tests for FastAPI endpoints.

Tests the API routes using FastAPI's TestClient.
"""

import pytest
from fastapi.testclient import TestClient

from src.main import app


@pytest.fixture
def client():
    """Create a test client for the API."""
    return TestClient(app)


class TestHealthEndpoint:
    """Test suite for health check endpoint."""

    @pytest.mark.integration
    def test_health_check(self, client):
        """Test health check endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"


class TestSentimentEndpoint:
    """Test suite for AI-powered sentiment analysis endpoint."""

    @pytest.mark.integration
    @pytest.mark.requires_model
    def test_sentiment_analysis_valid_input(self, client):
        """Test enhanced sentiment analysis with valid input."""
        payload = {"text": "This is a great day! I love it!"}
        response = client.post("/analyze/sentiment", json=payload)

        # Should succeed or return 503 if models not loaded
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            # Check basic sentiment fields
            assert "sentiment" in data
            assert "confidence" in data
            assert data["sentiment"] in ["positive", "negative", "neutral"]
            assert 0 <= data["confidence"] <= 1

            # Check enhanced fields
            assert "scores" in data
            assert "positive" in data["scores"]
            assert "negative" in data["scores"]
            assert "neutral" in data["scores"]

            # Check emotion detection
            assert "emotions" in data
            assert isinstance(data["emotions"], dict)
            assert len(data["emotions"]) > 0

            # Check contextual interpretation
            assert "contextual_sentiment" in data
            assert isinstance(data["contextual_sentiment"], str)
            assert len(data["contextual_sentiment"]) > 0

            # Check metadata
            assert "num_chunks" in data
            assert data["num_chunks"] >= 1

    @pytest.mark.integration
    def test_sentiment_analysis_empty_text(self, client):
        """Test sentiment analysis with empty text."""
        payload = {"text": ""}
        response = client.post("/analyze/sentiment", json=payload)
        # Should return validation error
        assert response.status_code == 422

    @pytest.mark.integration
    def test_sentiment_analysis_missing_text(self, client):
        """Test sentiment analysis with missing text field."""
        response = client.post("/analyze/sentiment", json={})
        assert response.status_code == 422


class TestWordFrequencyEndpoint:
    """Test suite for word frequency endpoint."""

    @pytest.mark.integration
    def test_word_frequency_valid_input(self, client):
        """Test word frequency analysis with valid input."""
        payload = {"text": "hello world hello python world world"}
        response = client.post("/analyze/words", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "total_tokens" in data
        assert "unique_tokens" in data
        assert "top_words" in data
        assert isinstance(data["top_words"], list)

    @pytest.mark.integration
    def test_word_frequency_top_n_parameter(self, client):
        """Test word frequency with top_n parameter."""
        payload = {"text": "word " * 20}
        response = client.post("/analyze/words?top_n=5", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert len(data["top_words"]) <= 5

    @pytest.mark.integration
    def test_word_frequency_empty_text(self, client):
        """Test word frequency with empty text."""
        payload = {"text": ""}
        response = client.post("/analyze/words", json=payload)
        assert response.status_code == 422


class TestTopicEndpoint:
    """Test suite for topic extraction endpoint."""

    @pytest.mark.integration
    def test_topic_extraction_valid_input(self, client):
        """Test topic extraction with valid input."""
        payload = {"text": "economy jobs market growth employment"}
        response = client.post("/analyze/topics", json=payload)

        # May return 503 if topic service not initialized (no LLM)
        if response.status_code == 503:
            pytest.skip("Topic service not initialized in test environment")

        assert response.status_code == 200
        data = response.json()
        # Enhanced response has clustered_topics, snippets, summary
        assert "clustered_topics" in data
        assert "snippets" in data
        assert "metadata" in data
        assert isinstance(data["clustered_topics"], list)

    @pytest.mark.integration
    def test_topic_extraction_top_n_parameter(self, client):
        """Test topic extraction with top_n parameter."""
        payload = {"text": "word " * 20}
        response = client.post("/analyze/topics?top_n=3", json=payload)

        # May return 503 if topic service not initialized
        if response.status_code == 503:
            pytest.skip("Topic service not initialized in test environment")

        assert response.status_code == 200
        data = response.json()
        assert len(data["clustered_topics"]) <= 3


class TestNGramEndpoint:
    """Test suite for n-gram extraction endpoint."""

    @pytest.mark.integration
    def test_ngram_extraction_bigrams(self, client):
        """Test n-gram extraction for bigrams."""
        payload = {"text": "the quick brown fox jumps", "n": 2, "top_n": 10}
        response = client.post("/analyze/ngrams", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "n" in data
        assert data["n"] == 2
        assert "top_ngrams" in data
        assert isinstance(data["top_ngrams"], list)

    @pytest.mark.integration
    def test_ngram_extraction_trigrams(self, client):
        """Test n-gram extraction for trigrams."""
        payload = {"text": "one two three four five", "n": 3, "top_n": 10}
        response = client.post("/analyze/ngrams", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["n"] == 3

    @pytest.mark.integration
    def test_ngram_invalid_n_value(self, client):
        """Test n-gram extraction with invalid n value."""
        payload = {"text": "test text", "n": 10, "top_n": 10}
        response = client.post("/analyze/ngrams", json=payload)
        # n > 5 should be rejected by validation
        assert response.status_code == 422


class TestStatisticsEndpoint:
    """Test suite for dataset statistics endpoint."""

    @pytest.mark.integration
    def test_dataset_statistics(self, client):
        """Test getting dataset statistics."""
        response = client.get("/analyze/speeches/stats")

        assert response.status_code == 200
        data = response.json()
        assert "total_speeches" in data
        assert "total_words" in data
        assert "avg_words_per_speech" in data
        assert isinstance(data["total_speeches"], int)


class TestSpeechListEndpoint:
    """Test suite for speech listing endpoint."""

    @pytest.mark.integration
    def test_list_speeches(self, client):
        """Test listing all speeches."""
        response = client.get("/analyze/speeches/list")

        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "speeches" in data
        assert isinstance(data["speeches"], list)


class TestTextCleanEndpoint:
    """Test suite for text cleaning endpoint."""

    @pytest.mark.integration
    def test_clean_text_basic(self, client):
        """Test text cleaning endpoint."""
        payload = {"text": "Hello World! https://example.com"}
        response = client.post("/analyze/clean", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "cleaned_text" in data
        assert "original_length" in data
        assert "cleaned_length" in data

    @pytest.mark.integration
    def test_clean_text_with_stopwords_param(self, client):
        """Test text cleaning with remove_stopwords parameter."""
        payload = {"text": "This is a test"}
        response = client.post("/analyze/clean?remove_stopwords=true", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "cleaned_text" in data
