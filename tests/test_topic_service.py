"""
Tests for enhanced topic extraction service.

Tests semantic clustering, snippet extraction, and AI summary generation.
"""

import pytest

from src.services.topic_service import TopicExtractionService


class TestTopicExtractionService:
    """Test suite for enhanced topic extraction."""

    @pytest.fixture
    def topic_service(self):
        """Create topic extraction service instance."""
        # Initialize without LLM to avoid API calls in tests
        return TopicExtractionService(llm_service=None)

    @pytest.mark.unit
    def test_extract_topics_basic(self, topic_service):
        """Test basic topic extraction with clustering."""
        text = """
        The economy is strong. Jobs are growing. Employment rates are up.
        The border wall is important. Immigration policy needs reform.
        Media coverage is unfair. Fake news attacks constantly.
        """

        result = topic_service.extract_topics_enhanced(text, top_n=5)

        assert "clustered_topics" in result
        assert "snippets" in result
        assert "summary" in result
        assert "metadata" in result

        # Should have multiple clusters
        assert len(result["clustered_topics"]) > 0
        assert result["metadata"]["num_clusters"] > 0

    @pytest.mark.unit
    def test_keywords_extraction(self, topic_service):
        """Test that keywords are extracted correctly."""
        text = "economy economy economy jobs jobs market"

        keywords = topic_service._extract_keywords(text, top_n=10)

        assert len(keywords) > 0
        # Most frequent should be "economy"
        assert keywords[0]["word"] == "economy"
        assert keywords[0]["count"] == 3
        # Positions should be tracked
        assert "positions" in keywords[0]
        assert len(keywords[0]["positions"]) > 0

    @pytest.mark.unit
    def test_clustering_multiple_keywords(self, topic_service):
        """Test clustering with multiple related keywords."""
        keywords_data = [
            {"word": "economy", "count": 10, "relevance": 1.0, "positions": [0, 50, 100]},
            {"word": "jobs", "count": 8, "relevance": 0.8, "positions": [10, 60, 110]},
            {"word": "employment", "count": 7, "relevance": 0.7, "positions": [20, 70, 120]},
            {"word": "border", "count": 9, "relevance": 0.9, "positions": [30, 80, 130]},
            {"word": "wall", "count": 6, "relevance": 0.6, "positions": [40, 90, 140]},
        ]

        clusters = topic_service._cluster_keywords(keywords_data, num_clusters=2)

        assert len(clusters) == 2
        # Each cluster should have keywords
        for cluster in clusters:
            assert "keywords" in cluster
            assert "avg_relevance" in cluster
            assert "total_mentions" in cluster
            assert len(cluster["keywords"]) > 0

    @pytest.mark.unit
    def test_snippet_extraction(self, topic_service):
        """Test contextual snippet extraction."""
        text = "The economy is strong and jobs are plentiful. We have the best economy in history."
        position = 4  # Position of "economy"
        keyword = "economy"

        snippet = topic_service._extract_snippet(text, position, keyword, context_chars=20)

        assert snippet is not None
        assert keyword in snippet.lower() or "**economy**" in snippet.lower()

    @pytest.mark.unit
    def test_deduplicate_positions(self, topic_service):
        """Test position deduplication."""
        positions = [(0, "word"), (10, "word"), (200, "word"), (210, "word"), (500, "word")]

        deduplicated = topic_service._deduplicate_positions(positions, min_distance=100)

        # Should remove nearby positions
        assert len(deduplicated) < len(positions)
        # Should keep positions that are far apart
        assert (0, "word") in deduplicated
        assert (200, "word") in deduplicated
        assert (500, "word") in deduplicated

    @pytest.mark.unit
    def test_empty_text_handling(self, topic_service):
        """Test handling of empty text."""
        result = topic_service.extract_topics_enhanced("", top_n=10)

        assert result["clustered_topics"] == []
        assert result["snippets"] == []
        assert result["summary"] is None
        assert result["metadata"]["num_clusters"] == 0

    @pytest.mark.unit
    def test_short_text_handling(self, topic_service):
        """Test handling of very short text."""
        result = topic_service.extract_topics_enhanced("hello world", top_n=10)

        # Should return some result even for short text
        assert "clustered_topics" in result
        assert "metadata" in result

    @pytest.mark.unit
    def test_metadata_accuracy(self, topic_service):
        """Test that metadata is accurate."""
        text = "economy jobs market economy employment jobs"

        result = topic_service.extract_topics_enhanced(text, top_n=3)

        metadata = result["metadata"]
        assert metadata["text_length"] == len(text)
        assert metadata["num_clusters"] <= 3  # Requested top_n
        assert metadata["has_ai_summary"] is False  # No LLM configured

    @pytest.mark.integration
    @pytest.mark.requires_model
    def test_full_pipeline_with_real_text(self, topic_service):
        """Test complete pipeline with realistic text."""
        text = """
        They're burning Minneapolis. You don't think of Minneapolis that way, right?
        The city is burning down. You have this fake CNN reporter. This is a friendly protest.
        It's a mostly genteel protest. And it's really quite nice. Now people are shooting
        bullets at him. He's being hit with tear gas. The economy is great. Jobs are booming.
        Employment is at record highs. We need to protect our borders. The wall is being built.
        Immigration reform is necessary.
        """

        result = topic_service.extract_topics_enhanced(text, top_n=5, snippets_per_topic=2)

        # Should have multiple clusters
        assert len(result["clustered_topics"]) > 0

        # Each cluster should have label and keywords
        for cluster in result["clustered_topics"]:
            assert "label" in cluster
            assert "keywords" in cluster
            assert len(cluster["keywords"]) > 0

        # Should have snippets
        assert len(result["snippets"]) > 0

        # Snippets should match clusters
        assert len(result["snippets"]) == len(result["clustered_topics"])
